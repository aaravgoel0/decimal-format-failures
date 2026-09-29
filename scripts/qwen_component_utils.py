#!/usr/bin/env python3
"""Shared exact-intervention helpers for the Qwen component experiment."""

import math
import random

import mlx.core as mx
import numpy as np
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.qwen3 import scaled_dot_product_attention

from run_causal_generalization import formatted_ids, hits, mapping, replace


LAYER = 2
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"


def input_at(model, ids):
    h = model.model.embed_tokens(mx.array([ids]))
    mask = create_attention_mask(h, None)
    for layer in model.model.layers[:LAYER]:
        h = layer(h, mask, None)
    mx.eval(h)
    return h, mask


def layer_parts(layer, x, mask):
    attention = layer.self_attn
    normalized = layer.input_layernorm(x)
    batch, length, _ = normalized.shape
    queries = attention.q_proj(normalized)
    keys = attention.k_proj(normalized)
    values = attention.v_proj(normalized)
    queries = attention.q_norm(
        queries.reshape(batch, length, attention.n_heads, -1)).transpose(0, 2, 1, 3)
    keys = attention.k_norm(
        keys.reshape(batch, length, attention.n_kv_heads, -1)).transpose(0, 2, 1, 3)
    values = values.reshape(batch, length, attention.n_kv_heads, -1).transpose(0, 2, 1, 3)
    queries = attention.rope(queries)
    keys = attention.rope(keys)
    head_output = scaled_dot_product_attention(
        queries, keys, values, cache=None, scale=attention.scale, mask=mask)
    head_output = head_output.transpose(0, 2, 1, 3)
    attention_output = attention.o_proj(head_output.reshape(batch, length, -1))
    post_attention = x + attention_output
    mlp_output = layer.mlp(layer.post_attention_layernorm(post_attention))
    block_output = post_attention + mlp_output
    mx.eval(head_output, mlp_output, block_output)
    return {
        "input": x,
        "head_output": head_output,
        "post_attention": post_attention,
        "mlp_output": mlp_output,
        "block_output": block_output,
    }


def patch_heads(target, source, mappings, heads):
    head_count = target.shape[2]
    selected = set(heads)
    mask = mx.array([index in selected for index in range(head_count)]).reshape(1, 1, head_count, 1)
    pieces, last = [], 0
    for target_position, source_position in sorted(mappings):
        pieces.append(target[:, last:target_position, :, :])
        target_token = target[:, target_position:target_position + 1, :, :]
        source_token = source[:, source_position:source_position + 1, :, :]
        pieces.append(mx.where(mask, source_token, target_token))
        last = target_position + 1
    pieces.append(target[:, last:, :, :])
    return mx.concatenate(pieces, axis=1)


def intervened_output(layer, hard, source, mappings, kind, heads=None):
    if kind == "whole_block":
        return replace(hard["block_output"], source["block_output"], mappings)
    if kind == "mlp":
        patched_mlp = replace(hard["mlp_output"], source["mlp_output"], mappings)
        return hard["post_attention"] + patched_mlp
    if kind in ("attention", "heads"):
        selected = range(hard["head_output"].shape[2]) if kind == "attention" else heads
        patched_heads = patch_heads(hard["head_output"], source["head_output"], mappings, selected)
        batch, length, head_count, head_dim = patched_heads.shape
        attention_output = layer.self_attn.o_proj(
            patched_heads.reshape(batch, length, head_count * head_dim))
        post_attention = hard["input"] + attention_output
        mlp_output = layer.mlp(layer.post_attention_layernorm(post_attention))
        return post_attention + mlp_output
    raise ValueError(kind)


def selected_logits(model, layer_outputs, mask, ordered_ids, batch_size):
    values = []
    index = mx.array(ordered_ids)
    for start in range(0, len(layer_outputs), batch_size):
        h = mx.concatenate(layer_outputs[start:start + batch_size], axis=0)
        for layer in model.model.layers[LAYER + 1:]:
            h = layer(h, mask, None)
        h = model.model.norm(h)[:, -1, :]
        weights = model.model.embed_tokens.weight[index]
        logits = (h @ weights.T).astype(mx.float32)
        mx.eval(logits)
        correct = logits[:, 2]
        alternative = mx.maximum(logits[:, 0], logits[:, 1])
        margins = np.asarray((correct - alternative).astype(mx.float32), dtype=float)
        values.extend(float(value) for value in margins)
        del h, logits
        mx.clear_cache()
    return values


def unique_random_sets(eligible, size, seed, count):
    if math.comb(len(eligible), size) < count:
        raise RuntimeError("insufficient random-position subsets")
    rng = random.Random(seed)
    seen = set()
    while len(seen) < count:
        seen.add(tuple(sorted(rng.sample(eligible, size))))
    return sorted(seen)


def prepare_case(model, tokenizer, row, template_mode):
    easy_ids = formatted_ids(tokenizer, row["easy_prompt"], template_mode)
    hard_ids = formatted_ids(tokenizer, row["hard_prompt"], template_mode)
    if len(easy_ids) != len(hard_ids):
        raise RuntimeError(f"unaligned sequence length: {row['id']}")
    short_ids = tokenizer.encode(row["short"], add_special_tokens=False)
    padded_ids = tokenizer.encode(row["padded"], add_special_tokens=False)
    easy_short = min(hits(easy_ids, short_ids))
    easy_padded = max(hits(easy_ids, padded_ids))
    hard_padded = min(hits(hard_ids, padded_ids))
    hard_short = max(hits(hard_ids, short_ids))
    mappings = (mapping(hard_short, easy_short, len(short_ids)) +
                mapping(hard_padded, easy_padded, len(padded_ids)))
    numeral_positions = {target for target, _ in mappings}
    eligible = [position for position in range(1, len(hard_ids) - 1)
                if position not in numeral_positions]
    easy_input, _ = input_at(model, easy_ids)
    hard_input, hard_mask = input_at(model, hard_ids)
    layer = model.model.layers[LAYER]
    easy = layer_parts(layer, easy_input, hard_mask)
    hard = layer_parts(layer, hard_input, hard_mask)
    return {
        "easy_ids": easy_ids,
        "hard_ids": hard_ids,
        "short_ids": short_ids,
        "padded_ids": padded_ids,
        "mappings": mappings,
        "eligible": eligible,
        "mask": hard_mask,
        "easy": easy,
        "hard": hard,
        "layer": layer,
    }
