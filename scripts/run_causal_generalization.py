#!/usr/bin/env python3
"""Fixed-site causal generalization and incompatible-donor interventions."""
import argparse
import csv
import json
import random
import re
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask

SYSTEM = "You are a helpful assistant that compares numbers."


def formatted_ids(tokenizer, user, template_mode):
    messages = ([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
                if template_mode == "system-role" else
                [{"role": "user", "content": SYSTEM + "\n\n" + user}])
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors=None)


def hits(sequence, subsequence):
    found = [i for i in range(len(sequence) - len(subsequence) + 1)
             if sequence[i:i + len(subsequence)] == subsequence]
    if not found:
        raise RuntimeError(f"numeral token span not found: {subsequence}")
    return found


def state_at(model, ids, selected_layer):
    h = model.model.embed_tokens(mx.array([ids]))
    if getattr(model, "model_type", "") == "gemma2":
        h = h * (model.args.hidden_size ** .5)
        mask = create_attention_mask(h, None, return_array=True)
    else:
        mask = create_attention_mask(h, None)
    for layer_index, layer in enumerate(model.model.layers):
        h = layer(h, mask, None)
        if layer_index == selected_layer:
            mx.eval(h)
            return h, mask
    raise RuntimeError("selected layer outside model")


def final_logits(model, h, mask, selected_layer):
    for layer in model.model.layers[selected_layer + 1:]:
        h = layer(h, mask, None)
    h = model.model.norm(h)
    logits = (model.model.embed_tokens.as_linear(h)
              if getattr(model.args, "tie_word_embeddings", True) else model.lm_head(h))
    if getattr(model, "model_type", "") == "gemma2":
        logits = mx.tanh(logits / model.final_logit_softcapping) * model.final_logit_softcapping
    logits = logits[:, -1]
    mx.eval(logits)
    return logits


def margin(logits, answer_ids):
    value = logits[:, answer_ids[3]] - mx.maximum(logits[:, answer_ids[1]], logits[:, answer_ids[2]])
    return float(np.asarray(value.astype(mx.float32))[0])


def replace(target, source, mappings):
    pieces, last = [], 0
    for target_pos, source_pos in sorted(mappings):
        pieces.extend((target[:, last:target_pos, :], source[:, source_pos:source_pos + 1, :]))
        last = target_pos + 1
    pieces.append(target[:, last:, :])
    return mx.concatenate(pieces, axis=1)


def mapping(target_start, source_start, count):
    return [(target_start + offset, source_start + offset) for offset in range(count)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--layer", required=True, type=int)
    args = parser.parse_args()
    rows = [json.loads(line) for line in open("data/causal_generalization.jsonl")]
    model, tokenizer = load(args.model, revision=args.revision)
    template_mode = "system-role"
    try:
        tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM},
                                       {"role": "user", "content": "test"}],
                                      tokenize=False, add_generation_prompt=True)
    except Exception:
        template_mode = "system-prepended-to-user"
    answer_ids = {choice: tokenizer.encode(str(choice), add_special_tokens=False)[0] for choice in (1, 2, 3)}
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.model)
    output_path = Path(f"results/causal_generalization_{safe}.csv")
    output = list(csv.DictReader(output_path.open())) if output_path.exists() else []
    completed = {int(row["case"]) for row in output
                 if sum(int(other["case"]) == int(row["case"]) for other in output) == 6}

    for row in rows:
        case = row["case"]
        if case in completed:
            print(f"{case + 1}/{len(rows)} already complete", flush=True)
            continue
        easy_ids = formatted_ids(tokenizer, row["easy_prompt"], template_mode)
        hard_ids = formatted_ids(tokenizer, row["hard_prompt"], template_mode)
        donor_ids = formatted_ids(tokenizer, row["donor_prompt"], template_mode)
        if not (len(easy_ids) == len(hard_ids) == len(donor_ids)):
            raise RuntimeError(f"unaligned sequence length in {row['id']}: {len(easy_ids)}, {len(hard_ids)}, {len(donor_ids)}")

        short_ids = tokenizer.encode(row["short"], add_special_tokens=False)
        padded_ids = tokenizer.encode(row["padded"], add_special_tokens=False)
        donor_short_ids = tokenizer.encode(f"{row['whole']}.{row['donor_digit']}", add_special_tokens=False)
        if len(donor_short_ids) != len(short_ids):
            raise RuntimeError(f"donor short token count differs in {row['id']}")

        easy_short, easy_padded = min(hits(easy_ids, short_ids)), max(hits(easy_ids, padded_ids))
        hard_padded, hard_short = min(hits(hard_ids, padded_ids)), max(hits(hard_ids, short_ids))
        donor_padded, donor_short = min(hits(donor_ids, padded_ids)), max(hits(donor_ids, donor_short_ids))
        easy_numeral = mapping(hard_padded, easy_padded, len(padded_ids)) + mapping(hard_short, easy_short, len(short_ids))
        donor_numeral = mapping(hard_padded, donor_padded, len(padded_ids)) + mapping(hard_short, donor_short, len(short_ids))
        target_numeral_positions = {position for position, _ in easy_numeral}
        eligible = [position for position in range(1, len(hard_ids) - 1) if position not in target_numeral_positions]
        random_positions = random.Random(81_000 + case).sample(eligible, len(easy_numeral))
        easy_random = [(position, position) for position in random_positions]
        donor_random = [(position, position) for position in random_positions]

        easy_state, _ = state_at(model, easy_ids, args.layer)
        hard_state, hard_mask = state_at(model, hard_ids, args.layer)
        donor_state, _ = state_at(model, donor_ids, args.layer)
        hard_margin = margin(final_logits(model, hard_state, hard_mask, args.layer), answer_ids)
        easy_margin = margin(final_logits(model, easy_state, hard_mask, args.layer), answer_ids)
        donor_margin = margin(final_logits(model, donor_state, hard_mask, args.layer), answer_ids)
        controls = (
            ("easy_number_tokens", easy_state, easy_numeral),
            ("easy_random_positions", easy_state, easy_random),
            ("easy_answer_position", easy_state, [(len(hard_ids) - 1, len(easy_ids) - 1)]),
            ("donor_number_tokens", donor_state, donor_numeral),
            ("donor_random_positions", donor_state, donor_random),
            ("donor_answer_position", donor_state, [(len(hard_ids) - 1, len(donor_ids) - 1)]),
        )
        case_rows = []
        for control, source_state, mappings in controls:
            patched = replace(hard_state, source_state, mappings)
            patched_margin = margin(final_logits(model, patched, hard_mask, args.layer), answer_ids)
            case_rows.append({
                "id": row["id"], "case": case, "template": row["template"],
                "whole": row["whole"], "digit": row["digit"], "zeros": row["zeros"],
                "control": control, "layer": args.layer, "model": args.model,
                "revision": args.revision, "template_mode": template_mode,
                "sequence_length": len(hard_ids), "numeral_token_count": len(easy_numeral),
                "hard_margin": hard_margin, "easy_margin": easy_margin,
                "donor_margin": donor_margin, "patched_margin": patched_margin,
                "margin_effect": patched_margin - hard_margin,
                "hard_correct": hard_margin > 0, "patched_correct": patched_margin > 0,
                "target_positions": json.dumps(sorted(target_numeral_positions)),
                "source_positions": json.dumps([source for _, source in mappings]),
                "random_positions": json.dumps(random_positions),
            })
        with output_path.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=case_rows[0])
            if file.tell() == 0:
                writer.writeheader()
            writer.writerows(case_rows)
        output.extend(case_rows)
        del easy_state, hard_state, donor_state
        mx.clear_cache()
        print(f"{case + 1}/{len(rows)}", flush=True)
    print(output_path, len(output))


if __name__ == "__main__":
    main()
