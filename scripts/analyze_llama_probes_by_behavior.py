#!/usr/bin/env python3
"""Break down held-out Llama probe tests by its exact greedy yes/no decision."""
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from scipy.stats import rankdata

from finalize_probe_inference import bal, corr, design, indices

MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SAFE = "meta-llama-Meta-Llama-3.1-8B-Instruct"
REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
BOOTSTRAPS, SEED = 10_000, 73_193


def subgroup_metric(task, truth, prediction):
    if len(truth) < 2:
        return None
    if task == "value":
        return corr(truth, prediction)
    if not ({-1, 1} <= set(truth.tolist())):
        return None
    return bal(truth, prediction)


def main():
    rows = [json.loads(line) for line in open("data/mechanistic_values.jsonl")]
    acts = np.load(f"activations/{SAFE}.npy", mmap_mode="r")
    model, tokenizer = load(MODEL, revision=REVISION)
    test = np.asarray([i for i, row in enumerate(rows) if row["split"] == "test"])
    decisions = {}
    for start in range(0, len(test), 25):
        ix = test[start:start + 25]
        h = mx.array(np.asarray(acts[ix, -1, 1], dtype=np.float32))[:, None, :]
        h = model.model.norm(h)
        logits = model.model.embed_tokens.as_linear(h) if model.args.tie_word_embeddings else model.lm_head(h)
        token_ids = np.asarray(mx.argmax(logits[:, -1], axis=-1))
        mx.eval(logits)
        for row_index, token_id in zip(ix, token_ids):
            text = tokenizer.decode([int(token_id)]).strip().lower().strip(".,:;!?")
            predicted = True if text == "yes" else False if text == "no" else None
            truth = bool(rows[row_index]["is_equivalent"])
            decisions[int(row_index)] = {"id": rows[row_index]["id"], "token_id": int(token_id),
                                         "decoded_first_token": text, "prediction_equivalent": predicted,
                                         "correct": predicted == truth if predicted is not None else False,
                                         "parse_status": "ok" if predicted is not None else "unparsed"}
    del model
    decision_rows = [decisions[int(i)] for i in test]
    assert len(decision_rows) == 300
    Path("results/llama_mechanistic_behavior.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in decision_rows))

    specs = [s for s in json.load(open("results/cross_format_probe_inference.json")) if s["model"] == SAFE]
    rng = np.random.default_rng(SEED)
    output = []
    for spec in specs:
        train_forms = {"canonical"} if spec["direction"].startswith("canonical") else {"padded_1", "padded_2"}
        test_forms = {"padded_1", "padded_2"} if spec["direction"].startswith("canonical") else {"canonical"}
        near = spec["task"] == "equality"
        tr = indices(rows, "train", train_forms, near)
        te = indices(rows, "test", test_forms, near)
        position = 0 if spec["position"] == "numeral_final" else 1
        x = np.asarray(acts[tr, spec["selected_layer"], position], np.float32)
        z = np.asarray(acts[te, spec["selected_layer"], position], np.float32)
        x, z, cross, gram = design(x, z)
        if spec["task"] == "value":
            raw = np.asarray([rows[i]["value"] for i in tr]); mu, sd = raw.mean(), raw.std()
            y = (raw - mu) / sd; truth = np.asarray([rows[i]["value"] for i in te])
            prediction = cross @ np.linalg.solve(gram + spec["alpha"] * np.eye(len(tr)), y) * sd + mu
        else:
            y = np.asarray([1 if rows[i]["is_equivalent"] else -1 for i in tr], float)
            truth = np.asarray([1 if rows[i]["is_equivalent"] else -1 for i in te])
            prediction = cross @ np.linalg.solve(gram + spec["alpha"] * np.eye(len(tr)), y)
        behavior_correct = np.asarray([decisions[int(i)]["correct"] for i in te])
        groups = []
        for label, mask in (("correct", behavior_correct), ("incorrect", ~behavior_correct)):
            observed = subgroup_metric(spec["task"], truth[mask], prediction[mask])
            value_keys = list(dict.fromkeys((rows[i]["whole"], rows[i]["digit"]) for i in te if mask[list(te).index(i)]))
            boots = []
            for _ in range(BOOTSTRAPS):
                sampled = [value_keys[j] for j in rng.integers(0, len(value_keys), len(value_keys))] if value_keys else []
                selected = np.concatenate([np.where(mask & np.asarray([(rows[i]["whole"], rows[i]["digit"]) == key for i in te]))[0]
                                           for key in sampled]) if sampled else np.asarray([], dtype=int)
                metric = subgroup_metric(spec["task"], truth[selected], prediction[selected])
                if metric is not None and np.isfinite(metric):
                    boots.append(metric)
            groups.append({"behavior_group": label, "n_rows": int(mask.sum()),
                           "n_numerical_values": len(value_keys), "metric": observed,
                           "bootstrap_95_ci": ([float(np.quantile(boots, .025)), float(np.quantile(boots, .975))]
                                                if boots else None)})
        output.append({k: spec[k] for k in ("model", "position", "direction", "task", "selected_layer", "alpha")}
                      | {"behavior_definition": "exact greedy first response token parsed as yes/no",
                         "groups": groups})
    Path("results/llama_probe_behavior_breakdown.json").write_text(json.dumps(output, indent=2) + "\n")
    print("behavior rows", len(decision_rows), "parsed", sum(r["parse_status"] == "ok" for r in decision_rows))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
