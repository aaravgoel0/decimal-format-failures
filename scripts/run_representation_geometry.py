#!/usr/bin/env python3
"""Held-out representation-geometry analysis."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

MODELS = ["meta-llama-Meta-Llama-3.1-8B-Instruct", "Qwen-Qwen3-4B-Instruct-2507", "google-gemma-2-9b-it"]
FORMS = ["canonical", "padded_1", "padded_2", "near_minus", "near_plus"]
BOOTSTRAPS, SEED = 10_000, 73_201


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def cka(x, y):
    return cka_grams(x @ x.T, y @ y.T)


def cka_grams(k, l):
    k = k - k.mean(0)[None, :] - k.mean(1)[:, None] + k.mean()
    l = l - l.mean(0)[None, :] - l.mean(1)[:, None] + l.mean()
    return float((k * l).sum() / np.sqrt((k * k).sum() * (l * l).sum() + 1e-24))


def nuisance(rows, meta, indices):
    return np.asarray([[1, meta[i]["target_token_count"], meta[i]["numeral_final_position"],
                        rows[i]["target_position"], rows[i]["whole"], rows[i]["digit"]]
                       for i in indices], dtype=np.float64)


def statistic(mats):
    canonical = mats["canonical"]
    eq = np.mean([[cosine(a, b) for a, b in zip(canonical, mats[f])] for f in ("padded_1", "padded_2")], axis=0)
    near = np.mean([[cosine(a, b) for a, b in zip(canonical, mats[f])] for f in ("near_minus", "near_plus")], axis=0)
    rsa = float(spearmanr(np.r_[np.ones(len(eq)), np.zeros(len(near))], np.r_[eq, near]).statistic)
    cka_eq = float(np.mean([cka(canonical, mats[f]) for f in ("padded_1", "padded_2")]))
    cka_near = float(np.mean([cka(canonical, mats[f]) for f in ("near_minus", "near_plus")]))
    return float(np.mean(eq - near)), rsa, cka_eq, cka_near


def interval(values):
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))]


def main():
    rows = [json.loads(line) for line in open("data/mechanistic_values.jsonl")]
    output = []
    for model_number, model in enumerate(MODELS):
        acts = np.load("activations/" + model + ".npy", mmap_mode="r")
        meta = [json.loads(line) for line in open("activations/" + model + ".jsonl")]
        for activation_position, position_name in ((0, "numeral_final"), (1, "answer")):
            for target_position in (1, 2):
                train = np.asarray([i for i, r in enumerate(rows) if r["split"] == "train" and r["target_position"] == target_position])
                test = np.asarray([i for i, r in enumerate(rows) if r["split"] == "test" and r["target_position"] == target_position])
                lookup = {rows[i]["id"]: j for j, i in enumerate(test)}
                keys = [(whole, digit) for whole in range(51, 61) for digit in (1, 4, 7)]
                rng = np.random.default_rng(SEED + 100 * model_number + 10 * activation_position + target_position)
                bootstrap_indices = rng.integers(0, len(keys), size=(BOOTSTRAPS, len(keys)))
                curves = []
                for layer in range(acts.shape[1]):
                    x_train = np.asarray(acts[train, layer, activation_position], dtype=np.float32)
                    x_test = np.asarray(acts[test, layer, activation_position], dtype=np.float32)
                    beta = np.linalg.lstsq(nuisance(rows, meta, train), x_train, rcond=None)[0]
                    residual = x_test - nuisance(rows, meta, test) @ beta
                    mats = {form: np.stack([residual[lookup[f"mech-{whole:02d}-{digit}-{target_position}-{form}"]]
                                             for whole, digit in keys]) for form in FORMS}
                    effect, rsa, cka_eq, cka_near = statistic(mats)
                    canonical = mats["canonical"]
                    eq_values = np.mean([[cosine(a, b) for a, b in zip(canonical, mats[f])]
                                         for f in ("padded_1", "padded_2")], axis=0)
                    near_values = np.mean([[cosine(a, b) for a, b in zip(canonical, mats[f])]
                                           for f in ("near_minus", "near_plus")], axis=0)
                    # The prespecified claim criterion is the paired equivalence-minus-nearby
                    # effect; bootstrap it over numerical values, preserving all forms.
                    boot_effect = np.mean(eq_values[bootstrap_indices] - near_values[bootstrap_indices], axis=1)
                    curves.append({"layer": layer, "equivalence_minus_nearby_cosine": effect,
                                   "cosine_bootstrap_95_ci": interval(boot_effect),
                                   "residualized_rsa": rsa,
                                   "linear_cka_equivalent": cka_eq, "linear_cka_nearby": cka_near,
                                   "cka_difference": cka_eq - cka_near})
                output.append({"model": model, "activation_position": position_name,
                               "target_position": target_position, "n_held_out_values": len(keys),
                               "bootstrap_repetitions": BOOTSTRAPS, "layers": curves})
                print(model, position_name, "target", target_position, "done", flush=True)
    Path("results/representation_geometry.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
