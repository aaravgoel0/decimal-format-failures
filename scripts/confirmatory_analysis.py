#!/usr/bin/env python3
"""Execute the held-out confirmatory statistical analysis."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import binomtest, chi2, fisher_exact

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FILES = {
    "Llama 3.1 8B": RESULTS / "llama3.1-8b__confirmatory_zero_padding__p0.jsonl",
    "Qwen3 4B Instruct 2507": RESULTS / "Qwen-Qwen3-4B-Instruct-2507__confirmatory_zero_padding__p0__mlx.jsonl",
    "Gemma 2 9B": RESULTS / "gemma2-9b__confirmatory_zero_padding__p0.jsonl",
}


def wilson(k, n, z=1.959963984540054):
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(center - radius), float(center + radius)


def holm(pairs):
    ordered = sorted(pairs, key=lambda x: x[1])
    adjusted = {}
    running = 0.0
    m = len(ordered)
    for rank, (name, p) in enumerate(ordered):
        value = min(1.0, (m - rank) * p)
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def bootstrap_position_difference(frame, iterations=10000):
    rng = np.random.default_rng(8192026)
    cells = sorted(set(zip(frame.digit, frame.zeros)))
    cell_differences = []
    for digit, zeros in cells:
        cell = frame[(frame.digit == digit) & (frame.zeros == zeros)]
        first = cell[cell.padded_position == 1].correct.mean()
        second = cell[cell.padded_position == 2].correct.mean()
        cell_differences.append(first - second)
    cell_differences = np.asarray(cell_differences)
    indices = rng.integers(0, len(cells), size=(iterations, len(cells)))
    diffs = cell_differences[indices].mean(axis=1)
    return tuple(float(x) for x in np.quantile(diffs, [0.025, 0.975]))


def cell_position_differences(frame):
    values = []
    for digit in range(10):
        for zeros in range(1, 6):
            cell = frame[(frame.digit == digit) & (frame.zeros == zeros)]
            first = cell[cell.padded_position == 1].correct.mean()
            second = cell[cell.padded_position == 2].correct.mean()
            values.append(first - second)
    return np.asarray(values)


def bootstrap_model_contrasts(frames_by_model, iterations=10000):
    rng = np.random.default_rng(8192026)
    cell_effects = {
        model: cell_position_differences(frame)
        for model, frame in frames_by_model.items()
    }
    models = list(cell_effects)
    contrasts = {}
    for i, left in enumerate(models):
        for right in models[i + 1:]:
            diffs = []
            for _ in range(iterations):
                li = rng.integers(0, 50, 50)
                ri = rng.integers(0, 50, 50)
                diffs.append(cell_effects[left][li].mean() -
                             cell_effects[right][ri].mean())
            estimate = cell_effects[left].mean() - cell_effects[right].mean()
            lo, hi = np.quantile(diffs, [0.025, 0.975])
            contrasts[f"{left} minus {right}"] = {
                "difference_in_position_penalties": float(estimate),
                "bootstrap_ci_low": float(lo), "bootstrap_ci_high": float(hi),
            }
    return contrasts


def main():
    frames = []
    summaries = []
    primary_tests = []
    thresholds = {
        "Llama 3.1 8B": (0.5, "less"),
        "Qwen3 4B Instruct 2507": (0.9, "greater"),
        "Gemma 2 9B": (0.8, "greater"),
    }
    position_tests = {}
    digit_tests = {}

    for model, path in FILES.items():
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != 2000 or len({r["id"] for r in rows}) != 2000:
            raise ValueError(f"Invalid run size or duplicate IDs: {path}")
        if any(r.get("parse_status") == "error" for r in rows):
            raise ValueError(f"Execution errors present: {path}")
        frame = pd.DataFrame(rows)
        frame["model"] = model
        frame["correct"] = frame.correct.astype(int)
        frame["padded_first"] = (frame.padded_position == 1).astype(int)
        frames.append(frame)

        k, n = int(frame.correct.sum()), len(frame)
        lo, hi = wilson(k, n)
        first = frame[frame.padded_position == 1]
        second = frame[frame.padded_position == 2]
        first_k, second_k = int(first.correct.sum()), int(second.correct.sum())
        rd = first.correct.mean() - second.correct.mean()
        rd_lo, rd_hi = bootstrap_position_difference(frame)
        summaries.append({
            "model": model, "n": n, "correct": k, "accuracy": k / n,
            "ci_low": lo, "ci_high": hi,
            "padded_first_accuracy": first_k / len(first),
            "padded_second_accuracy": second_k / len(second),
            "risk_difference_first_minus_second": rd,
            "risk_difference_ci_low": rd_lo,
            "risk_difference_ci_high": rd_hi,
            "invalid_responses": int((~frame.prediction.isin([1, 2, 3])).sum()),
        })

        threshold, alternative = thresholds[model]
        p_overall = binomtest(k, n, threshold, alternative=alternative).pvalue
        primary_tests.append((f"overall:{model}", p_overall))

        table = [[first_k, len(first) - first_k],
                 [second_k, len(second) - second_k]]
        p_position = fisher_exact(table, alternative="less").pvalue
        position_tests[model] = p_position
        primary_tests.append((f"position:{model}", p_position))

        d1 = frame[frame.digit == 1]
        d0 = frame[frame.digit == 0]
        table_digit = [[int(d1.correct.sum()), len(d1) - int(d1.correct.sum())],
                       [int(d0.correct.sum()), len(d0) - int(d0.correct.sum())]]
        digit_tests[model] = fisher_exact(table_digit, alternative="less").pvalue

    all_rows = pd.concat(frames, ignore_index=True)
    frames_by_model = {model: frame for model, frame in
                       zip(FILES.keys(), frames)}
    all_rows["digit"] = all_rows.digit.astype("category")
    all_rows["zeros"] = all_rows.zeros.astype("category")
    full = smf.glm(
        "correct ~ C(model) * padded_first + C(digit) + C(zeros)",
        data=all_rows, family=sm.families.Binomial()
    ).fit()
    reduced = smf.glm(
        "correct ~ C(model) + padded_first + C(digit) + C(zeros)",
        data=all_rows, family=sm.families.Binomial()
    ).fit()
    lr = 2 * (full.llf - reduced.llf)
    lr_df = int(full.df_model - reduced.df_model)
    interaction_p = float(chi2.sf(lr, lr_df))

    adjusted = holm(primary_tests)
    tests = []
    for name, raw_p in primary_tests:
        tests.append({"test": name, "raw_p": float(raw_p),
                      "holm_p": float(adjusted[name]),
                      "reject_0.05": bool(adjusted[name] < 0.05)})

    payload = {
        "summaries": summaries,
        "primary_tests": tests,
        "secondary_digit_1_less_than_0_p": {
            key: float(value) for key, value in digit_tests.items()
        },
        "model_by_position_lr_statistic": float(lr),
        "model_by_position_lr_df": lr_df,
        "model_by_position_lr_p": interaction_p,
        "logistic_interaction_separation_warning": True,
        "logistic_interaction_note": (
            "Qwen and Gemma have perfect padded-second accuracy, producing "
            "complete separation and unstable ordinary-GLM interaction "
            "estimates. Treat the prespecified LR p-value as invalid."
        ),
        "bootstrap_position_penalty_contrasts": bootstrap_model_contrasts(
            frames_by_model
        ),
    }
    (RESULTS / "confirmatory_analysis.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
