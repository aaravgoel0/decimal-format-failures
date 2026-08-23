#!/usr/bin/env python3
"""Fixed-site padded-versus-short numeral causal decomposition."""
import argparse, csv, json, random, re
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

from run_causal_generalization import (SYSTEM, final_logits, formatted_ids, hits,
                                       mapping, margin, replace, state_at)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True); parser.add_argument("--revision", required=True)
    parser.add_argument("--layer", required=True, type=int)
    args = parser.parse_args()
    cases = [json.loads(line) for line in open("data/causal_generalization.jsonl")]
    model, tokenizer = load(args.model, revision=args.revision)
    template_mode = "system-role"
    try: tokenizer.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":"test"}], tokenize=False, add_generation_prompt=True)
    except Exception: template_mode = "system-prepended-to-user"
    answer_ids = {i: tokenizer.encode(str(i), add_special_tokens=False)[0] for i in (1,2,3)}
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.model)
    path = Path(f"results/token_decomposition_{safe}.csv")
    existing = list(csv.DictReader(path.open())) if path.exists() else []
    completed = {case for case in range(100) if sum(int(row["case"]) == case for row in existing) == 8}
    for row in cases:
        case = row["case"]
        if case in completed: print(f"{case+1}/100 already complete", flush=True); continue
        easy_ids = formatted_ids(tokenizer, row["easy_prompt"], template_mode)
        hard_ids = formatted_ids(tokenizer, row["hard_prompt"], template_mode)
        donor_ids = formatted_ids(tokenizer, row["donor_prompt"], template_mode)
        assert len(easy_ids) == len(hard_ids) == len(donor_ids)
        short_ids = tokenizer.encode(row["short"], add_special_tokens=False)
        padded_ids = tokenizer.encode(row["padded"], add_special_tokens=False)
        donor_short_ids = tokenizer.encode(f"{row['whole']}.{row['donor_digit']}", add_special_tokens=False)
        assert len(short_ids) == len(donor_short_ids)
        easy_short, easy_padded = min(hits(easy_ids, short_ids)), max(hits(easy_ids, padded_ids))
        hard_padded, hard_short = min(hits(hard_ids, padded_ids)), max(hits(hard_ids, short_ids))
        donor_padded, donor_short = min(hits(donor_ids, padded_ids)), max(hits(donor_ids, donor_short_ids))
        maps = {
            "easy_padded_tokens": mapping(hard_padded, easy_padded, len(padded_ids)),
            "easy_short_tokens": mapping(hard_short, easy_short, len(short_ids)),
            "donor_padded_tokens": mapping(hard_padded, donor_padded, len(padded_ids)),
            "donor_short_tokens": mapping(hard_short, donor_short, len(short_ids)),
        }
        excluded = set(range(hard_padded, hard_padded + len(padded_ids))) | set(range(hard_short, hard_short + len(short_ids)))
        eligible = [i for i in range(1, len(hard_ids)-1) if i not in excluded]
        padded_random = random.Random(83_000 + case).sample(eligible, len(padded_ids))
        short_random = random.Random(84_000 + case).sample(eligible, len(short_ids))
        maps.update({
            "easy_padded_random": [(p,p) for p in padded_random],
            "easy_short_random": [(p,p) for p in short_random],
            "donor_padded_random": [(p,p) for p in padded_random],
            "donor_short_random": [(p,p) for p in short_random],
        })
        easy_state, _ = state_at(model, easy_ids, args.layer)
        hard_state, hard_mask = state_at(model, hard_ids, args.layer)
        donor_state, _ = state_at(model, donor_ids, args.layer)
        hard_margin = margin(final_logits(model, hard_state, hard_mask, args.layer), answer_ids)
        output = []
        for control, positions in maps.items():
            source = easy_state if control.startswith("easy") else donor_state
            patched_margin = margin(final_logits(model, replace(hard_state, source, positions), hard_mask, args.layer), answer_ids)
            output.append({"id":row["id"],"case":case,"template":row["template"],"whole":row["whole"],
                           "digit":row["digit"],"zeros":row["zeros"],"control":control,"layer":args.layer,
                           "model":args.model,"revision":args.revision,"hard_margin":hard_margin,
                           "patched_margin":patched_margin,"margin_effect":patched_margin-hard_margin,
                           "hard_correct":hard_margin>0,"patched_correct":patched_margin>0,
                           "patched_token_count":len(positions)})
        with path.open("a",newline="") as file:
            writer=csv.DictWriter(file,fieldnames=output[0]);
            if file.tell()==0: writer.writeheader()
            writer.writerows(output)
        del easy_state,hard_state,donor_state; mx.clear_cache()
        print(f"{case+1}/100",flush=True)
    print(path)


if __name__=="__main__": main()
