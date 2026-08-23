#!/usr/bin/env python3
import json
from decimal import Decimal
from pathlib import Path

rows=[]
for whole in range(21,61):
    split="train" if whole<=40 else "validation" if whole<=50 else "test"
    for digit in (1,4,7):
        canonical=f"{whole}.{digit}"
        forms={"canonical":canonical,"padded_1":canonical+"0","padded_2":canonical+"00"}
        nearby={"near_minus":str(Decimal(canonical)-Decimal('0.01')),
                "near_plus":str(Decimal(canonical)+Decimal('0.01'))}
        for target_position in (1,2):
            for form_name,text in {**forms,**nearby}.items():
                rows.append({"id":f"mech-{whole:02d}-{digit}-{target_position}-{form_name}",
                             "whole":whole,"digit":digit,"value":float(Decimal(canonical)),
                             "split":split,"target_position":target_position,
                             "form":form_name,"text":text,"canonical":canonical,
                             "is_equivalent":form_name in forms})
out=Path("data/mechanistic_values.jsonl")
out.write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in rows))
print(out,len(rows))
