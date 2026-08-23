#!/usr/bin/env python3
"""Bootstrap token-level causal components and descriptive strata."""
import csv, json
from pathlib import Path
import numpy as np


def ci(values, seed):
    values=np.asarray(values,float); rng=np.random.default_rng(seed)
    means=values[rng.integers(0,len(values),(10000,len(values)))].mean(1)
    return [float(np.quantile(means,.025)),float(np.quantile(means,.975))]


def main():
    output=[]
    for model_index,path in enumerate(sorted(Path("results").glob("token_decomposition_*.csv"))):
        rows=list(csv.DictReader(path.open())); assert len(rows)==800
        by={(int(r["case"]),r["control"]):r for r in rows}; cases=range(100); analyses=[]
        joint_path = Path("results") / path.name.replace("token_decomposition_", "causal_generalization_")
        joint_rows = list(csv.DictReader(joint_path.open()))
        assert len(joint_rows) == 600
        joint_by = {(int(r["case"]), r["control"]): r for r in joint_rows}
        for source in ("easy","donor"):
            for component in ("padded","short"):
                aligned=f"{source}_{component}_tokens"; random=f"{source}_{component}_random"
                contrast={c:float(by[c,aligned]["margin_effect"])-float(by[c,random]["margin_effect"]) for c in cases}
                for template in (None,"relation_statements","direct_choice"):
                    selected=[c for c in cases if template is None or by[c,aligned]["template"]==template]
                    vals=[contrast[c] for c in selected]
                    analyses.append({"source":source,"component":component,"template":template or "pooled",
                                     "n":len(vals),"aligned_minus_random":float(np.mean(vals)),
                                     "bootstrap_95_ci":ci(vals,83100+model_index*100+len(analyses))})
        # The addendum explicitly calls for a descriptive comparison with the
        # already-run joint patch.  This is not a factorial interaction test.
        nonadditivity=[]
        for source in ("easy", "donor"):
            joint_aligned=f"{source}_number_tokens"; joint_random=f"{source}_random_positions"
            for template in (None,"relation_statements","direct_choice"):
                selected=[c for c in cases if template is None or by[c,f"{source}_padded_tokens"]["template"]==template]
                values=[]
                for c in selected:
                    joint=float(joint_by[c,joint_aligned]["margin_effect"])-float(joint_by[c,joint_random]["margin_effect"])
                    padded=float(by[c,f"{source}_padded_tokens"]["margin_effect"])-float(by[c,f"{source}_padded_random"]["margin_effect"])
                    short=float(by[c,f"{source}_short_tokens"]["margin_effect"])-float(by[c,f"{source}_short_random"]["margin_effect"])
                    values.append(joint-padded-short)
                nonadditivity.append({"source":source,"template":template or "pooled","n":len(values),
                                      "joint_minus_component_sum":float(np.mean(values)),
                                      "bootstrap_95_ci":ci(values,83500+model_index*100+len(nonadditivity)),
                                      "interpretation":"descriptive; not a formal factorial interaction"})

        strata=[]
        for field in ("zeros","digit","patched_token_count"):
            values=sorted({int(r[field]) for r in rows})
            for value in values:
                for source in ("easy","donor"):
                    for component in ("padded","short"):
                        aligned=f"{source}_{component}_tokens"; random=f"{source}_{component}_random"
                        selected=[c for c in cases if int(by[c,aligned][field])==value]
                        vals=[float(by[c,aligned]["margin_effect"])-float(by[c,random]["margin_effect"]) for c in selected]
                        if vals: strata.append({"field":field,"value":value,"source":source,"component":component,
                                                "n":len(vals),"mean_contrast":float(np.mean(vals))})
        # Joint-patch strata use the same case metadata.  The joint token count
        # is recorded as numeral_token_count in the joint-patch output.
        for field in ("zeros", "digit", "numeral_token_count"):
            values=sorted({int(r[field]) for r in joint_rows})
            for value in values:
                for source in ("easy", "donor"):
                    aligned=f"{source}_number_tokens"; random=f"{source}_random_positions"
                    selected=[c for c in cases if int(joint_by[c,aligned][field])==value]
                    vals=[float(joint_by[c,aligned]["margin_effect"])-float(joint_by[c,random]["margin_effect"]) for c in selected]
                    if vals: strata.append({"field":field,"value":value,"source":source,"component":"joint",
                                            "n":len(vals),"mean_contrast":float(np.mean(vals))})
        output.append({"model":rows[0]["model"],"revision":rows[0]["revision"],
                       "fixed_layer_zero_based":int(rows[0]["layer"]),"analyses":analyses,
                       "joint_minus_component_sum":nonadditivity,"strata":strata})
    Path("results/token_decomposition_analysis.json").write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps(output,indent=2))


if __name__=="__main__": main()
