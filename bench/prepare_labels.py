"""ProteinBase bulk CSV -> labels.pkl: one row per (target, design) with an aggregated
experimental binding label (True if any experimental 'binding' evaluation is True)."""
import json, pandas as pd
d = pd.read_csv("proteinbase_all_data_28_01_2026.csv")
rows = []
for _, r in d.iterrows():
    try: ev = json.loads(r.evaluations)
    except Exception: continue
    for e in ev:
        if e.get("metric") == "binding" and e.get("type") == "experimental":
            rows.append(dict(id=r["id"], seq=r.sequence, method=r.designMethod, target=e["target"], val=str(e["value"])))
x = pd.DataFrame(rows)
agg = x.groupby(["target", "id"]).val.agg(lambda s: "True" if (s == "True").any() else "False").rename("val_agg").reset_index()
x = x.drop_duplicates(["target", "id"]).merge(agg, on=["target", "id"])
x.to_pickle("labels.pkl"); print(x.groupby("target").val_agg.value_counts().unstack(fill_value=0).sort_values("True", ascending=False).head(12))
