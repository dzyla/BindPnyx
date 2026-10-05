"""Select the ProteinBase benchmark set: per target, a label-balanced sample of
designs with experimental binding labels, plus the target construct sequence."""
import json, sys, pandas as pd
from pathlib import Path

# target slug -> (UniProt full-length file, first, last residue of the construct)
TARGETS = {
    "pd-l1": ("Q9NZQ7", 19, 238),
    "il7r": ("P16871", 21, 239),
    "mdm2": ("Q00987", 17, 125),
    "egfr": ("P00533", 25, 645),
}
N_PER_CLASS = 30
SEED = 0

def main(out="inputs"):
    out = Path(out); out.mkdir(exist_ok=True)
    x = pd.read_pickle("labels.pkl")
    manifest = []
    for slug, (acc, a, b) in TARGETS.items():
        # full-length UniProt sequence (EBI proteins API); EGFR's is already in the repo
        fa = Path(f"uniprot/{slug}.txt")
        full = fa.read_text().strip() if fa.exists() else "".join(
            l.strip() for l in open("../targets/egfr_ecd/P00533_uniprot_wt.fasta") if not l.startswith(">"))
        tseq = full[a - 1:b]
        s = x[(x.target == slug) & x.seq.str.fullmatch("[ACDEFGHIKLMNPQRSTVWY]+")]
        s = s[(s.seq.str.len() >= 40) & (s.seq.str.len() <= 200)]
        for lab in ("True", "False"):
            ss = s[s.val_agg == lab]
            ss = ss.sample(min(N_PER_CLASS, len(ss)), random_state=SEED)
            for _, r in ss.iterrows():
                manifest.append(dict(target=slug, id=r["id"], label=int(lab == "True"), method=r.method,
                                     binder_seq=r.seq, target_seq=tseq))
        print(slug, len(tseq), "aa;", sum(m['target'] == slug for m in manifest), "designs")
    pd.DataFrame(manifest).to_csv(out / "manifest.csv", index=False)

if __name__ == "__main__":
    main()
