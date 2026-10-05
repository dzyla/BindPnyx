"""Choose binding sites (hotspots) from a target structure when none is specified: surface screening.

Per-residue features from the target alone (CPU, seconds): solvent exposure (freesasa), local concavity (neighbour count of CB atoms within 10 A),
hydrophobic / aromatic / charged character, optional conservation from an MSA. Residue scores are smoothed over spatial neighbours, then patches are grown
greedily and returned as ranked candidate hotspot sets ("Y56"-style names that funnel/targets/*.json accepts).

The default weights are a hand-set prior (hydrophobic/aromatic, exposed, concave; glycine and charge penalised). Weights fitted to binder epitopes
(`funnel/hotspot_analysis.py`) generalised no better and were NOT adopted; see docs/HOTSPOTS.md.

    python funnel/hotspots.py --pdb 1YCR.pdb --chain A [--range 1-100] [--msa x.a3m] [--top 5] [--json out.json]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pisa  # noqa: F401  (adds the freesasa / gemmi paths)

AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M",
       "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
HYD = set("AVLIMFWYC"); ARO = set("FWYH"); CHG = set("DEKR")
# default weights (z-scored features). Replaced by the fitted values in funnel/results/hotspot_weights.json when present.
DEFAULT_W = dict(exposed=1.0, concave=0.5, hydrophobic=0.6, aromatic=0.6, charged=-0.2, conserved=0.3, glycine=-0.8)
FEATURES = list(DEFAULT_W)


def load_weights():
    f = Path(__file__).resolve().parent / "results" / "hotspot_weights.json"
    return json.load(open(f)) if f.exists() else dict(DEFAULT_W)


def read_chain(path, chain, rng=None):
    """-> list of residues: dict(num, ins, aa, ca, cb, atoms[n,3]) for standard amino acids of one chain (PDB or mmCIF)."""
    from Bio.PDB import PDBParser, MMCIFParser
    p = MMCIFParser(QUIET=True) if str(path).endswith((".cif", ".mmcif")) else PDBParser(QUIET=True)
    model = p.get_structure("x", str(path))[0]
    if chain not in [c.id for c in model]: raise SystemExit(f"chain {chain} not in {path}; has {[c.id for c in model]}")
    out = []
    for r in model[chain]:
        if r.id[0] != " " or r.get_resname() not in AA3 or "CA" not in r: continue
        n = r.id[1]
        if rng and not (rng[0] <= n <= rng[1]): continue
        atoms = np.array([a.coord for a in r if a.element != "H"])
        cb = r["CB"].coord if "CB" in r else (r["CA"].coord + 1.5 * (atoms.mean(0) - r["CA"].coord) / (np.linalg.norm(atoms.mean(0) - r["CA"].coord) + 1e-6))
        out.append(dict(num=n, ins=r.id[2].strip(), aa=AA3[r.get_resname()], ca=r["CA"].coord, cb=cb, atoms=atoms, res=r))
    return out


def sasa_per_residue(residues):
    """Relative side-chain-ish exposure from freesasa on the isolated chain (A^2, then divided by a per-aa maximum)."""
    import freesasa
    MAX = dict(A=129, R=274, N=195, D=193, C=167, Q=225, E=223, G=104, H=224, I=197, L=201, K=236, M=224, F=240, P=159, S=155, T=172, W=285, Y=263, V=174)
    coords, radii, owner = [], [], []
    for i, r in enumerate(residues):
        for a in r["res"]:
            if a.element == "H": continue
            coords += list(a.coord); radii.append({"C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8}.get(a.element, 1.7)); owner.append(i)
    res = freesasa.calcCoord(coords, radii); tot = np.zeros(len(residues))
    for k, i in enumerate(owner): tot[i] += res.atomArea(k)
    return np.array([min(1.5, tot[i] / MAX[r["aa"]]) for i, r in enumerate(residues)])


def conservation(msa_path, query):
    """Fraction of aligned sequences matching the query residue at each position (gaps count as mismatches); None if the MSA does not fit the query."""
    seqs = []
    for line in open(msa_path):
        line = line.strip()
        if not line or line.startswith(("#", ">")) or line.lower().startswith("key"): continue
        s = "".join(c for c in line.split(",")[-1] if c.isupper() or c == "-")
        if len(s) == len(query): seqs.append(s)
    if len(seqs) < 5: return None
    A = np.array([list(s) for s in seqs]); return (A == np.array(list(query))[None]).mean(0)


def residue_features(residues, cons=None):
    """z-scored feature matrix [n, len(FEATURES)] + raw exposure."""
    n = len(residues); cb = np.array([r["cb"] for r in residues]); rel = sasa_per_residue(residues)
    d = np.linalg.norm(cb[:, None] - cb[None], axis=-1); neigh = (d < 10.0).sum(1) - 1
    # a residue on a flat surface has ~ half the neighbours of a buried one; a pocket floor sits in a crowd but remains exposed -> use neighbours among exposed context
    aa = [r["aa"] for r in residues]
    F = np.zeros((n, len(FEATURES)))
    F[:, 0] = np.minimum(rel, 1.0)
    F[:, 1] = neigh
    F[:, 2] = [a in HYD for a in aa]; F[:, 3] = [a in ARO for a in aa]; F[:, 4] = [a in CHG for a in aa]
    F[:, 5] = cons if cons is not None and len(cons) == n else 0.0
    F[:, 6] = [a == "G" for a in aa]
    Z = (F - F.mean(0)) / (F.std(0) + 1e-9)
    return Z, rel, d


def score_residues(residues, weights=None, cons=None, smooth=8.0):
    """Residue score = weighted features, averaged over spatial neighbours (Gaussian, sigma = smooth/2 A) among exposed residues. Buried residues get -inf."""
    w = weights or load_weights(); Z, rel, d = residue_features(residues, cons)
    raw = Z @ np.array([w[k] for k in FEATURES])
    K = np.exp(-(d / (smooth / 2)) ** 2) * (rel[None] > 0.15)
    sm = (K @ raw) / (K.sum(1) + 1e-9)
    s = 0.5 * raw + 0.5 * sm
    s[rel < 0.15] = -np.inf
    return s, rel, d


def pick_patches(residues, score, rel, d, n_hot=5, top=5, radius=9.0, min_sep=14.0):
    """Greedy: seed = best-scoring exposed residue not already used; take the n_hot best-scoring exposed residues within `radius` of the seed.
    Patches must be at least `min_sep` A apart (seed to seed) so the alternatives are genuinely different sites."""
    order = np.argsort(-score); seeds, patches = [], []
    for i in order:
        if not np.isfinite(score[i]) or len(patches) >= top: break
        if any(d[i, j] < min_sep for j in seeds): continue
        near = [j for j in np.argsort(-score) if d[i, j] <= radius and np.isfinite(score[j]) and rel[j] >= 0.25]
        sel = sorted(near[:n_hot], key=lambda j: residues[j]["num"])
        if len(sel) < 3: continue
        seeds.append(i)
        patches.append(dict(rank=len(patches) + 1, score=float(np.mean(score[sel])), seed=f"{residues[i]['aa']}{residues[i]['num']}",
                            hotspots=[f"{residues[j]['aa']}{residues[j]['num']}{residues[j]['ins']}" for j in sel],
                            mean_exposure=float(np.mean(rel[sel])), hydrophobic_frac=float(np.mean([residues[j]['aa'] in HYD for j in sel])),
                            charged_frac=float(np.mean([residues[j]['aa'] in CHG for j in sel]))))
    return patches


def propose(path, chain, rng=None, msa=None, n_hot=5, top=5, weights=None):
    res = read_chain(path, chain, rng); seq = "".join(r["aa"] for r in res)
    cons = conservation(msa, seq) if msa else None
    s, rel, d = score_residues(res, weights, cons)
    return pick_patches(res, s, rel, d, n_hot, top), res, s


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pdb", required=True); ap.add_argument("--chain", default="A"); ap.add_argument("--range", help="e.g. 1-158")
    ap.add_argument("--msa"); ap.add_argument("--n-hot", type=int, default=5); ap.add_argument("--top", type=int, default=5); ap.add_argument("--json")
    ap.add_argument("--pdb-id", help="with --write-targets: PDB id that fetch_target.py should download"); ap.add_argument("--name", default="site")
    ap.add_argument("--write-targets", help="directory: write one funnel target JSON per proposed patch (<name>_s1.json ...) for site scouting"); ap.add_argument("--binder-length", type=int, default=75)
    a = ap.parse_args(); rng = tuple(int(x) for x in a.range.split("-")) if a.range else None
    patches, res, _ = propose(a.pdb, a.chain, rng, a.msa, a.n_hot, a.top)
    print(f"{len(res)} residues; weights: {load_weights()}")
    for p in patches:
        print(f"  #{p['rank']} score {p['score']:+.2f}  hotspots {p['hotspots']}  exposure {p['mean_exposure']:.2f}  hydrophobic {p['hydrophobic_frac']:.0%}  charged {p['charged_frac']:.0%}")
    if a.json: json.dump(patches, open(a.json, "w"), indent=1)
    if a.write_targets:
        if not a.pdb_id: sys.exit("--write-targets needs --pdb-id (fetch_target.py downloads from RCSB)")
        out = Path(a.write_targets); out.mkdir(parents=True, exist_ok=True)
        for p in patches:
            cfg = dict(name=f"{a.name}_s{p['rank']}", description=f"auto-proposed site #{p['rank']} (surface screen, score {p['score']:+.2f})",
                       source=dict(pdb_id=a.pdb_id, chain=a.chain, **({"range": a.range} if a.range else {})), hotspots=p["hotspots"], binder_length=a.binder_length)
            json.dump(cfg, open(out / f"{a.name}_s{p['rank']}.json", "w"), indent=2)
        print(f"wrote {len(patches)} target files to {out}; build with funnel/fetch_target.py, scout with run_funnel.py --n-backbones 100 --rounds 0")


if __name__ == "__main__": main()
