"""One parent -> many RELEASE-direction Proton-PottsMPNN designs (CPU, ~3 min). usage: gen_parent.py <name> <pdb>   (binder chain D, target A,B,C)
env: PROTON_POTTS_DIR (the package folder holding checkpoints/ and foundry/), PP_OUT (output folder), HBPLUS_PATH (the transform pipeline calls it even for design).
DIRECTION: the package's default objective MINIMISES the selective gap sum(e_P - e_D), i.e. it favours the protonated state = acid BINDING. A NEGATIVE combined_lambda flips it
(selective > 0 = neutral preferred = acid release). Measured on one complex: lambda +0.3 -> -7.5; -0.3 -> +1.8; -1.0 -> +5.3, Potts stability unchanged.
RESULT NOTE: a positive selective energy did NOT predict PROPKA release (see docs/AGENT_PLAYBOOK_PHBIND.md section 11); use designs from this only as candidates to be folded and checked."""
import os, sys, json, time, itertools
os.environ.pop("DEBUG", None); os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["OMP_NUM_THREADS"] = "2"; os.environ["MKL_NUM_THREADS"] = "2"
from pathlib import Path
import torch; torch.set_num_threads(2)
from biotite.structure.io.pdb import PDBFile
import biotite.sequence as bs
PKG = Path(os.environ["PROTON_POTTS_DIR"]); sys.path[:0] = [str(PKG / d) for d in ("inference", "scoring")]
from mpnn.inference_engines.potts_mpnn_ph import PottsMPNNPHEngine, PHDesignCriteria
CKPT = PKG / "checkpoints" / "potts_v6_afdb_edge_his0.3_acid0.06" / "epoch-0125.ckpt"
name, pdb = sys.argv[1], sys.argv[2]
eng = PottsMPNNPHEngine(checkpoint_path=str(CKPT), extended_vocab="v6", out_directory=None, write_fasta=False, write_structures=False)
aa = PDBFile.read(pdb).get_structure(model=1); ca = aa[(aa.chain_id == "D") & (aa.atom_name == "CA")]
native = "".join(bs.ProteinSequence.convert_letter_3to1(r) for r in ca.res_name); out = []; t0 = time.time()
for lam, T, cnt, seed in itertools.product((-0.3, -0.5, -1.0, -2.0), (0.05, 0.3), (1, 2), (0, 1, 2, 3)):
    try:
        crit = PHDesignCriteria(method="block_descent", backend="potts", temperature=T, samples_per_site=1, block_size=3, combined_lambda=lam, seed_source="native", center_types=["HIS-P"], center_count=cnt,
            dep_map={"HIS-P": ["HIS-S"], "ASP-P": ["ASP-D"], "GLU-P": ["GLU-D"]}, forbidden_tokens=["HIS-A", "ASP-A", "GLU-A", "UNK"], placement_by="scan_potts", placement_region=["interface"],
            repetitive_window_parents=["ARG", "LYS", "HIS", "ASP", "GLU"], repetitive_window_radius=2, repetitive_window_weight=1.0, neighbour_k=16, max_mutations=20)
        b = eng.run_ph_redesign(atom_array=aa, binder_chain="D", criteria_list=[crit], seed=seed)[0]
        out.append(dict(parent=name, lam=lam, T=T, count=cnt, seed=seed, centres=[[int(a), str(c)] for a, c in zip(b.center_res_ids, b.center_protonation_types)], selective=float(b.selective_energy),
                        potts=float(b.final_potts_energy), n_mut=int(sum(x != y for x, y in zip(native, b.canonical_sequence))), n_his=b.canonical_sequence.count("H"), native_n_his=native.count("H"), seq=b.canonical_sequence))
    except Exception as e:
        out.append(dict(parent=name, lam=lam, T=T, count=cnt, seed=seed, error=str(e)[:120]))
json.dump(dict(parent=name, native=native, rows=out, secs=round(time.time() - t0)), open(Path(os.environ.get("PP_OUT", ".")) / f"{name}.json", "w"))
print(name, "done", len(out), "rows in", round(time.time() - t0), "s;", sum(1 for r in out if r.get("selective", -9) >= 0.4), "release-direction")
