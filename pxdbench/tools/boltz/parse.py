"""Read and score one Boltz-2 prediction directory.

Layout is Boltz 2.2.1's:

    <seed_dir>/boltz_results_<name>/predictions/<name>/
        <name>_model_0.pdb
        pae_<name>_model_0.npz
        plddt_<name>_model_0.npz
        confidence_<name>_model_0.json

Failures never raise and never produce NaN: they return None metrics plus a
`bz_status` reason, so a failed design cannot be mistaken for a low score or
(after the filter fix) pass a threshold.

Biopython is used rather than gemmi because gemmi is not installed in pxlocal
and coordinate reading is convention-free, so this does not diverge from
run_boltz_campaign.py's scoring.
"""
import json
import os

import numpy as np
from Bio.PDB import PDBParser

from pxdbench.metrics import epitope as ep
from pxdbench.metrics import interface as im

BOLTZ_VERSION_EXPECTED = "2.2.1"

METRIC_KEYS = (
    "bz_ipdae",
    "bz_ipsae",
    "bz_pae_interface_min",
    "bz_interface_plddt",
    "bz_iptm",
    "bz_ptm",
    "bz_complex_plddt",
)


#: the directory name _fold_batch stages YAMLs into; Boltz derives the results
#: directory name from it (see prediction_dir)
BATCH_INPUT_STEM = "input"


def prediction_dir(out_dir, sample_name, input_stem=None):
    """Where Boltz writes the predictions for one design.

    Boltz forms the results directory from the INPUT PATH's stem
    (boltz/main.py:1134 -- `out_dir / f"boltz_results_{data.stem}"`), then puts
    one subdirectory per design under `predictions/`. So the stem depends on
    what was passed in, not on the design:

      batched: input is a DIRECTORY, e.g. .../seed_1/input
               -> boltz_results_input/predictions/<design>/
      single:  input is one file, e.g. c.yaml
               -> boltz_results_c/predictions/c/

    `input_stem` defaults to `sample_name`, which is the single-file case where
    the YAML happens to be named after the design. Batched callers must pass
    BATCH_INPUT_STEM; the two coincide only when a design is folded alone from a
    file named after it, which is why the batched path needed a real run to
    catch.
    """
    stem = sample_name if input_stem is None else input_stem
    return os.path.join(
        out_dir, f"boltz_results_{stem}", "predictions", sample_name
    )


def _failure(reason):
    out = {key: None for key in METRIC_KEYS}
    out["bz_status"] = reason
    return out


def read_prediction(pred_dir, sample_name, binder_chain):
    """Return arrays and confidence scalars for one prediction.

    Raises FileNotFoundError / ValueError; `score_prediction` converts those
    into a status.
    """
    pdb_path = os.path.join(pred_dir, f"{sample_name}_model_0.pdb")
    pae_path = os.path.join(pred_dir, f"pae_{sample_name}_model_0.npz")
    plddt_path = os.path.join(pred_dir, f"plddt_{sample_name}_model_0.npz")
    conf_path = os.path.join(pred_dir, f"confidence_{sample_name}_model_0.json")

    if not os.path.exists(pdb_path):
        raise FileNotFoundError(pdb_path)
    if not os.path.exists(pae_path):
        raise FileNotFoundError(pae_path)

    model = PDBParser(QUIET=True).get_structure("c", pdb_path)[0]
    ca, chains = [], []
    for chain in model:
        for residue in chain:
            if "CA" not in residue:
                continue
            ca.append(residue["CA"].coord)
            chains.append(chain.id)
    ca = np.asarray(ca, dtype=float)
    chains = np.asarray(chains)
    if binder_chain not in set(chains.tolist()):
        raise ValueError(f"binder chain {binder_chain!r} not in {sorted(set(chains))}")
    is_binder = chains == binder_chain

    pae = np.load(pae_path)["pae"].astype(float)
    if pae.shape[0] != len(ca):
        raise ValueError(f"PAE dim {pae.shape[0]} != residues {len(ca)}")

    if os.path.exists(plddt_path):
        plddt = np.load(plddt_path)["plddt"].astype(float)
    else:
        plddt = np.full(len(ca), np.nan)
    if plddt.size == len(ca) and np.nanmax(plddt) <= 1.0:
        plddt = plddt * 100.0

    conf = json.load(open(conf_path)) if os.path.exists(conf_path) else {}
    return {
        "pae": pae,
        "ca": ca,
        "is_binder": is_binder,
        "plddt_0_100": plddt,
        "iptm": conf.get("iptm"),
        "ptm": conf.get("ptm"),
        "complex_plddt": conf.get("complex_plddt"),
    }



def _epitope(pred_dir, sample_name, binder_chain, policy):
    """ep_* columns for one prediction; an absent policy yields the blank set."""
    pdb_path = os.path.join(pred_dir, f"{sample_name}_model_0.pdb")
    policy = policy or {}
    return ep.evaluate_epitope_policy(
        pdb_path,
        binder_chain,
        required=policy.get("required"),
        forbidden=policy.get("forbidden"),
        cutoff=float(policy.get("cutoff", ep.DEFAULT_CONTACT_CUTOFF)),
    )


def score_prediction(
    pred_dir,
    sample_name,
    binder_chain,
    ipdae_cutoff=im.IPDAE_CUTOFF,
    ipsae_pae_cutoff=im.IPSAE_PAE_CUTOFF,
    epitope_policy=None,
):
    """All bz_* metrics for one prediction, or a failure status.

    `epitope_policy` is {"required": {...}, "forbidden": {...}, "cutoff": float}.
    When given, ep_* columns record whether the predicted complex actually
    contacts the residues the design was conditioned on - which nothing else
    checks, hotspots being only a soft generation condition.
    """
    try:
        data = read_prediction(pred_dir, sample_name, binder_chain)
    except FileNotFoundError as exc:
        return _failure("missing_pae" if "pae_" in str(exc) else "missing_pdb")
    except ValueError as exc:
        if "not in" in str(exc):
            return _failure("binder_chain_absent")
        return _failure("shape_mismatch")
    except Exception:  # noqa: BLE001 - a parse failure is data, not a crash
        return _failure("parse_error")

    pae, ca, is_binder = data["pae"], data["ca"], data["is_binder"]
    ipdae, _track = im.i_pdae(pae, ca, is_binder, cutoff=ipdae_cutoff)
    if ipdae is None:
        out = _failure("no_interface")
        out.update(_epitope(pred_dir, sample_name, binder_chain, epitope_policy))
        return out

    plddt = data["plddt_0_100"]
    has_plddt = plddt.size == len(ca) and bool(np.isfinite(plddt).any())
    scored = {
        "bz_ipdae": ipdae,
        "bz_ipsae": im.ipsae(pae, is_binder, pae_cutoff=ipsae_pae_cutoff),
        "bz_pae_interface_min": im.pae_interface_min(pae, is_binder),
        "bz_interface_plddt": (
            im.interface_plddt(plddt, ca, is_binder, cutoff=ipdae_cutoff)
            if has_plddt
            else None
        ),
        "bz_iptm": data["iptm"],
        "bz_ptm": data["ptm"],
        "bz_complex_plddt": data["complex_plddt"],
        "bz_status": "ok",
    }
    scored.update(_epitope(pred_dir, sample_name, binder_chain, epitope_policy))
    return scored
