# Choosing the binding site when none is given

`python funnel/hotspots.py --pdb X.pdb --chain A --range 1-158 [--msa x.a3m] --top 5` screens the target surface and proposes ranked hotspot sets (3-5 residues, `"Y56"`-style names).
Add `--pdb-id 3MCY --name mytarget --write-targets funnel/targets/` to write one ready-to-build target JSON per proposed patch (`mytarget_s1.json` ...).

## What it does
Per residue, from the structure alone: relative solvent exposure (freesasa), local concavity (CB neighbours within 10 A), hydrophobic / aromatic / charged / glycine character, optional conservation
from an MSA. A hand-set weighted sum is smoothed over spatial neighbours (sigma 4 A), buried residues are excluded, and patches are grown greedily (seeds >= 14 A apart so alternatives are genuinely different sites).
Seconds on CPU.

## What the evidence says (206 wet-lab-labelled ProteinBase designs on EGFR, IL-7R, MDM2, PD-L1; `funnel/hotspot_analysis.py`, `funnel/hotspot_validate.py`)

1. **Binders and non-binders land on the same place.** The per-residue epitope maps of labelled binders and non-binders correlate 0.88-0.98 on every target. The site a design goes to does **not**
   tell you whether it binds (EGFR: 54% binder rate for designs hitting the dominant patch vs 33% for the rest; the other targets show no such split). Site choice sets *where*, not *whether*.
2. **Where designs go is shared across methods**, i.e. targets have strong attractors for designed proteins. Epitope residues of binders are, on all four targets, enriched in hydrophobic residues
   (z-shift +0.12 to +0.30) and aromatics (+0.05 to +0.38) relative to the rest of the exposed surface. Charge, exposure, concavity and conservation shifts changed sign between targets - not usable.
3. **A fitted surface score does not generalise.** Fitting weights to "residue is in the binders' epitope" on three targets and testing on the fourth gave AUROC 0.56-0.62 (default hand-set weights 0.48-0.54,
   exposure alone 0.45-0.57). Fitted weights were therefore **not adopted**; the shipped weights are the hand-set prior.
4. **Recovering known functional sites:** on the four targets where we know the site (PD-L1 PD-1 face, MDM2 p53 pocket, FimH mannose pocket, FimA donor-strand groove) the real site is in the top 3 of 8 proposed
   patches every time (ranks 3, 2, 2, 1). The fitted weights did worse (rank none / 2 / 8 / 1). Four targets, all of them classic pockets or grooves: this says the screen is a useful shortlist generator, not that it finds
   the right site in general. Many proposals are not druggable or biologically meaningful.

## Recommended protocol when the site is not specified
1. Run `hotspots.py` and look at the top 5 patches in a viewer (PyMOL/ChimeraX). Remove patches that are glycans, the membrane-proximal end, a crystal contact, or an irrelevant face. **If you know the biology, choose the site yourself; this screen is for when you do not.**
2. Build the surviving candidates (`--write-targets` + `fetch_target.py`) and **scout**: for each, `run_funnel.py --n-backbones 100 --rounds 0 --final-m 20` (~0.3 GPU-h each).
3. Compare sites by the number of consensus passes (Boltz-2 + Protenix-v2) **and** the decoy/shuffle margin (`controls.py`). Pick the best, then run the full protocol (docs/PROTOCOL.md) on it.
   A site with many passes but a poor control margin is a sticky patch for the predictor, not a good site.
4. Or skip steps 2-3 and let the full run use the top-ranked patch - cheaper, lower expected quality (untested; no head-to-head against scouting).

## Blind-docking consensus ("sticky epitope") - experimental, validation in progress
`python funnel/dock_epitope.py --pdb X.pdb --chain A --range a-b --out out/epitope/X` docks small public probe proteins (ubiquitin, protein G B1, SH3, protein A B; `funnel/scaffolds.json`)
over the whole target with LightDock **without restraints** and counts, for each target residue, how often the 300 best-scoring poses per probe touch it (heavy atoms < 5 A). Residues contacted by all probes form a
consensus 'sticky' region; patches are proposed from it exactly as from the surface screen. CPU only, about 2.5 min per probe for an 85-residue target and 8-9 min for 210 residues (30 cores).
`funnel/epitope_validate.py` scores the map against (a) the known sites and (b) where labelled binders land; `funnel/run_epitope_validation.sh` runs the whole check.
**Status:** the tool runs and is resumable; whether the consensus is better than the surface screen at finding functional sites is not yet established - treat its output as one more shortlist to inspect, not as a result.

## Not tested / limits
- Scouting is untested end-to-end (the pieces are tested; no target was run through the whole auto-site loop).
- The labelled data show where *designs* go, not which sites are functionally useful (an inhibitor needs the active site; a degrader or sensor needs a different one).
- Conservation was available for the benchmark targets only partially; it was not shown to help.
