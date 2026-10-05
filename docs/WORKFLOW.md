# How the software works: workflow graphs

A PNG of the main flow is in `docs/figures/workflow.png` (source: `docs/make_workflow_figure.py`). The Mermaid diagrams below render on GitHub and are the
editable versions. Times are one RTX 5090 for a 70–85-residue binder.

## 1. End-to-end flow

```mermaid
flowchart TD
    subgraph SETUP["Setup (human + CPU)"]
        T1["1 Target structure (PDB) + WT sequence + MSA<br/>choose ONE face from the structure"]
        T2["2 scripts/prepare_target.py<br/>shard (.pkl.gz) + provenance + MSA check"]
        T3["3 funnel/targets/NAME.json<br/>hotspots as 'Y56' = PDB number + residue letter<br/>identity-checked, converted to shard indices"]
        T1 --> T2 --> T3
    end

    subgraph FUNNEL["Funnel (GPU, strictly one job at a time)"]
        G["4 Generate: PXDesign diffusion<br/>500 backbones, hotspot-conditioned (~2 s each)"]
        D["5 Design: ProteinMPNN, 4 sequences per backbone<br/>target fixed; optional interface-only bias"]
        S["6 Fast screen: Protenix-0.5-mini (2 cycles, 5 steps)<br/>all 2000 designs, 0.4-0.7 s each<br/>best sequence per backbone, keep top 100"]
        C["7 Cycle (optional, 3 rounds)<br/>MPNN on the predicted complex, 8 children,<br/>fast refold, keep the best (elitist)"]
        K["8 Consensus on the top 60<br/>Boltz-2 (3 seeds) + Protenix-v2<br/>consensus = min(ipSAE)"]
        G --> D --> S --> C --> K
        S -. "skip cycling if the top-20 already passes" .-> K
    end

    subgraph TRIAGE["Triage (CPU, ~1-2 s per design)"]
        X["9 De-duplicate (<60% identity)<br/>fastPISA, shape complementarity,<br/>hotspot burial, composition flags"]
        L["10 Shortlist: final_*.csv + predicted complexes"]
        X --> L
    end

    subgraph LAB["Validation loop"]
        P["12 Order a DIVERSE panel<br/>spread over tiers + flagged + controls"]
        W["13 Wet lab: expression + binding"]
        B["14 Calibrate (>= 20 results)<br/>binder rate per tier / band with intervals"]
        P --> W --> B
    end

    T3 --> G
    K --> X
    L --> P
    B -. "update bands and gate; change generator / sequence design / site" .-> T3
    L -. "method comparisons only" .-> J["11 Independent judge<br/>fresh seeds, same protocol for every arm"]
```

## 2. What each stage reads and writes

```mermaid
flowchart LR
    A["targets/NAME.json"] --> G["run_funnel.py<br/>stage_generate"]
    G -->|"gen/run/.../predictions/*.cif<br/>gen/converted.json (GLY-binder PDBs)"| D["stage_screen: MPNN"]
    D -->|"mpnn.csv"| F["Protenix fast<br/>(common.protenix_fold)"]
    F -->|"screen.csv (fast_ipsae, fast_iptm, cif path)<br/>screen_fast/pred/*"| Y["stage_cycle"]
    Y -->|"start_parents.csv, cycled_parents.csv<br/>cycle_children.csv, cycle/*"| Z["consensus()"]
    Z -->|"consensus_nocycle|cycled.csv<br/>consensus_*/boltz/seed*/**.cif, v2/pred/*"| Q["shortlist()"]
    Q -->|"final_nocycle|cycled.csv"| R["judge.py / compare.py<br/>add_pisa_to_runs.py"]
    G -. "timers.json (seconds per stage)" .-> R
```

## 3. Decision rules

```mermaid
flowchart TD
    S0["Run: run_funnel.py --rounds 0   (~0.7-1.2 GPU-h)"] --> Q1{"Top-20: >= 90% consensus-pass<br/>AND Protenix-v2 median >= 0.7 ?"}
    Q1 -- yes --> OUT["Triage with flags, order a diverse panel"]
    Q1 -- no --> S1["Re-run the same --out with --rounds 3 (reuses stages 4-6; +0.4-0.6 h)"]
    S1 --> Q2{"Still low?"}
    Q2 -- no --> OUT
    Q2 -- yes --> H["Hard target: change the site / binder length,<br/>try interface-bias or a co-design generator,<br/>do not just add more backbones"]
```

Evidence for the rule: MDM2 (20/20 without cycling), PD-L1 (20/20), FimA (**16/20 without cycling, 20/20 with it**). See `docs/REPORT.md` §9 and `funnel/results/`.
