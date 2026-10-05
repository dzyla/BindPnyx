#!/usr/bin/env bash
# Run one binder design campaign: diffusion -> MPNN -> Boltz-2 -> ranked export.
#
# This wrapper exists because the correct invocation is not guessable. It finds
# the right interpreter and scorer (project env first - see scripts/setup.sh),
# sets PYTHONPATH, keeps the three MPNN settings together, and leaves the traps
# switched off. Everything it decides is explained in CLAUDE.md.
#
#   ./scripts/run_campaign.sh -i target.json -o out/run1
#   ./scripts/run_campaign.sh -i target.json -o out/run1 --backbones 20 --seqs 4
#   ./scripts/run_campaign.sh -i target.json -o out/run1 --protenix   # also score with Protenix-v2
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# No hardcoded environment paths. Resolution order is the project's own
# .pxd/envs (built by scripts/setup.sh), then $PXD_PYTHON, then discovery across
# the conda roots on this machine. An earlier revision baked absolute paths here
# and the app was unusable anywhere else.
CKPT="${PXD_CHECKPOINTS:-$REPO/checkpoints}"

usage() {
    sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options
  -i, --input PATH      campaign JSON (required)
  -o, --out DIR         output directory (required)
      --backbones N     diffusion samples per round        [8]
      --seqs N          MPNN sequences per backbone        [4]
      --seeds-gate N    Boltz seeds for the gate pass      [3]
      --workers N       concurrent Boltz seeds             [3]
      --keep N          designs to return                  [20]
      --protenix        also score with Protenix-v2 (slower, recorded not ranked)
      --                everything after this goes to the pipeline verbatim
EOF
}

# --- arguments first ----------------------------------------------------------
# Validated BEFORE resolve_python, which probes every candidate environment with
# its own 120 s timeout. With this after the probe, a typo'd flag cost minutes
# before being rejected - and the 2026-10-01 smoke run spent two launches
# discovering that, one of them on a flag that does not exist (--seeds; the real
# one is --seeds-gate). Rejection is now immediate.
# Defaults MUST be set before the loop below, not after it. They used to sit
# further down, left behind when the parsing was moved up here to reject bad
# flags immediately - so every value the caller passed was parsed, validated,
# and then overwritten. $OUT became "" and the run died on `mkdir -p ""` after
# a full preflight, with the arguments apparently accepted.
INPUT=""; OUT=""; BACKBONES=8; SEQS=4; SEEDS_GATE=3; WORKERS=3
KEEP=20; PROTENIX=false; EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--input)      INPUT="$2"; shift 2 ;;
        -o|--out)        OUT="$2"; shift 2 ;;
        --backbones)     BACKBONES="$2"; shift 2 ;;
        --seqs)          SEQS="$2"; shift 2 ;;
        --seeds-gate)    SEEDS_GATE="$2"; shift 2 ;;
        --workers)       WORKERS="$2"; shift 2 ;;
        --keep)          KEEP="$2"; shift 2 ;;
        --protenix)      PROTENIX=true; shift ;;
        -h|--help)       usage; exit 0 ;;
        --)              shift; EXTRA=("$@"); break ;;
        *)               echo "unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

[[ -n "$INPUT" && -n "$OUT" ]] || { usage; exit 2; }
[[ -f "$INPUT" ]] || { echo "input not found: $INPUT" >&2; exit 2; }

resolve_python() {
    local own="$REPO/.pxd/envs/pxd/bin/python"
    [[ -x "$own" ]] && { printf '%s' "$own"; return 0; }
    [[ -n "${PXD_PYTHON:-}" && -x "${PXD_PYTHON}" ]] && { printf '%s' "$PXD_PYTHON"; return 0; }
    local found
    found=$(python3 "$REPO/scripts/pxd_env.py" --what diffusion_python 2>/dev/null) || return 1
    [[ -n "$found" ]] && { printf '%s' "$found"; return 0; }
    return 1
}

PY="$(resolve_python)" || {
    cat >&2 <<'MSG'
FAIL: no interpreter can run the diffusion stage.

It needs protenix 2.x (pxdesign/model/embedders.py imports
update_input_feature_dict, absent before 2.x) and a working CUDA device.

Build the project's own environment - this touches no other env:
    ./scripts/setup.sh --pxd

Or point at an existing one:
    export PXD_PYTHON=/path/to/env/bin/python

To see what this machine has:
    python3 scripts/pxd_env.py
MSG
    exit 1
}



# --- preflight: fail here, not forty minutes in --------------------------------
echo "== preflight =="

# A repo-root directory masking an installed package breaks imports in a way
# whose error message points at the package, not at the shadowing. Refuse early.
# --shadowing, not --json: this is a pure filesystem question, but --json
# probes EVERY interpreter on the machine (120 s each, plus ~80 s for the
# always-failing LayerNorm build in any env holding protenix). A launch used to
# sit silent at this line for minutes, indistinguishable from a hang.
SHADOW=$(python3 "$REPO/scripts/pxd_env.py" --shadowing 2>/dev/null) || true
if [[ -n "${SHADOW:-}" ]]; then
    echo "  FAIL: repo directories are masking installed packages:" >&2
    echo "$SHADOW" | sed 's/^/    /' >&2
    cat >&2 <<'MSG'
  Everything runs with PYTHONPATH=<repo>, so these win over site-packages.
  An incomplete protenix tree here once produced
    "No module named protenix.data.infer_data_pipeline"
  from a package that was installed and complete.
  Move it aside (e.g. into .pxd/quarantine/) and re-run.
MSG
    exit 1
fi
echo "  no package shadowing"
[[ -x "$PY" ]] || { echo "interpreter not found: $PY" >&2; exit 1; }

# The target/MSA pair, checked with the SAME code BoltzBackend.prepare_json
# uses - target_chains_from_orig_seqs then validate_target_chains. prepare_json
# runs after diffusion and MPNN; on 2026-10-02 a one-residue disagreement
# therefore cost a full GPU pass (targets/egfr_ecd/PROVENANCE.md). Here it
# costs 0.4 s, because preflight_target.py reconstructs orig_seqs from the
# shard instead of from the dataloader. It runs BEFORE the CUDA and protenix
# probes below, which cost ~80 s between them: a bad target should not have to
# wait for a healthy GPU to be reported.
if ! PYTHONPATH="$REPO" "$PY" "$REPO/scripts/preflight_target.py" "$INPUT"; then
    cat >&2 <<'MSG'
  FAIL: the target does not match the MSA it is paired with.
        Boltz would discard that alignment and fold the chain unconditioned,
        which looks like weak designs, not like a broken input.
        Rebuild the target with its provenance:
            python scripts/prepare_target.py --help
MSG
    exit 1
fi
echo "  target/MSA pairs OK"

# PYTHONPATH must match the real run below: without it the preflight imports
# protenix from site-packages instead of the vendored (patched) tree, which
# triggers the fused-LayerNorm JIT build and fails before the campaign starts.
PYTHONPATH="$REPO" "$PY" - <<'PREFLIGHT' || exit 1
import sys
try:
    import torch
    from protenix.model.protenix import update_input_feature_dict  # noqa: F401
except Exception as exc:
    sys.exit(f"  FAIL: {exc}\n  The diffusion stage needs protenix 2.x; see CLAUDE.md.")
if not torch.cuda.is_available():
    sys.exit("  FAIL: no CUDA device")
# a real matmul, never the availability flag - torch 2.3/cu121 passes the flag
# on sm_120 and then dies on the first kernel launch
a = torch.randn(512, 512, device="cuda")
float((a @ a).sum())
print(f"  torch {torch.__version__} on {torch.cuda.get_device_name(0)} "
      f"sm{torch.cuda.get_device_capability(0)} - real matmul OK")
PREFLIGHT

for f in pxdesign_v0.1.0.pt; do
    [[ -e "$CKPT/$f" ]] || { echo "  FAIL: missing checkpoint $CKPT/$f" >&2; exit 1; }
done
echo "  checkpoints OK: $CKPT"

# Resolved by the same code the pipeline uses, so preflight cannot disagree with
# the run - the FileNotFoundError that killed a completed diffusion stage was
# exactly that disagreement.
BOLTZ_OUT=$(PYTHONPATH="$REPO" "$PY" -c "
from pxdbench.toolenv import resolve_boltz_bin, ToolNotFound
import sys
try:
    print(resolve_boltz_bin())
except ToolNotFound as exc:
    print(str(exc), file=sys.stderr)
    sys.exit(1)
" 2>&1) || {
    # the explanation is the useful part - print it, do not swallow it
    printf '  FAIL: %s\n' "$BOLTZ_OUT" >&2
    exit 1
}
BOLTZ=$(printf '%s' "$BOLTZ_OUT" | tail -1)
echo "  boltz OK: $BOLTZ"
export PXD_BOLTZ_BIN="$BOLTZ"

# Layer 4 of the MSA defence greps boltz's output for its own warning string,
# which nothing else verifies. If a boltz upgrade rewords it, discarded MSAs
# would go undetected and score as weak designs, so check it in the installed
# source. Derived from $BOLTZ's interpreter, never a literal path.
BOLTZ_FEAT=$("$(dirname "$BOLTZ")/python" -c \
    "import boltz.data.feature.featurizerv2 as m; print(m.__file__)" 2>/dev/null || true)
if [[ -z "$BOLTZ_FEAT" || ! -f "$BOLTZ_FEAT" ]] || \
   ! grep -qF "MSA does not match input sequence, creating dummy." "$BOLTZ_FEAT"; then
    echo "  FAIL: could not find boltz's discarded-MSA warning in its installed" >&2
    echo "        source (${BOLTZ_FEAT:-featurizerv2.py not found}). Detection of" >&2
    echo "        discarded MSAs would silently stop; update DUMMY_MSA_WARNING in" >&2
    echo "        pxdbench/tools/boltz/msa_check.py to match." >&2
    exit 1
fi
echo "  boltz warning string OK"
echo

# --- the run -------------------------------------------------------------------
# NOTE: PXDBENCH_BACKEND is deliberately NOT set. It is global, so pointing it at
# BoltzBackend also replaces the Protenix backend and the Protenix stage then
# silently emits no ptx_* columns. get_boltz() resolves the class directly.
mkdir -p "$OUT"
cp "$INPUT" "$OUT/campaign_input.json" 2>/dev/null || true

set -x
PYTHONPATH="$REPO" "$PY" -u -m pxdesign.runner.pipeline \
    --input_json_path "$INPUT" \
    --dump_dir "$OUT" \
    --load_checkpoint_dir "$CKPT" \
    --N_sample "$BACKBONES" \
    --N_max_runs 1 \
    --min_total_return "$KEEP" \
    --max_success_return "$KEEP" \
    --per_backbone_cap 1 \
    --eval.binder.eval_boltz true \
    --eval.binder.eval_complex false \
    --eval.binder.eval_binder_monomer false \
    --eval.binder.eval_protenix "$PROTENIX" \
    --eval.binder.eval_protenix_mini false \
    --eval.binder.tools.ptx.model_name protenix-v2 \
    --eval.binder.num_seqs "$SEQS" \
    --eval.binder.tools.mpnn.weights soluble \
    --eval.binder.tools.mpnn.temperature 0.1 \
    --eval.binder.tools.boltz.seeds_gate "$SEEDS_GATE" \
    --eval.binder.tools.boltz.workers "$WORKERS" \
    --eval.binder.tools.boltz.boltz_bin "$BOLTZ" \
    "${EXTRA[@]+"${EXTRA[@]}"}"
set +x

# --- afterwards ----------------------------------------------------------------
echo
echo "== results =="
SUMMARY=$(find "$OUT/design_outputs" -name summary.csv 2>/dev/null | head -1)
if [[ -z "$SUMMARY" ]]; then
    echo "  NO summary.csv - the run produced nothing. State=COMPLETED is not success."
    # WHY it produced nothing decides what to change next, and these read
    # identically without help. Measured over six campaigns: three were
    # policy_excluded_all, two were confirmation_failed, one crashed - three
    # different responses behind one message.
    SCORED=$(find "$OUT" -name sample_level_output.csv 2>/dev/null | head -1)
    if grep -q "Traceback (most recent call last)" "$OUT/../$(basename "$OUT").log" 2>/dev/null; then
        echo "  cause:      a traceback. This is a code failure, not a result."
    elif [[ -n "$SCORED" ]]; then
        echo "  cause:      designs were scored ($SCORED) but none were"
        echo "              selected. If an epitope policy is enforced this is"
        echo "              most likely eligibility: see"
        echo "              (nothing was ever eligible, or something was eligible"
        echo "              and did not confirm at gate depth)."
    else
        echo "  cause:      nothing was scored at all - look earlier than selection."
    fi
    exit 1
fi
echo "  summary:    $SUMMARY"
echo "  structures: $(dirname "$SUMMARY")/boltz_docked/"
echo "  rows:       $(( $(wc -l < "$SUMMARY") - 1 ))"

# A summary full of boltz_failed rows is not a result. This once printed "rows: 8"
# for a run in which every fold had crashed (GPU out of memory).
SCORED=$("$PY" - "$SUMMARY" <<'PYEOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
print(sum(1 for r in rows if r.get("bz_status") == "ok"))
PYEOF
)
echo "  scored ok:  $SCORED"
if [[ "${SCORED:-0}" -eq 0 ]]; then
    echo "  FAIL: no row has bz_status=ok - every fold failed. Look for CUDA/out-of-memory" >&2
    echo "        errors in the log (e.g. concurrent GPU jobs, or --workers too high)." >&2
    exit 1
fi

if [[ -f "$OUT/resolved_hotspots.json" ]]; then
    echo
    echo "== epitope check =="
    PYTHONPATH="$REPO" "$PY" "$REPO/scripts/hotspot_e2e_check.py" "$OUT" || true
fi

cat <<'EOF'

Reminders
  * bz_gate_egfr_provisional_v1 is PROVISIONAL and unvalidated. It cannot early-stop
    a campaign without --allow_provisional_early_stop, and a pass is not a binder.
  * Never compare bz_* scores across runs. A design's score depends on which other
    designs shared its Boltz invocation; only rows sharing a bz_final_batch_id
    are comparable.
  * Epitope engagement is REPORTED, not ranked on. Check ep_best_patch_frac before
    picking designs - it has been measured to anti-correlate with bz_ipsae.
EOF
