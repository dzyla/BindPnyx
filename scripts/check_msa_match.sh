#!/usr/bin/env bash
# Does a cached MSA actually apply to the sequence we hand the predictor?
#
# `grep -c '^>'` checks DEPTH and cannot see this. Boltz compares its MSA query
# line against the input sequence and, on ANY mismatch, silently substitutes a
# dummy - printing one warning in the middle of a progress bar:
#
#   Warning: MSA does not match input sequence, creating dummy.
#
# Measured on this project's reference target: a single difference at chain B
# position 5 (auth 516) threw away all 3,109 sequences, and every prediction on
# that target ran with chain B unconditioned.
#
# WHICH SIDE IS WRONG (corrected 2026-10-02, verified against UniProt P00533):
# the cached MSA is RIGHT and the shard is wrong. WT is ASPARAGINE at precursor
# position 540 = mature 516 (`...RECVDKCN LLEGEPREF...`), which is what the MSA
# and panel/target_obj2.json carry. The shard - and the 1NQL crystal it derives
# from - carries a LYSINE there. An earlier note here had this backwards and
# blamed the MSA for being "the assay construct"; it is the structure that
# departs from wild type, so the fix is a WT-matching shard, not a new MSA.
#
# Usage: scripts/check_msa_match.sh <sequence> <msa_dir> [label]
set -euo pipefail
seq="$1"; msa_dir="$2"; label="${3:-chain}"
# One comparison, not two that disagree: the Python module replicates boltz's
# tolerance (an all-MET-vs-UNK mismatch set is repaired, not discarded), which
# the exact string compare this script used to do got wrong in the strict
# direction.
#
# Loaded by file path, not imported through the package: pxdbench/__init__
# pulls in protenix, which a bare system python3 does not have, and msa_check
# itself needs only the standard library. That also means no environment path
# has to be named here.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 -c '
import importlib.util, sys
root, seq, msa_dir, label = sys.argv[1:5]
spec = importlib.util.spec_from_file_location(
    "msa_check", root + "/pxdbench/tools/boltz/msa_check.py")
mc = importlib.util.module_from_spec(spec)
sys.modules["msa_check"] = mc
spec.loader.exec_module(mc)
try:
    rec = mc.validate_msa_dir(seq, msa_dir, label=label)
except mc.MsaMismatch as e:
    print("FAIL " + str(e))
    raise SystemExit(1)
print("OK   %s: query matches, depth %d" % (rec["label"], rec["depth_unique"]))
' "$root" "$seq" "$msa_dir" "$label"
