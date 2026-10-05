#!/usr/bin/env bash
# Reproduce the ProteinBase scorer benchmark (phase 1). ~2 h on one RTX 5090.
# Needs: ./scripts/setup.sh done; internet for the ColabFold MSA server on first run.
# Run on an IDLE GPU: a concurrent job caused CUDA OOM / cusolver failures here.
set -euo pipefail
cd "$(dirname "$0")"
CSV=proteinbase_all_data_28_01_2026.csv
[ -f $CSV ] || curl -sO https://storage.proteinbase.com/$CSV
python3 prepare_labels.py                 # CSV -> labels.pkl
python3 make_inputs.py                    # -> inputs/manifest.csv
python3 make_msas.py                      # ColabFold server, one MSA per target
python3 -u score_boltz.py                 # Boltz-2
for arm in ptx_v2 ptx_v1 ptx05_fast; do python3 -u score_protenix.py $arm; done
python3 analyze.py                        # AUROC table -> out/auroc.csv
