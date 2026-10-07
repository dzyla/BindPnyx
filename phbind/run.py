"""Single entry point for an agent.  python phbind/run.py <command> [args]

  status                          counted artifacts, GPU jobs, next step
  config [path]                   print the validated effective config (defaults + overrides)
  scorers                         every registered scorer with its validation record and which roles it may play
  require <scorer> <screen|gate>  exit 0 if allowed, 2 with the measured reason if refused
  validate <scorer> <set.csv> <out_dir> [--write]    positive controls + reference set -> decision (phbind/validate.py)
  log <name> <decision> <json-change> <json-result> <hypothesis>     append to the experiment log
Stages keep their own scripts (s0_target, s1_queue, s2_prescreen, s3_gate, s4_variants, s8_assemble); read docs/AGENT_PLAYBOOK_PHBIND.md first.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def main(a):
    if not a or a[0] in ("-h", "--help"): print(__doc__); return 0
    cmd = a[0]
    if cmd == "status":
        from phbind import status; print(json.dumps(status.report(), indent=1)); return 0
    if cmd == "config":
        from phbind import config; print(json.dumps(config.load(a[1] if len(a) > 1 else None), indent=1)); return 0
    if cmd == "scorers":
        from phbind import scorers
        for n, s in scorers.SCORERS.items():
            rec = scorers.records().get(n, {}); print(f"{n:9s} prefix={s.prefix:6s} ~{s.s_per_design:>4.0f}s/design  status={rec.get('status', 'untested'):10s} roles={rec.get('roles')}  {rec.get('why', '')[:90]}")
        return 0
    if cmd == "require":
        from phbind import scorers
        try: scorers.require(a[1], a[2]); print(f"{a[1]} allowed for {a[2]}"); return 0
        except scorers.ScorerRefused as e: print("REFUSED:", e); return 2
    if cmd == "validate":
        from phbind import validate; validate.main(a[1:]); return 0
    if cmd == "log":
        from phbind import experiments; print(json.dumps(experiments.log(a[1], a[5], json.loads(a[3]), json.loads(a[4]), a[2]), indent=1)); return 0
    print(__doc__); return 1

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
