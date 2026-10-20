#!/bin/bash
# BLOCK 0 - everything that is cheap to check before a GPU is committed.
#
# Each check here corresponds to a batch this campaign actually lost or nearly lost. None of them
# takes more than a second, and the whole point is that they run BEFORE hours of compute, not
# after. Exit status is the number of failed checks, so this can gate a queue:
#     bash preflight.sh $B && GPU=0 bash run_queue.sh $B af3
#
# Usage: preflight.sh <batch_dir> [expected_samples_per_case]
set -u
B=$(readlink -f "${1:?usage: preflight.sh <batch_dir> [samples_per_case]}")
NS=${2:-25}
# $2 is samples-per-case, NOT the oracle - blocks 2 and 3 take an oracle as $2 and this one does
# not, so `preflight.sh $B af3` is an easy slip. Caught here with a sentence, because under
# `set -u` a non-numeric NS instead dies inside an arithmetic expansion fifty lines below as
# "af3: unbound variable", which reads like a bug in this script.
case "$NS" in
  ''|*[!0-9]*) echo "[FATAL] second argument is samples-per-case, a number (got '$NS'). This"\
                    "block takes no oracle argument; blocks 2 and 3 do."; exit 1 ;;
esac
fail=0
ok()   { printf '  ok   %s\n' "$1"; }
bad()  { printf '  FAIL %s\n' "$1"; fail=$((fail+1)); }

echo "preflight: $B"

# 1. the scorer reads these; the builder must have written them
for f in cases.txt designs.csv manifest.json; do
  [ -s "$B/$f" ] && ok "$f present" || bad "$f missing or empty - the scorer reads it"
done

# 2. a case name with a dot is truncated by some runners at the LAST dot
if [ -s "$B/cases.txt" ]; then
  d=$(grep -c '\.' "$B/cases.txt" || true)
  [ "$d" -eq 0 ] && ok "no dots in case names" || bad "$d case name(s) contain a dot"
fi

# 3. controls first, so an interrupted queue still has its control block
if [ -s "$B/cases.txt" ]; then
  first=$(head -1 "$B/cases.txt")
  case "$first" in ctl_*) ok "cases.txt starts with a control ($first)" ;;
    *) bad "cases.txt starts with $first - put controls first" ;; esac
  np=$(grep -c '^ctl_\(pos\|ago\)' "$B/cases.txt" || true)
  nn=$(grep -c '^ctl_\(neg\|antag\)' "$B/cases.txt" || true)
  [ "$np" -ge 3 ] && ok "$np positive controls" || bad "$np positive controls, need >= 3 (the band comes from them)"
  [ "$nn" -ge 1 ] && ok "$nn negative control(s)" || bad "no negative control - the batch cannot be certified"
fi

# 4. every case has an input
miss=0
while read -r c; do
  [ -z "$c" ] && continue
  [ -f "$B/json/$c.json" ] || { miss=$((miss+1)); echo "       no input for $c"; }
done < "$B/cases.txt"
[ "$miss" -eq 0 ] && ok "every case has an input json" || bad "$miss case(s) have no input json"

# 5. referenced MSA files must exist NOW, not at fold time
msa=$(grep -ho '"unpairedMsaPath": *"[^"]*"' "$B"/json/*.json 2>/dev/null | sed 's/.*: *"//;s/"$//' | sort -u)
m=0
for p in $msa; do [ -r "$p" ] || { m=$((m+1)); echo "       unreadable: $p"; }; done
[ "$m" -eq 0 ] && ok "referenced MSA paths readable ($(echo "$msa" | grep -c . ) distinct)" || bad "$m MSA path(s) unreadable"

# 6. disk: structures are large and a full disk fails late and silently
avail=$(df -Pk "$B" | awk 'NR==2{print int($4/1048576)}')
[ "$avail" -ge 20 ] && ok "${avail} GB free on the batch filesystem" || bad "only ${avail} GB free"

# 7. what is already there - this is a resumable queue, so report, do not judge
if [ -d "$B/out" ]; then
  # count exactly what the scorer counts: per-seed sample files, not the ranked copy each
  # case also writes, or the number reads high and a short batch looks complete
  have=$(find "$B/out" -name '*_model.cif' -path '*seed-*' 2>/dev/null | wc -l)
  want=$(( $(grep -cve '^[[:space:]]*$' "$B/cases.txt") * NS ))
  ok "out/ holds $have of $want expected structures (queue will skip complete cases)"
else
  ok "no out/ yet - fresh batch"
fi

echo
[ "$fail" -eq 0 ] && echo "preflight PASSED" || echo "preflight FAILED: $fail check(s)"
exit "$fail"
