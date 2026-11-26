#!/usr/bin/env bash
# Third campaign: the original POPSTAR program against MK-BPX and VNS on one machine,
# blocked by instance-p pair (see experiments/campaign_blocked.py).
#
#   experiments/run_campaign3.sh [OUT]          (default OUT = results_orig_new)
#
# Needs third_party/popstar/popstar (tools/get_popstar.sh) and a machine with at least two cores:
# two processes run at any time, pinned to cores 0 and 1.  Complete blocks are kept, so the
# script can simply be started again if it is interrupted.  About 5 hours.
# When it ends:  python experiments/analyze_orig.py DEST --results OUT
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
OUT="${1:-results_orig_new}"; mkdir -p "$OUT"; PY="${PYTHON:-python3}"
{ date; uname -a; lscpu; free -h; $PY -c "import sys,numpy,numba,scipy,pandas;print('python',sys.version.split()[0],'numpy',numpy.__version__,'numba',numba.__version__,'scipy',scipy.__version__,'pandas',pandas.__version__)"; g++ --version | head -1; } > "$OUT/machine.txt" 2>&1
for g in test_small test_big dev orlib; do
    if [ ! -f "$OUT/.done_$g" ]; then
        echo "$(date +%T) start $g"
        $PY experiments/campaign_blocked.py "$g" --out "$OUT" && touch "$OUT/.done_$g"
        echo "$(date +%T) end $g"
    fi
done
echo "third campaign finished: logs in $OUT"
