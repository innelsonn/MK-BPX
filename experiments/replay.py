"""Replay one logged run of VNS, VNS20, MK-BPX or MK-BPX4 exactly, on any machine.

Usage:  python experiments/replay.py LOG.csv INSTANCE P RUN
Example python experiments/replay.py results/S1_BPX.csv fl1400 100 3

A run is determined by its seed; only the point at which the time limit stopped it depends on the
machine.  The script repeats the run with the logged seed for the logged number of iterations and
compares the value and the counters with the log.  It works with the logs written by
campaign.py (columns seed, iters, best); the algorithm is read from the column `algo`.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'experiments'))
import pandas as pd
from mkbpx import core as K, instances as inst
import campaign as CP


def main():
    if len(sys.argv) != 5:
        print(__doc__); sys.exit(1)
    log, name, p, run = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    d = pd.read_csv(log); d = d[(d.inst == name) & (d.p == p) & (d.run == run)]
    if len(d) != 1: sys.exit('run not found in the log')
    r = d.iloc[0]; aid, use_px, kmax0, N = CP.ALGOS[r.algo]; kmax = p if kmax0 == 0 else min(p, kmax0)
    C = inst.load_matrix(name); order = K.build_order(C)
    z, sol, st, tr = K.run_main_ext(C, order, p, use_px, 0, kmax, N, int(r.seed), float(inst.opt(name, p)), 1e300, int(r.iters), 0, 0.0)
    assert abs(K.eval_sol(sol, C) - z) < 1e-6
    print(f'{r.algo} {name} p={p} run {run} (seed {int(r.seed)})')
    print(f'  logged : value {int(r.best)}, {int(r.iters)} iterations, {int(r.imp_plain)} + {int(r.imp_px)} improvements, {r.time:.2f} s')
    print(f'  replay : value {int(round(z))}, {int(st[0])} iterations, {int(st[1])} + {int(st[2])} improvements, {st[8]:.2f} s on this machine')
    print('  improvements (time, value, iteration):'); [print(f'    {a:8.3f}  {int(b)}  {int(c)}') for a, b, c in tr]
    ok = int(round(z)) == int(r.best) and int(st[1]) == int(r.imp_plain) and int(st[2]) == int(r.imp_px)
    print('IDENTICAL' if ok else 'DIFFERENT'); sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
