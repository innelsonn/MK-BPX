"""Development runs for the time-limited PopStar: number of multistart iterations per cycle.

Usage:  python experiments/development_popstar.py INSTANCE MAXIT[,MAXIT...] [OUT.csv]
Same protocol as development.py: p = 100, 300, 500; 10 s on rl1304 and 40 s on pcb3038; two runs.
Only the two development instances are used, as for MK-BPX.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from mkbpx import core as K, instances as inst, popstar as P
name = sys.argv[1]; its = [int(x) for x in sys.argv[2].split(',')]
out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ROOT, 'results_sota', 'dev_popstar.csv')
tl = {'rl1304': 10., 'pcb3038': 40.}[name]
C = inst.load_matrix(name); order = K.build_order(C)
P.run_popstar(C, order, 5, 2, 3, 1, 0., 0.05, 0)
new = not os.path.exists(out)
f = open(out, 'a')
if new: f.write('inst,p,maxit,esize,run,seed,tl,best,best_at_tl,gap_at_tl,iters,cycles,time\n')
for maxit in its:
    for p in (100, 300, 500):
        o = inst.opt(name, p)
        for r in range(2):
            seed = 9000000 + 1000 * maxit + p + r
            z, sol, st, tr = P.run_popstar(C, order, p, maxit, 10, seed, float(o), tl, 0)
            assert abs(K.eval_sol(sol, C) - z) < 1e-6 and len(set(sol.tolist())) == p
            zt = min([b for a, b in tr if a <= tl] or [float('nan')])
            f.write(f'{name},{p},{maxit},10,{r},{seed},{tl},{int(round(z))},{int(round(zt))},{100 * (zt - o) / o:.6f},{int(st[0])},{int(st[4])},{st[9]:.3f}\n'); f.flush()
            print(name, p, maxit, r, round(100 * (zt - o) / o, 4), flush=True)
