"""Fidelity check of the re-implementation of PopStar against the published results.

Runs the heuristic exactly as published (one cycle: 32 multistart iterations with
path-relinking, 10 elite solutions, post-optimization; no time limit), nine runs per instance
as in Resende & Werneck (2004), on
  fl1400   with real-valued Euclidean distances (neither rounded nor truncated) and the 18
           values of p of their Table 6; deviation from the best-known values of that table
  orlib    the 40 OR-Library instances; deviation from the optimum
Usage:  python experiments/popstar_fidelity.py fl1400|orlib [OUT.csv]
Writes cls, inst, p, run, seed, value, ref, dev (%), pub_dev (published average deviation, %), time.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from mkbpx import core as K, instances as inst, popstar as P
# p: (best known, published average deviation in %), Resende & Werneck (2004), Table 6
FL = {10: (101249.47, 0.000), 20: (57857.55, 0.001), 30: (44013.48, 0.000), 40: (35002.52, 0.003), 50: (29089.78, 0.002), 60: (25161.12, 0.012),
      70: (22125.53, 0.002), 80: (19872.72, 0.018), 90: (17987.94, 0.013), 100: (16551.20, 0.051), 150: (12026.47, 0.068), 200: (9359.15, 0.017),
      250: (7741.51, 0.057), 300: (6620.92, 0.041), 350: (5720.91, 0.097), 400: (5006.83, 0.087), 450: (4474.96, 0.059), 500: (4047.90, 0.044)}
cls = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'results_sota', f'fidelity_{cls}.csv')
f = open(out, 'w'); f.write('cls,inst,p,run,seed,value,ref,dev,pub_dev,time\n')
def go(name, C, p, refv, pub):
    order = go.order
    for r in range(9):
        seed = 7000000 + 1000 * p + r
        z, sol, st, tr = P.run_popstar(C, order, p, 32, 10, seed, 0., 1e18, 1)
        assert abs(K.eval_sol(sol, C) - z) < 1e-5 * max(1, z) and len(set(sol.tolist())) == p
        f.write(f'{cls},{name},{p},{r},{seed},{z:.2f},{refv},{100 * (z - refv) / refv:.5f},{pub},{st[9]:.3f}\n'); f.flush()
    print(name, p, flush=True)
if cls == 'fl1400':
    xy = inst.read_coords('fl1400'); d = xy[:, None, :] - xy[None, :, :]
    C = np.ascontiguousarray(np.sqrt((d ** 2).sum(-1)))
    go.order = K.build_order(C); P.run_popstar(C, go.order, 5, 2, 3, 1, 0., 0.05, 0)
    for p, (bk, pub) in FL.items(): go('fl1400', C, p, bk, pub)
else:
    for i in range(1, 41):
        name = f'pmed{i}'; C = inst.load_matrix(name).copy(); go.order = K.build_order(C)
        if i == 1: P.run_popstar(C, go.order, 5, 2, 3, 1, 0., 0.05, 0)
        go(name, C, inst.pmed_p(name), inst.opt(name, inst.pmed_p(name)), 0.001)
f.close()
