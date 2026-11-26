"""Second campaign: PopStar (GRASP with path-relinking, Resende & Werneck 2004) under the
conditions of the main campaign.

Usage:  python experiments/campaign_sota.py SET [--out DIR] [--maxit K] [--elite E]

SET   S0, S1, S2 or S3, exactly as in campaign.py (same pairs, time limits and numbers of runs).
Writes DIR/<SET>_POPSTAR.csv and DIR/<SET>_POPSTAR_trace.json (default DIR: results_sota/).
Seed of a run = 10^7 * 5 + 100 * pair id + run index (algorithm id 5).

The paper runs this script side by side with
    python experiments/campaign.py BPX SET --out results_sota
(one process per core), so that PopStar and MK-BPX are measured on the same machine, at the
same moment and under the same load.  `best_tl` is the best value found within the time limit
(taken from the improvement trace); `best` may include an improvement completed just after it.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'experiments'))
import json, csv, numpy as np
from mkbpx import core as K, instances as inst, popstar as P
import campaign as CP
AID = 5
def main():
    argv = sys.argv[1:]; OUT = os.path.join(ROOT, 'results_sota'); maxit = 32; esize = 10
    for flag in ('--out', '--maxit', '--elite'):
        if flag in argv:
            k = argv.index(flag); v = argv[k + 1]; del argv[k:k + 2]
            if flag == '--out': OUT = v
            elif flag == '--maxit': maxit = int(v)
            else: esize = int(v)
    if len(argv) != 1 or argv[0] not in ('S0', 'S1', 'S2', 'S3'):
        print(__doc__); sys.exit(1)
    setname = argv[0]; algo = 'POPSTAR'
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f'{setname}_{algo}.csv'); tpath = os.path.join(OUT, f'{setname}_{algo}_trace.json')
    header = 'algo inst n p opt tl run seed best best_tl gap iters pr_calls pr_found gens cycles imp_ms imp_pr imp_po scans time t_hit pr_steps maxit elite'.split()
    done = set(); keep = []
    if os.path.exists(path):
        rows = list(csv.reader(open(path)))[1:]
        need = {(q[0], q[1]): q[3] for q in CP.pairs(setname)}
        cnt = {}
        for r in rows: cnt[(r[1], int(r[3]))] = cnt.get((r[1], int(r[3])), 0) + 1
        done = {k for k, v in cnt.items() if v == need.get(k)}
        keep = [r for r in rows if (r[1], int(r[3])) in done]
    f = open(path, 'w', newline=''); w = csv.writer(f); w.writerow(header); w.writerows(keep); f.flush()
    traces = json.load(open(tpath)) if os.path.exists(tpath) else {}
    cur = None
    for name, p, tl, R in CP.pairs(setname):
        if (name, p) in done: continue
        C = inst.load_matrix(name)
        if cur != name:
            order = K.build_order(C); cur = name
            P.run_popstar(C, order, min(5, p), 2, 3, 1, 0., 0.05, 0)
        o = inst.opt(name, p)
        for r in range(R):
            seed = AID * 10**7 + CP.PAIRID[(name, p)] * 100 + r
            z, sol, st, tr = P.run_popstar(C, order, p, maxit, esize, seed, float(o), tl, 0)
            assert abs(K.eval_sol(sol, C) - z) < 1e-6 and len(set(sol.tolist())) == p, (name, p, z, o)
            ztl = min([b for a, b in tr if a <= tl] or [float('inf')])
            w.writerow([algo, name, C.shape[0], p, o, tl, r, seed, int(round(z)), int(round(ztl)) if ztl < 1e299 else -1, round(100 * (z - o) / o, 6)]
                       + [int(st[k]) for k in range(9)] + [round(st[9], 3), round(st[10], 4), int(st[11]), maxit, esize])
            traces[f'{name}_{p}_{r}'] = [[round(a, 4), int(round(b))] for a, b in tr]
        f.flush(); json.dump(traces, open(tpath, 'w')); print(algo, name, p, flush=True)
    f.close()
if __name__ == '__main__':
    main()
