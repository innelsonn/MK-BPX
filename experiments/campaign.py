"""Seeded experimental campaign.

Usage:  python experiments/campaign.py ALGO SET [--out DIR]

ALGO  VNS    basic VNS, k_max = p (control: MK-BPX with the crossover switched off)
      BPX    MK-BPX (Algorithm 1 of the paper)
      VNS20  VNS with k_max = min(p, 20) (tuned control)
      BPX4   MK-BPX with four copies of the elite solution fused after every round
SET   S0  40 OR-Library instances, 30 runs, time limits of data/reference/orlib_time_limits.csv
      S1  fl1400, u1432, rl1304 (10 s) and d2103 (20 s), nine values of p, 10 runs
      S2  pcb3038 (30 s), fl3795 (40 s), rl5934 (60 s), 5 runs
      S3  long runs: five times the limit of S1 on fl1400, u1432, d2103 with p >= 100, 5 runs
--out results directory (default: results/). Complete instance-p pairs already present in the
      output file are kept and skipped, so an interrupted campaign can simply be restarted.

Writes DIR/<SET>_<ALGO>.csv (one row per run) and DIR/<SET>_<ALGO>_trace.json (time and value of
every improvement). Seed of a run = 10^7 * algorithm id + 100 * pair id + run index.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import json, csv, numpy as np
from mkbpx import core as K, instances as inst
ALGOS = {'VNS': (1, 0, 0, 1), 'BPX': (2, 1, 0, 1), 'VNS20': (3, 0, 20, 1), 'BPX4': (4, 1, 0, 4)}   # id, use_px, kmax (0 = p), N
def pairs(setname):
    out = []
    if setname == 'S0':
        t = {r['inst']: float(r['time_limit_s']) for r in csv.DictReader(open(os.path.join(inst.DATA, 'reference', 'orlib_time_limits.csv')))}
        for i in range(1, 41):
            out.append((f'pmed{i}', inst.PMED[i][1], max(0.02, t[f'pmed{i}']), 30))
    elif setname == 'S1':
        for nm, tl in (('fl1400', 10.), ('u1432', 10.), ('rl1304', 10.), ('d2103', 20.)):
            for p in (5, 10, 20, 50, 100, 200, 300, 400, 500): out.append((nm, p, tl, 10))
    elif setname == 'S2':
        for p in (5, 10, 20, 50, 100, 150, 200, 300, 400, 500): out.append(('pcb3038', p, 30., 5))
        for p in (50, 100, 150, 200, 300, 400, 500): out.append(('fl3795', p, 40., 5))
        for p in (200, 300, 400, 500): out.append(('rl5934', p, 60., 5))
    elif setname == 'S3':      # long runs: five times the time limit, same seeds as runs 0-4 of S1
        for nm, tl in (('fl1400', 50.), ('u1432', 50.), ('d2103', 100.)):
            for p in (100, 200, 300, 400, 500): out.append((nm, p, tl, 5))
    return out
PAIRID = {}
for sn in ('S0', 'S1', 'S2', 'S3'):
    for q in pairs(sn): PAIRID.setdefault((q[0], q[1]), len(PAIRID))
def main():
    argv = sys.argv[1:]; OUT = os.path.join(ROOT, 'results')
    if '--out' in argv:
        k = argv.index('--out'); OUT = argv[k + 1]; del argv[k:k + 2]
    if len(argv) != 2 or argv[0] not in ALGOS or argv[1] not in ('S0', 'S1', 'S2', 'S3'):
        print(__doc__); sys.exit(1)
    algo, setname = argv
    os.makedirs(OUT, exist_ok=True)
    aid, use_px, kmax0, N = ALGOS[algo]
    path = os.path.join(OUT, f'{setname}_{algo}.csv'); tpath = os.path.join(OUT, f'{setname}_{algo}_trace.json')
    header = 'algo inst n p opt tl run seed best gap iters imp_plain imp_px px_calls px_take comps comps_taken scans time t_hit viol dsum'.split()
    done = set(); keep = []
    if os.path.exists(path):          # resume: keep complete pairs only
        rows = list(csv.reader(open(path)))[1:]
        need = {(q[0], q[1]): q[3] for q in pairs(setname)}
        cnt = {}
        for r in rows: cnt[(r[1], int(r[3]))] = cnt.get((r[1], int(r[3])), 0) + 1
        done = {k for k, v in cnt.items() if v == need.get(k)}
        keep = [r for r in rows if (r[1], int(r[3])) in done]
    f = open(path, 'w', newline=''); w = csv.writer(f); w.writerow(header); w.writerows(keep); f.flush()
    traces = json.load(open(tpath)) if os.path.exists(tpath) else {}
    cur = None
    for name, p, tl, R in pairs(setname):
        if (name, p) in done: continue
        C = inst.load_matrix(name)
        if cur != name:
            order = K.build_order(C); cur = name
            K.run_main(C, order, min(5, p), 1, 0, 5, 2, 1, 0., 0.05); K.run_main(C, order, min(5, p), 0, 0, 5, 1, 1, 0., 0.05)
        o = inst.opt(name, p); kmax = p if kmax0 == 0 else min(p, kmax0)
        for r in range(R):
            seed = aid * 10**7 + PAIRID[(name, p)] * 100 + r
            z, sol, st, tr = K.run_main(C, order, p, use_px, 0, kmax, N, seed, float(o), tl)
            assert abs(K.eval_sol(sol, C) - z) < 1e-6 and len(set(sol.tolist())) == p, (name, p, z, o)
            w.writerow([algo, name, C.shape[0], p, o, tl, r, seed, int(round(z)), round(100 * (z - o) / o, 6)] + [int(st[k]) for k in range(8)] + [round(st[8], 3), round(st[9], 4), int(st[10]), int(st[11])])
            traces[f'{name}_{p}_{r}'] = [[round(a, 4), int(round(b))] for a, b in tr]
        f.flush(); json.dump(traces, open(tpath, 'w')); print(algo, name, p, flush=True)
    f.close()
if __name__ == '__main__':
    main()
