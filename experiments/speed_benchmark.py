"""Time per neighbourhood scan: fast interchange (Whitaker, 1983) against the local search of the
study, complete local searches from the same three random solutions per case.

Usage:  python experiments/speed_benchmark.py     (writes results/speed_benchmark.csv)"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import time, csv, numpy as np
from mkbpx import core as K, instances as inst, whitaker as H
rng = np.random.default_rng(11)
cases = [(f'pmed{i}', inst.PMED[i][1]) for i in (5, 10, 15, 20, 25, 30, 34, 37, 40, 38, 39)] + [('fl1400', p) for p in (10, 50, 100, 300, 500)] + [('d2103', p) for p in (10, 50, 100, 300, 500)]
w = csv.writer(open(os.path.join(ROOT, 'results', 'speed_benchmark.csv'), 'w', newline='')); w.writerow(['inst', 'n', 'p', 'scans_w', 't_w', 'scans_new', 't_new', 'ratio', 'both_local_opt'])
first = True
for name, p in cases:
    C = inst.load_matrix(name); order = K.build_order(C); n = C.shape[0]
    tw = tn = 0.; sw = sn = 0; ok = True
    for rep in range(3):
        s = np.sort(rng.choice(n, p, replace=False)).astype(np.int64); c1 = np.zeros(4, np.int64); c2 = np.zeros(4, np.int64)
        if first: K.ls_swap(s, C, order, 1, c1); H.ls_swap(s, C, C, 1, c2); c1[:] = 0; c2[:] = 0; first = False
        t = time.perf_counter(); a, za = K.ls_swap(s, C, order, 10**8, c1); tn += time.perf_counter() - t; sn += c1[0]
        t = time.perf_counter(); b, zb = H.ls_swap(s, C, C, 10**8, c2); tw += time.perf_counter() - t; sw += c2[0]
        c3 = np.zeros(4, np.int64); a2, za2 = H.ls_swap(a, C, C, 5, c3); ok &= bool(za2 == za and abs(K.eval_sol(a, C) - za) < 1e-6)
    w.writerow([name, n, p, sw, round(tw, 5), sn, round(tn, 5), round((tw / sw) / (tn / sn), 2), ok]); print(name, p, round((tw / sw) / (tn / sn), 1), ok, flush=True)
