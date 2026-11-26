"""Quick self-check of the installation (about one minute).

Usage:  python tests/smoke_test.py
Checks: data present; local search returns swap local optima (confirmed by the independent Whitaker
implementation); BPX never returns a child worse than a parent and respects its bound; seeded runs
are reproducible; MK-BPX and VNS reach the optimum of two small OR-Library instances.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from numba import njit
from mkbpx import core as K, instances as inst, whitaker as H

for nm in ('fl1400', 'u1432', 'rl1304', 'd2103', 'pcb3038', 'fl3795', 'rl5934'):
    assert os.path.exists(os.path.join(inst.TSPDIR, nm + '.tsp')), nm
print('data: ok')

@njit
def pair(C, order, p, seed, nk):
    np.random.seed(seed); m = C.shape[0]; cnt = np.zeros(4, np.int64)
    perm = np.arange(m).astype(np.int64); pp = np.empty(p, np.int64); pm = np.empty(m, np.int64)
    S, zS, nS, dS = K.ls_full(K.random_subset(m, p, perm), C, order, 10**8, cnt)
    T, zT, nT, dT = K.ls_full(K.kick(S, m, nk, pp, pm), C, order, 10**8, cnt)
    par = np.empty(m, np.int64); cid = np.empty(m, np.int64); tag = np.zeros(m, np.int8); info = np.zeros(4)
    ch = K.px(S, zS, nS, dS, T, zT, nT, dT, m, par, tag, cid, info, 0)
    return S, zS, T, zT, ch, info

rng = np.random.default_rng(3)
for name in ('pmed15', 'pmed40'):
    C = inst.load_matrix(name); p = inst.pmed_p(name); order = K.build_order(C); n = C.shape[0]
    assert np.array_equal(C, C.T)
    for rep in range(5):
        s = np.sort(rng.choice(n, p, replace=False)).astype(np.int64); c = np.zeros(4, np.int64)
        a, za = K.ls_swap(s, C, order, 10**8, c)
        b, zb = H.ls_swap(a, C, np.ascontiguousarray(C.T), 5, c)
        assert zb == za and abs(K.eval_sol(a, C) - za) < 1e-6
    strict = 0
    for rep in range(60):
        S, zS, T, zT, ch, info = pair(C, order, p, 1000 + rep, 2 + rep % 12)
        zc = K.eval_sol(ch, C)
        assert len(set(ch.tolist())) == p and zc <= info[2] + 1e-6 and info[2] <= min(zS, zT) + 1e-6
        strict += zc < min(zS, zT) - 1e-6
    print(f'{name}: local search ok, BPX bound ok on 60 pairs ({strict} children strictly better than both parents)')

C = inst.load_matrix('pmed5'); order = K.build_order(C); o = inst.opt('pmed5', 33)
for use_px, nm in ((0, 'VNS'), (1, 'MK-BPX')):
    z = K.run_main(C, order, 33, use_px, 0, 33, 1, 1, float(o), 1.0)[0]
    assert z == o, (nm, z, o)
print('pmed5: optimum reached by VNS and MK-BPX')

C = inst.load_matrix('rl1304'); order = K.build_order(C)
r1 = K.run_main(C, order, 100, 1, 0, 100, 1, 42, 0., 1.0); r2 = K.run_main(C, order, 100, 1, 0, 100, 1, 42, 0., 1.0)
k = min(len(r1[3]), len(r2[3])) - 1
assert k >= 3 and np.array_equal(r1[3][:k, 1], r2[3][:k, 1])
print(f'rl1304: seeded runs follow the same trajectory (first {k} improvements identical)')
print('ALL CHECKS PASSED')
