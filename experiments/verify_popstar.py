"""Verification of the path-relinking of PopStar against a brute-force implementation.

For pairs of swap local optima of four OR-Library instances (independent local optima and a
local optimum with the local optimum reached after a kick), the path built by
popstar.path_relink is compared with a path in which every candidate swap is evaluated from
scratch, with the same tie-breaking.  The solution returned (best local minimum of the path)
must be identical.  Writes results_sota/verify_popstar.json.
"""
import os, sys, json, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mkbpx import core as K, instances as inst, popstar as P
rng = np.random.default_rng(5)
def brute(S1, S2, C):
    p = len(S1); sol = S1.copy(); in1 = set(S1.tolist()); in2 = set(S2.tolist())
    Dpos = [k for k in range(p) if sol[k] not in in2]; As = [int(x) for x in S2 if x not in in1]; q = len(Dpos)
    if q < 2: return 0, sol
    pc = [K.eval_sol(sol, C)]; sw = []
    rem = q
    for step in range(q):
        best = -1e300; bd = ba = -1; base = pc[-1]
        for d in range(rem):
            for a in range(rem):
                t = sol.copy(); t[Dpos[d]] = As[a]; pr = base - K.eval_sol(t, C)
                if pr > best: best = pr; bd = d; ba = a
        bi = Dpos[bd]; bj = As[ba]; Dpos[bd] = Dpos[rem-1]; As[ba] = As[rem-1]; rem -= 1
        sol[bi] = bj; pc.append(base - best); sw.append((bi, bj))
    assert sorted(sol.tolist()) == sorted(S2.tolist())
    bt = -1; bc = 1e300; eps = 1e-6
    for t in range(1, q):
        a = t-1
        while a > 0 and abs(pc[a]-pc[t]) <= eps: a -= 1
        b = t+1
        while b < q and abs(pc[b]-pc[t]) <= eps: b += 1
        if pc[a] > pc[t]+eps and pc[b] > pc[t]+eps and pc[t] < bc-eps: bt = t; bc = pc[t]
    if bt < 0: return 0, sol
    out = S1.copy()
    for t in range(bt): out[sw[t][0]] = sw[t][1]
    out.sort(); return 1, out
nf = 0; tot = 0
for name in ('pmed1', 'pmed5', 'pmed9', 'pmed14'):
    C = inst.load_matrix(name).copy(); p = inst.pmed_p(name); n = C.shape[0]; order = K.build_order(C)
    cnt = np.zeros(4, np.int64); pc = np.zeros(2, np.int64)
    for rep in range(12):
        np.random.seed(rep)
        A = K.ls_swap(np.sort(rng.choice(n, p, replace=False)).astype(np.int64), C, order, 10**8, cnt)[0]
        B = K.ls_swap(np.sort(rng.choice(n, p, replace=False)).astype(np.int64), C, order, 10**8, cnt)[0]
        if rep % 2: B = K.ls_swap(K.kick(A, n, min(p, 6), np.empty(p, np.int64), np.empty(n, np.int64)), C, order, 10**8, cnt)[0]
        f1, s1 = P.path_relink(A, B, C, order, pc); f2, s2 = brute(A, B, C)
        assert f1 == f2 and (f1 == 0 or (s1 == s2).all()), (name, rep, f1, f2)
        tot += 1; nf += f1
    # pool rules and greedy
    s = P.sample_greedy(C, p, np.arange(n).astype(np.int64)); assert len(set(s.tolist())) == p
print('path-relinking identical to brute force in', tot, 'pairs; with local minimum:', nf)
a = np.array([1,2,3,4,5,6]); b = np.array([1,2,3,7,8,9]); assert P.diff_sorted(a, b) == 3
os.makedirs(os.path.join(ROOT, 'results_sota'), exist_ok=True)
json.dump({'pairs': tot, 'with_local_minimum': int(nf), 'mismatch': 0}, open(os.path.join(ROOT, 'results_sota', 'verify_popstar.json'), 'w'))
print('OK')
