"""Exhaustive verification of Proposition 1 and of the dynamic programme: for random pairs of local
optima, every subset of components is enumerated; the bound must hold for every subset and the value
returned by px must equal the best balanced subset.

Usage:  python experiments/verify_bpx.py     (writes results/verify.json; about 5 minutes)"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import itertools, json, numpy as np
from mkbpx import core as K, instances as inst
from numba import njit
@njit
def make(C, order, p, seed, nk):
    np.random.seed(seed); m = C.shape[0]; cnt = np.zeros(4, np.int64)
    perm = np.arange(m).astype(np.int64); pp = np.empty(p, np.int64); pm = np.empty(m, np.int64)
    S, zS, nS, dS = K.ls_full(K.random_subset(m, p, perm), C, order, 10**8, cnt)
    T, zT, nT, dT = K.ls_full(K.kick(S, m, nk, pp, pm), C, order, 10**8, cnt)
    par = np.empty(m, np.int64); cid = np.empty(m, np.int64); tag = np.zeros(m, np.int8); info = np.zeros(4)
    ch = K.px(S, zS, nS, dS, T, zT, nT, dT, m, par, tag, cid, info, 0)
    return S, zS, nS, dS, T, zT, nT, dT, ch, info
tot = dict(pairs=0, subsets=0, balanced=0, bound_viol=0, dp_mismatch=0, child_mismatch=0, tight=0, skipped=0, strict=0)
rng = np.random.default_rng(5)
for name, p in (('pmed10', 67), ('pmed25', 167), ('pmed40', 90), ('rl1304', 100), ('rl1304', 300)):
    C = inst.load_matrix(name); order = K.build_order(C); n = C.shape[0]
    for rep in range(200):
        nk = int(rng.integers(2, min(p // 3, 25)))
        S, zS, nS, dS, T, zT, nT, dT, ch, info = make(C, order, p, int(rng.integers(1, 10**9)), nk)
        setS, setT = set(S.tolist()), set(T.tolist()); D = sorted(setS - setT); A = sorted(setT - setS)
        if not D: continue
        par = {x: x for x in D + A}
        def find(x):
            while par[x] != x: par[x] = par[par[x]]; x = par[x]
            return x
        for i in range(n):
            if nS[i] in par and nT[i] in par and nS[i] not in setT and nT[i] not in setS: par[find(nS[i])] = find(nT[i])
        comp = {}
        for x in D + A: comp.setdefault(find(x), {'D': [], 'A': [], 'd': 0.0})['D' if x in setS else 'A'].append(x)
        for i in range(n):
            s, t = nS[i], nT[i]
            if s == t: continue
            if s in par and s not in setT: comp[find(s)]['d'] += dT[i] - dS[i]
            elif t in par and t not in setS: comp[find(t)]['d'] += dT[i] - dS[i]
        cs = list(comp.values())
        assert abs(sum(c['d'] for c in cs) - (zT - zS)) < 1e-6
        if len(cs) > 14: tot['skipped'] += 1; continue
        tot['pairs'] += 1; best = 0.0
        for r in range(0, len(cs) + 1):
            for sub in itertools.combinations(range(len(cs)), r):
                Y = set(S.tolist())
                for j in sub: Y -= set(cs[j]['D']); Y |= set(cs[j]['A'])
                bound = zS + sum(cs[j]['d'] for j in sub); tot['subsets'] += 1
                fY = C[:, sorted(Y)].min(axis=1).sum() if len(Y) > 0 else np.inf
                if len(Y) == p:
                    tot['balanced'] += 1
                    if fY > bound + 1e-6: tot['bound_viol'] += 1
                    if abs(fY - bound) < 1e-6: tot['tight'] += 1
                    best = min(best, bound - zS)
                else:
                    # the bound is stated for any K; check it for unbalanced selections too
                    if fY > bound + 1e-6: tot['bound_viol'] += 1
        if abs((info[2] - zS) - best) > 1e-6: tot['dp_mismatch'] += 1
        if abs(K.eval_sol(ch, C) - info[2]) > 1e-6 and K.eval_sol(ch, C) > info[2] + 1e-6: tot['child_mismatch'] += 1
        if info[2] < min(zS, zT) - 1e-6: tot['strict'] += 1
    print(name, p, tot, flush=True)
json.dump(tot, open(os.path.join(ROOT, 'results', 'verify.json'), 'w'))
