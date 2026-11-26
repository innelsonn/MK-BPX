"""Development prototype kept for the record (not part of MK-BPX).

run_pop: N trajectories of kick + local search + BPX acceptance, fused by BPX every G rounds
(fmode 0: all copies replaced by the best after fusing; fmode 1: copies kept distinct).
It produced the rows with two and four elite solutions of the development table of the paper.
"""
import numpy as np
from numba import njit
from .core import INF, now, random_subset, ls_full, kick, px, _same


@njit(cache=True)
def run_pop(C, order, p, N, G, fmode, kmax, seed, target, time_limit):
    """Population prototype: N trajectories of kick + LS + PX acceptance,
    fused by PX every G rounds."""
    np.random.seed(seed)
    m = C.shape[0]
    n = m
    t0 = now()
    perm = np.arange(m)
    perm_p = np.empty(p, dtype=np.int64)
    perm_m = np.empty(m, dtype=np.int64)
    par = np.empty(m, dtype=np.int64)
    cid = np.empty(m, dtype=np.int64)
    tag = np.zeros(m, dtype=np.int8)
    info = np.zeros(4, dtype=np.float64)
    counters = np.zeros(4, dtype=np.int64)
    stats = np.zeros(12, dtype=np.float64)
    pool = np.empty((N, p), dtype=np.int64)
    cost = np.empty(N, dtype=np.float64)
    near = np.empty((N, n), dtype=np.int64)
    dist = np.empty((N, n), dtype=np.float64)
    kk = np.ones(N, dtype=np.int64)
    for i in range(N):
        s = random_subset(m, p, perm)
        pool[i], cost[i], near[i], dist[i] = ls_full(s, C, order, 100000000, counters)
    rnd = 0
    while now() - t0 < time_limit:
        rnd += 1
        for i in range(N):
            if now() - t0 >= time_limit:
                break
            stats[0] += 1
            k = min(kk[i], m - p, p)
            cand = kick(pool[i], m, k, perm_p, perm_m)
            s2, z2, n2, d2 = ls_full(cand, C, order, 100000000, counters)
            improved = False
            if z2 < cost[i] - 1e-6:
                ch = px(s2, z2, n2, d2, pool[i], cost[i], near[i], dist[i], m, par, tag, cid, info)
                if info[1] > 0:
                    s3, z3, n3, d3 = ls_full(ch, C, order, 100000000, counters)
                    if z3 < z2 - 1e-6:
                        s2 = s3; z2 = z3; n2 = n3; d2 = d3
                pool[i] = s2; cost[i] = z2; near[i] = n2; dist[i] = d2
                improved = True
            else:
                ch = px(pool[i], cost[i], near[i], dist[i], s2, z2, n2, d2, m, par, tag, cid, info)
                if info[1] > 0:
                    s3, z3, n3, d3 = ls_full(ch, C, order, 100000000, counters)
                    if z3 < cost[i] - 1e-6:
                        pool[i] = s3; cost[i] = z3; near[i] = n3; dist[i] = d3
                        improved = True
                        stats[7] += 1
            if improved:
                kk[i] = 1
            else:
                kk[i] += 1
                if kk[i] > kmax:
                    kk[i] = 1
        if N > 1 and rnd % G == 0:
            ib = 0
            for i in range(1, N):
                if cost[i] < cost[ib]:
                    ib = i
            for j in range(N):
                if j == ib:
                    continue
                ch = px(pool[ib], cost[ib], near[ib], dist[ib], pool[j], cost[j], near[j], dist[j], m, par, tag, cid, info)
                if info[1] > 0:
                    s3, z3, n3, d3 = ls_full(ch, C, order, 100000000, counters)
                    if z3 < cost[ib] - 1e-6:
                        pool[ib] = s3; cost[ib] = z3; near[ib] = n3; dist[ib] = d3
                        stats[8] += 1
            if fmode == 0:
                for j in range(N):
                    if j != ib:
                        pool[j] = pool[ib]; cost[j] = cost[ib]; near[j] = near[ib]; dist[j] = dist[ib]
                        kk[j] = kk[ib]
            else:
                for j in range(N):
                    if j != ib and (cost[j] <= cost[ib] + 1e-6 or _same(pool[j], pool[ib])):
                        cand = kick(pool[ib], m, min(max(2, p // 10), m - p), perm_p, perm_m)
                        pool[j], cost[j], near[j], dist[j] = ls_full(cand, C, order, 100000000, counters)
                        kk[j] = 1
    ib = 0
    for i in range(1, N):
        if cost[i] < cost[ib]:
            ib = i
    stats[3] = now() - t0
    return cost[ib], pool[ib].copy(), stats
