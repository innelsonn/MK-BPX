"""Reference implementation of the fast interchange of Whitaker (1983): best-improvement swap
local search with one pass over the customers per candidate insertion, O(mn) per scan.
Used only by experiments/speed_benchmark.py. Call ls_swap(sol, C, C.T-contiguous, max_iter, counters)."""
import numpy as np
from numba import njit

INF = 1e300


@njit(cache=True)
def _closest(sol, C, pos1, pos2, d1, d2):
    n = C.shape[0]
    p = sol.shape[0]
    tot = 0.0
    for c in range(n):
        a = INF
        b = INF
        ia = -1
        ib = -1
        for k in range(p):
            d = C[c, sol[k]]
            if d < a:
                b = a
                ib = ia
                a = d
                ia = k
            elif d < b:
                b = d
                ib = k
        pos1[c] = ia
        pos2[c] = ib
        d1[c] = a
        d2[c] = b
        tot += a
    return tot


@njit(cache=True)
def ls_swap(sol_in, C, CT, max_iter, counters):
    """Best-improvement swap local search, fast interchange (Whitaker, 1983).

    For every candidate insertion r one pass over the customers yields the
    gain of opening r and, for every open facility, the loss of closing it,
    so a complete scan of the neighbourhood costs O(mn).
    Ties are broken as in the original implementation: lowest position of
    the removed facility in the solution vector, then lowest inserted site.
    counters[0] += number of complete neighbourhood scans.
    Returns (sorted solution, cost).
    """
    sol = sol_in.copy()
    p = sol.shape[0]
    n = C.shape[0]
    m = C.shape[1]
    inS = np.zeros(m, dtype=np.bool_)
    for k in range(p):
        inS[sol[k]] = True
    pos1 = np.empty(n, dtype=np.int64)
    pos2 = np.empty(n, dtype=np.int64)
    d1 = np.empty(n, dtype=np.float64)
    d2 = np.empty(n, dtype=np.float64)
    cost = _closest(sol, C, pos1, pos2, d1, d2)
    v = np.empty(p, dtype=np.float64)

    for _ in range(max_iter):
        counters[0] += 1
        best = 0.0
        bi = -1
        bj = -1
        for j in range(m):
            if inS[j]:
                continue
            col = CT[j]
            gain = 0.0
            for k in range(p):
                v[k] = 0.0
            for c in range(n):
                dj = col[c]
                if dj < d1[c]:
                    gain += d1[c] - dj
                else:
                    x = d2[c]
                    if dj < x:
                        x = dj
                    v[pos1[c]] += x - d1[c]
            for i in range(p):
                pr = gain - v[i]
                if pr > best + 1e-6:
                    best = pr
                    bi = i
                    bj = j
                elif bi != -1 and pr > best - 1e-6 and i < bi:
                    bi = i
                    bj = j
        if bi == -1:
            break
        inS[sol[bi]] = False
        inS[bj] = True
        sol[bi] = bj
        cost -= best
        col = CT[bj]
        for c in range(n):
            if pos1[c] == bi or pos2[c] == bi:
                a = INF
                b = INF
                ia = -1
                ib = -1
                for k in range(p):
                    d = C[c, sol[k]]
                    if d < a:
                        b = a
                        ib = ia
                        a = d
                        ia = k
                    elif d < b:
                        b = d
                        ib = k
                pos1[c] = ia
                pos2[c] = ib
                d1[c] = a
                d2[c] = b
            else:
                dj = col[c]
                if dj < d1[c]:
                    d2[c] = d1[c]
                    pos2[c] = pos1[c]
                    d1[c] = dj
                    pos1[c] = bi
                elif dj < d2[c]:
                    d2[c] = dj
                    pos2[c] = bi
    sol.sort()
    return sol, cost
