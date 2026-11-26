"""MK-BPX: multi-kick search with balanced partition crossover for the p-median problem.

Kernels shared by all algorithms of the study (Numba, single thread, seeded).

Public functions
    build_order(C)                       sorted site lists, computed once per instance
    ls_swap / ls_full                    best-improvement swap local search
    kick(sol, m, k, ...)                 replace k random open facilities by k random closed sites
    px(S, ..., T, ...)                   balanced partition crossover (BPX)
    run_main(...)                        VNS (use_px=0) or MK-BPX (use_px=1), time-limited


Local search: best-improvement swap with the gain/loss/extra decomposition of
Resende & Werneck and distance-ordered customer lists, so that the evaluation of
an insertion stops as soon as no remaining customer can be affected.
"""
import time
import numpy as np
from numba import njit, objmode

INF = 1e300


@njit(cache=True)
def build_order(C):
    n = C.shape[0]
    order = np.empty((n, n), dtype=np.int32)
    for r in range(n):
        order[r] = np.argsort(C[r]).astype(np.int32)
    return order


@njit(cache=True)
def eval_sol(sol, C):
    n = C.shape[0]
    tot = 0.0
    for c in range(n):
        md = INF
        row = C[c]
        for k in range(sol.shape[0]):
            d = row[sol[k]]
            if d < md:
                md = d
        tot += md
    return tot


@njit(cache=True)
def random_subset(m, p, perm):
    for i in range(p):
        j = i + np.random.randint(0, m - i)
        t = perm[i]
        perm[i] = perm[j]
        perm[j] = t
    out = perm[:p].copy()
    out.sort()
    return out


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
        row = C[c]
        for k in range(p):
            d = row[sol[k]]
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
def ls_full(sol_in, C, order, max_iter, counters):
    """Best-improvement swap local search. C must be symmetric with I = J.
    counters[0] += neighbourhood scans, counters[2] += customer visits.
    Returns (sorted solution, cost)."""
    sol = sol_in.copy()
    p = sol.shape[0]
    n = C.shape[0]
    m = n
    inS = np.zeros(m, dtype=np.bool_)
    for k in range(p):
        inS[sol[k]] = True
    pos1 = np.empty(n, dtype=np.int64)
    pos2 = np.empty(n, dtype=np.int64)
    d1 = np.empty(n, dtype=np.float64)
    d2 = np.empty(n, dtype=np.float64)
    cost = _closest(sol, C, pos1, pos2, d1, d2)
    loss = np.empty(p, dtype=np.float64)
    gain = np.empty(m, dtype=np.float64)
    extra = np.zeros(m, dtype=np.float64)
    seen = np.zeros(m, dtype=np.bool_)
    touched = np.empty(m, dtype=np.int64)
    start = np.empty(p + 1, dtype=np.int64)
    fill = np.empty(p, dtype=np.int64)
    members = np.empty(n, dtype=np.int64)
    visits = 0
    for _ in range(max_iter):
        counters[0] += 1
        # customers grouped by their closest facility
        for k in range(p + 1):
            start[k] = 0
        for k in range(p):
            loss[k] = 0.0
        for c in range(n):
            start[pos1[c] + 1] += 1
            loss[pos1[c]] += d2[c] - d1[c]
        for k in range(p):
            start[k + 1] += start[k]
            fill[k] = start[k]
        for c in range(n):
            members[fill[pos1[c]]] = c
            fill[pos1[c]] += 1
        # pass 1: gain of every insertion
        for r in range(m):
            gain[r] = 0.0
        for i in range(n):
            row = order[i]
            Ci = C[i]
            di = d1[i]
            for t in range(m):
                r = row[t]
                dj = Ci[r]
                if dj >= di:
                    break
                gain[r] += di - dj
            visits += t
        u0 = 0
        for k in range(1, p):
            if loss[k] < loss[u0]:
                u0 = k
        best = 1e-6
        bi = -1
        bj = -1
        for r in range(m):
            if not inS[r]:
                pr = gain[r] - loss[u0]
                if pr > best:
                    best = pr
                    bi = u0
                    bj = r
        # pass 2: corrections for the pairs (u, r) that share customers
        for u in range(p):
            nt = 0
            lu = loss[u]
            for q in range(start[u], start[u + 1]):
                i = members[q]
                row = order[i]
                Ci = C[i]
                di = d1[i]
                d2i = d2[i]
                for t in range(m):
                    r = row[t]
                    dj = Ci[r]
                    if dj >= d2i:
                        break
                    if not seen[r]:
                        seen[r] = True
                        extra[r] = 0.0
                        touched[nt] = r
                        nt += 1
                    if dj < di:
                        extra[r] += d2i - di
                    else:
                        extra[r] += d2i - dj
                visits += t
            for t in range(nt):
                r = touched[t]
                seen[r] = False
                if not inS[r]:
                    pr = gain[r] - lu + extra[r]
                    if pr > best:
                        best = pr
                        bi = u
                        bj = r
        if bi == -1:
            break
        inS[sol[bi]] = False
        inS[bj] = True
        sol[bi] = bj
        cost -= best
        col = C[bj]
        for c in range(n):
            if pos1[c] == bi or pos2[c] == bi:
                a = INF
                b = INF
                ia = -1
                ib = -1
                row2 = C[c]
                for k in range(p):
                    d = row2[sol[k]]
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
    counters[2] += visits
    near = np.empty(n, dtype=np.int64)
    for c in range(n):
        near[c] = sol[pos1[c]]
    sol.sort()
    return sol, cost, near, d1


@njit(cache=True)
def ls_swap(sol_in, C, order, max_iter, counters):
    s, z, near, d1 = ls_full(sol_in, C, order, max_iter, counters)
    return s, z


@njit(cache=True)
def _find(par, x):
    while par[x] != x:
        par[x] = par[par[x]]
        x = par[x]
    return x


@njit(cache=True)
def px(S, zS, nearS, dS, T, zT, nearT, dT, m, par, tag, cid, info, neutral=0):
    """Balanced partition crossover of two solutions.

    The facilities exclusive to S (set D) and to T (set A) are grouped into
    components: d in D and a in A are linked when some customer is served by d in S
    and by a in T.  For a child that takes, in every component, either the facilities
    of S or those of T, cost(S) + sum of the component differences Delta_c over the
    components taken from T is an upper bound on its cost.  The child returned
    minimizes this bound subject to the cardinality constraint
    sum (|A_c| - |D_c|) = 0 (dynamic programme over the unbalanced components).
    info[0] = number of components, info[1] = components taken from T,
    info[2] = bound on the cost of the child, info[3] = |D|.
    tag: work array (int8, size m, zero on entry/exit); par, cid: int64 size m.
    """
    p = S.shape[0]
    n = nearS.shape[0]
    for k in range(p):
        tag[S[k]] = 1
    q = 0
    for k in range(p):
        if tag[T[k]] == 1:
            tag[T[k]] = 3          # common
        else:
            tag[T[k]] = 2          # exclusive to T
            q += 1
    info[0] = 0
    info[1] = 0
    info[2] = zS
    info[3] = q
    if q == 0:
        for k in range(p):
            tag[S[k]] = 0
            tag[T[k]] = 0
        return S.copy()
    for k in range(p):
        par[S[k]] = S[k]
        par[T[k]] = T[k]
    for i in range(n):
        s = nearS[i]
        t = nearT[i]
        if tag[s] == 1 and tag[t] == 2:
            rs = _find(par, s)
            rt = _find(par, t)
            if rs != rt:
                par[rs] = rt
    # components: index them
    nc = 0
    excl = np.empty(2 * q, dtype=np.int64)
    ne = 0
    for k in range(p):
        if tag[S[k]] == 1:
            excl[ne] = S[k]
            ne += 1
        if tag[T[k]] == 2:
            excl[ne] = T[k]
            ne += 1
    for e in range(ne):
        x = excl[e]
        if par[x] == x:
            cid[x] = nc
            nc += 1
    delta = np.zeros(nc, dtype=np.float64)
    nD = np.zeros(nc, dtype=np.int64)
    nA = np.zeros(nc, dtype=np.int64)
    for e in range(ne):
        x = excl[e]
        c = cid[_find(par, x)]
        if tag[x] == 1:
            nD[c] += 1
        else:
            nA[c] += 1
    for i in range(n):
        s = nearS[i]
        t = nearT[i]
        if s == t:
            continue
        if tag[s] == 1:
            delta[cid[_find(par, s)]] += dT[i] - dS[i]
        elif tag[t] == 2:
            delta[cid[_find(par, t)]] += dT[i] - dS[i]
    take = np.zeros(nc, dtype=np.bool_)
    bound = zS
    nub = 0
    for c in range(nc):
        if nD[c] == nA[c]:
            if delta[c] < -1e-9:
                take[c] = True
                bound += delta[c]
            elif neutral == 1 and delta[c] < 1e-9 and np.random.random() < 0.5:
                take[c] = True
        else:
            nub += 1
    if nub > 0:
        ub = np.empty(nub, dtype=np.int64)
        j = 0
        for c in range(nc):
            if nD[c] != nA[c]:
                ub[j] = c
                j += 1
        W = 0
        for j in range(nub):
            W += abs(nA[ub[j]] - nD[ub[j]])
        W = W // 2 + 1
        width = 2 * W + 1
        f = np.full(width, INF)
        f[W] = 0.0
        choice = np.zeros((nub, width), dtype=np.bool_)
        g = np.empty(width, dtype=np.float64)
        for j in range(nub):
            c = ub[j]
            w = nA[c] - nD[c]
            for b in range(width):
                g[b] = f[b]
            for b in range(width):
                bb = b - w
                if bb >= 0 and bb < width and f[bb] < INF:
                    v = f[bb] + delta[c]
                    if v < g[b] - 1e-9:
                        g[b] = v
                        choice[j, b] = True
            for b in range(width):
                f[b] = g[b]
        if f[W] < -1e-9:
            b = W
            for j in range(nub - 1, -1, -1):
                if choice[j, b]:
                    c = ub[j]
                    take[c] = True
                    b -= nA[c] - nD[c]
            bound += f[W]
    ntake = 0
    for c in range(nc):
        if take[c]:
            ntake += 1
    child = np.empty(p, dtype=np.int64)
    j = 0
    for k in range(p):
        x = S[k]
        if tag[x] == 3 or not take[cid[_find(par, x)]]:
            child[j] = x
            j += 1
    for k in range(p):
        x = T[k]
        if tag[x] == 2 and take[cid[_find(par, x)]]:
            child[j] = x
            j += 1
    for k in range(p):
        tag[S[k]] = 0
        tag[T[k]] = 0
    info[0] = nc
    info[1] = ntake
    info[2] = bound
    child.sort()
    return child


@njit(cache=True)
def kick(sol, m, k, perm_p, perm_m):
    """Replace k random open facilities by k random closed sites."""
    p = sol.shape[0]
    out = sol.copy()
    for i in range(p):
        perm_p[i] = i
    for i in range(k):
        j = i + np.random.randint(0, p - i)
        t = perm_p[i]
        perm_p[i] = perm_p[j]
        perm_p[j] = t
    inS = np.zeros(m, dtype=np.bool_)
    for i in range(p):
        inS[sol[i]] = True
    q = 0
    for s in range(m):
        if not inS[s]:
            perm_m[q] = s
            q += 1
    for i in range(k):
        j = i + np.random.randint(0, q - i)
        t = perm_m[i]
        perm_m[i] = perm_m[j]
        perm_m[j] = t
        out[perm_p[i]] = perm_m[i]
    out.sort()
    return out


@njit(cache=True)
def now():
    with objmode(t='float64'):
        t = time.perf_counter()
    return t


@njit(cache=True)
def _same(a, b):
    for k in range(a.shape[0]):
        if a[k] != b[k]:
            return False
    return True


@njit(cache=True)
def run_main_ext(C, order, p, use_px, neutral, kmax, N, seed, target, time_limit, max_iters, min_iters, t_mark):
    """Proposed algorithm and its controls (extended interface).

    Same search as run_main, with additions that do not change the sequence of solutions:
    max_iters > 0  stops the run after that number of iterations (kicks), whatever the time; with
                   time_limit = INF this replays a logged run exactly on any machine.
    min_iters > 0  the run goes on after the time limit until that number of iterations has been
                   done, so that a time-limited run also covers a run logged on another machine.
    t_mark > 0     only records, in stats[14], the number of iterations completed at time t_mark.
    The trace has a third column with the iteration at which each improvement was found
    (0 = initial local search), and stats has three more entries: stats[12], the number of
    iterations completed when the time limit was reached; stats[13], the time at which
    iteration min_iters was completed (-1 if min_iters = 0); stats[14], see t_mark.

    use_px = 0, N = 1  -> basic VNS (same code path, crossover disabled)
    use_px = 1, N = 1  -> multi-kick search with balanced partition crossover
    N > 1              -> N elite solutions evolved in turn and fused by PX after
                          every round
    stats = [iterations, improvements by plain acceptance, improvements by PX,
             PX calls, PX calls with >= 1 component taken, components (sum),
             components taken (sum), scans, time, time of first hit, bound violations,
             sum of |D| over PX calls]
    """
    np.random.seed(seed)
    m = C.shape[0]
    n = m
    t0 = now()
    perm = np.arange(m).astype(np.int64)
    perm_p = np.empty(p, dtype=np.int64)
    perm_m = np.empty(m, dtype=np.int64)
    par = np.empty(m, dtype=np.int64)
    cid = np.empty(m, dtype=np.int64)
    tag = np.zeros(m, dtype=np.int8)
    info = np.zeros(4, dtype=np.float64)
    counters = np.zeros(4, dtype=np.int64)
    stats = np.zeros(15, dtype=np.float64)
    stats[12] = -1.0
    stats[13] = -1.0
    stats[14] = -1.0
    trace = np.empty((6000, 3), dtype=np.float64)
    ntr = 0
    pool = np.empty((N, p), dtype=np.int64)
    cost = np.empty(N, dtype=np.float64)
    near = np.empty((N, n), dtype=np.int64)
    dist = np.empty((N, n), dtype=np.float64)
    kk = np.ones(N, dtype=np.int64)
    best = INF
    t_hit = -1.0
    for i in range(N):
        s = random_subset(m, p, perm)
        pool[i], cost[i], near[i], dist[i] = ls_full(s, C, order, 100000000, counters)
        if cost[i] < best - 1e-6:
            best = cost[i]
            tn = now() - t0
            trace[ntr, 0] = tn; trace[ntr, 1] = best; trace[ntr, 2] = 0.0; ntr += 1
            if t_hit < 0 and best <= target + 1e-6:
                t_hit = tn
    stop = False
    while not stop:
        for i in range(N):
            el = now() - t0
            if el >= time_limit and stats[12] < 0:
                stats[12] = stats[0]
            if min_iters > 0 and stats[13] < 0 and stats[0] >= min_iters:
                stats[13] = el
            if t_mark > 0 and stats[14] < 0 and el >= t_mark:
                stats[14] = stats[0]
            if (el >= time_limit and stats[0] >= min_iters) or (max_iters > 0 and stats[0] >= max_iters):
                stop = True
                break
            stats[0] += 1
            k = min(kk[i], m - p, p)
            cand = kick(pool[i], m, k, perm_p, perm_m)
            s2, z2, n2, d2 = ls_full(cand, C, order, 100000000, counters)
            improved = False
            if z2 < cost[i] - 1e-6:
                stats[1] += 1
                if use_px == 1:
                    ch = px(s2, z2, n2, d2, pool[i], cost[i], near[i], dist[i], m, par, tag, cid, info, 0)
                    stats[3] += 1; stats[5] += info[0]; stats[6] += info[1]; stats[11] += info[3]
                    if info[1] > 0:
                        stats[4] += 1
                        s3, z3, n3, d3 = ls_full(ch, C, order, 100000000, counters)
                        if z3 > info[2] + 1e-6:
                            stats[10] += 1
                        if z3 < z2 - 1e-6:
                            s2 = s3; z2 = z3; n2 = n3; d2 = d3
                pool[i] = s2; cost[i] = z2; near[i] = n2; dist[i] = d2
                improved = True
            elif use_px == 1:
                ch = px(pool[i], cost[i], near[i], dist[i], s2, z2, n2, d2, m, par, tag, cid, info, neutral)
                stats[3] += 1; stats[5] += info[0]; stats[6] += info[1]; stats[11] += info[3]
                if info[1] > 0:
                    stats[4] += 1
                    s3, z3, n3, d3 = ls_full(ch, C, order, 100000000, counters)
                    if z3 > info[2] + 1e-6:
                        stats[10] += 1
                    if z3 < cost[i] - 1e-6:
                        pool[i] = s3; cost[i] = z3; near[i] = n3; dist[i] = d3
                        improved = True
                        stats[2] += 1
                    elif neutral == 1 and z3 < cost[i] + 1e-6:
                        pool[i] = s3; cost[i] = z3; near[i] = n3; dist[i] = d3
            if improved:
                kk[i] = 1
            else:
                kk[i] += 1
                if kk[i] > kmax:
                    kk[i] = 1
            if cost[i] < best - 1e-6:
                best = cost[i]
                tn = now() - t0
                if ntr < trace.shape[0]:
                    trace[ntr, 0] = tn; trace[ntr, 1] = best; trace[ntr, 2] = stats[0]; ntr += 1
                if t_hit < 0 and best <= target + 1e-6:
                    t_hit = tn
        if N > 1 and not stop:
            ib = 0
            for i in range(1, N):
                if cost[i] < cost[ib]:
                    ib = i
            for j in range(N):
                if j == ib:
                    continue
                ch = px(pool[ib], cost[ib], near[ib], dist[ib], pool[j], cost[j], near[j], dist[j], m, par, tag, cid, info, 0)
                if info[1] > 0:
                    s3, z3, n3, d3 = ls_full(ch, C, order, 100000000, counters)
                    if z3 < cost[ib] - 1e-6:
                        pool[ib] = s3; cost[ib] = z3; near[ib] = n3; dist[ib] = d3
            for j in range(N):
                if j != ib:
                    pool[j] = pool[ib]; cost[j] = cost[ib]; near[j] = near[ib]; dist[j] = dist[ib]
            if cost[ib] < best - 1e-6:
                best = cost[ib]
                tn = now() - t0
                if ntr < trace.shape[0]:
                    trace[ntr, 0] = tn; trace[ntr, 1] = best; trace[ntr, 2] = stats[0]; ntr += 1
                if t_hit < 0 and best <= target + 1e-6:
                    t_hit = tn
    ib = 0
    for i in range(1, N):
        if cost[i] < cost[ib]:
            ib = i
    stats[7] = counters[0]
    stats[8] = now() - t0
    stats[9] = t_hit
    return cost[ib], pool[ib].copy(), stats, trace[:ntr]


@njit(cache=True)
def run_main(C, order, p, use_px, neutral, kmax, N, seed, target, time_limit):
    """VNS (use_px = 0) or MK-BPX (use_px = 1) with a time limit: the interface used throughout the
    study.  Returns (cost, solution, stats, trace) with trace = [[time, value], ...]; see
    run_main_ext for the meaning of stats.
    """
    z, sol, stats, trace = run_main_ext(C, order, p, use_px, neutral, kmax, N, seed, target, time_limit, 0, 0, 0.0)
    return z, sol, stats[:12].copy(), trace[:, :2].copy()
