"""PopStar-style hybrid heuristic (GRASP with path-relinking) of Resende & Werneck (2004),
re-implemented on the kernels of this study so that it can be run under the same conditions
as MK-BPX: same language and compiler, same swap local search (core.ls_full), same distance
matrices, same random generator, same clock and same time limit.

Components, as described in "A hybrid heuristic for the p-median problem" (J. Heuristics 10):
    sample_greedy     constructive: at each step q = ceil(log2(m/p)) random candidates are
                      evaluated and the most profitable one is inserted
    core.ls_full      swap local search (shared with VNS and MK-BPX)
    path_relink       path from S1 to S2 by the most profitable (least costly) swap at each
                      step; returns the best local minimum of the path
    pool              elite pool with the insertion/replacement rules of the paper
                      (minimum difference of four facilities, replace the most similar
                      solution of equal or higher cost)
    multistart phase  `maxit` iterations: construct, local search, relink with an elite
                      solution chosen with probability proportional to its difference
                      (direction: from the better to the worse), local search, pool update
    post-optimization all pairs of the pool are relinked (from the worse to the better), the
                      results form a new generation; repeated while a generation improves

The only change with respect to the paper is the stopping rule: instead of stopping after one
cycle (multistart phase + post-optimization), cycles are repeated with the pool retained until
the time limit (max_cycles = 0), so that the heuristic uses the whole time budget.  With
max_cycles = 1 and no time limit it is the original algorithm (default: maxit = 32, 10 elite).
"""
import math
import numpy as np
from numba import njit
from . import core as K

INF = 1e300


@njit(cache=True)
def sample_greedy(C, p, perm):
    n = C.shape[0]
    m = n
    q = int(math.ceil(math.log2(m / p)))
    if q < 1:
        q = 1
    d1 = np.full(n, 1e15)
    for i in range(m):
        perm[i] = i
    sol = np.empty(p, dtype=np.int64)
    avail = m
    for s in range(p):
        qq = min(q, avail)
        bg = -1.0
        bpos = -1
        for c in range(qq):
            j = c + np.random.randint(0, avail - c)
            t = perm[c]; perm[c] = perm[j]; perm[j] = t
            r = perm[c]
            col = C[r]
            g = 0.0
            for i in range(n):
                dj = col[i]
                if dj < d1[i]:
                    g += d1[i] - dj
            if g > bg:
                bg = g
                bpos = c
        r = perm[bpos]
        sol[s] = r
        perm[bpos] = perm[avail - 1]
        perm[avail - 1] = r
        avail -= 1
        col = C[r]
        for i in range(n):
            if col[i] < d1[i]:
                d1[i] = col[i]
    sol.sort()
    return sol


@njit(cache=True)
def diff_sorted(a, b):
    """Number of facilities of a that are not in b (a, b sorted, same size)."""
    p = a.shape[0]
    i = 0; j = 0; common = 0
    while i < p and j < p:
        if a[i] == b[j]:
            common += 1; i += 1; j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return p - common


@njit(cache=True)
def path_relink(S1, S2, C, order, pc):
    """Path from S1 to S2.  Returns (found, solution): the best local minimum of the path
    (a solution with strictly higher costs on both sides, plateaus allowed); found = 0 if the
    path has none.  pc[0] += steps, pc[1] += customer-list visits."""
    p = S1.shape[0]
    n = C.shape[0]
    m = n
    sol = S1.copy()
    in1 = np.zeros(m, dtype=np.bool_)
    in2 = np.zeros(m, dtype=np.bool_)
    for k in range(p):
        in1[S1[k]] = True
        in2[S2[k]] = True
    Dpos = np.empty(p, dtype=np.int64)
    Asite = np.empty(p, dtype=np.int64)
    q = 0
    for k in range(p):
        if not in2[sol[k]]:
            Dpos[q] = k
            q += 1
    qa = 0
    for k in range(p):
        if not in1[S2[k]]:
            Asite[qa] = S2[k]
            qa += 1
    if q < 2:
        return 0, sol
    pos1 = np.empty(n, dtype=np.int64)
    pos2 = np.empty(n, dtype=np.int64)
    d1 = np.empty(n, dtype=np.float64)
    d2 = np.empty(n, dtype=np.float64)
    cost = K._closest(sol, C, pos1, pos2, d1, d2)
    idxD = np.full(p, -1, dtype=np.int64)
    idxA = np.full(m, -1, dtype=np.int64)
    pathc = np.empty(q + 1, dtype=np.float64)
    pathc[0] = cost
    sw_pos = np.empty(q, dtype=np.int64)
    sw_site = np.empty(q, dtype=np.int64)
    gain = np.empty(q, dtype=np.float64)
    loss = np.empty(q, dtype=np.float64)
    extra = np.empty((q, q), dtype=np.float64)
    rem = q
    lastvisits = 0
    for step in range(q):
        for a in range(rem):
            idxA[Asite[a]] = a
            gain[a] = 0.0
            idxD[Dpos[a]] = a
            loss[a] = 0.0
            for b in range(rem):
                extra[a, b] = 0.0
        if rem * n <= lastvisits:
            for i in range(n):
                du = idxD[pos1[i]]
                if du >= 0:
                    loss[du] += d2[i] - d1[i]
            for a in range(rem):
                col = C[Asite[a]]
                g = 0.0
                for i in range(n):
                    dj = col[i]
                    if dj < d2[i]:
                        di = d1[i]
                        if dj < di:
                            g += di - dj
                        du = idxD[pos1[i]]
                        if du >= 0:
                            if dj > di:
                                extra[du, a] += d2[i] - dj
                            else:
                                extra[du, a] += d2[i] - di
                gain[a] = g
            pc[1] += rem * n
        else:
            visits = 0
            for i in range(n):
                du = idxD[pos1[i]]
                di = d1[i]
                d2i = d2[i]
                if du >= 0:
                    loss[du] += d2i - di
                row = order[i]
                Ci = C[i]
                for t in range(m):
                    r = row[t]
                    dj = Ci[r]
                    if dj >= d2i:
                        break
                    a = idxA[r]
                    if a >= 0:
                        if dj < di:
                            gain[a] += di - dj
                        if du >= 0:
                            if dj > di:
                                extra[du, a] += d2i - dj
                            else:
                                extra[du, a] += d2i - di
                visits += t
            lastvisits = visits
            pc[1] += visits
        best = -INF
        bd = -1
        ba = -1
        for d in range(rem):
            ld = loss[d]
            for a in range(rem):
                pr = gain[a] - ld + extra[d, a]
                if pr > best:
                    best = pr
                    bd = d
                    ba = a
        bi = Dpos[bd]
        bj = Asite[ba]
        idxD[bi] = -1
        idxA[bj] = -1
        Dpos[bd] = Dpos[rem - 1]
        Asite[ba] = Asite[rem - 1]
        rem -= 1
        sol[bi] = bj
        cost -= best
        col = C[bj]
        for c in range(n):
            if pos1[c] == bi or pos2[c] == bi:
                x = INF; y = INF; ia = -1; ib = -1
                row2 = C[c]
                for k in range(p):
                    d = row2[sol[k]]
                    if d < x:
                        y = x; ib = ia; x = d; ia = k
                    elif d < y:
                        y = d; ib = k
                pos1[c] = ia; pos2[c] = ib; d1[c] = x; d2[c] = y
            else:
                dj = col[c]
                if dj < d1[c]:
                    d2[c] = d1[c]; pos2[c] = pos1[c]; d1[c] = dj; pos1[c] = bi
                elif dj < d2[c]:
                    d2[c] = dj; pos2[c] = bi
        pathc[step + 1] = cost
        sw_pos[step] = bi
        sw_site[step] = bj
    pc[0] += q
    eps = 1e-6
    bt = -1
    bc = INF
    for t in range(1, q):
        a = t - 1
        while a > 0 and abs(pathc[a] - pathc[t]) <= eps:
            a -= 1
        b = t + 1
        while b < q and abs(pathc[b] - pathc[t]) <= eps:
            b += 1
        if pathc[a] > pathc[t] + eps and pathc[b] > pathc[t] + eps and pathc[t] < bc - eps:
            bt = t
            bc = pathc[t]
    if bt < 0:
        return 0, sol
    out = S1.copy()
    for t in range(bt):
        out[sw_pos[t]] = sw_site[t]
    out.sort()
    return 1, out


@njit(cache=True)
def pool_add(pool, cost, cnt, S, z):
    """Insertion rules of the paper.  cnt: array of one element (current size).
    Returns True if S entered the pool."""
    E = pool.shape[0]
    nc = cnt[0]
    eps = 1e-6
    worst = -INF
    msim = -1          # most similar among those with cost >= z
    msd = 1 << 60
    close = False      # within distance four of some elite solution
    for e in range(nc):
        d = diff_sorted(S, pool[e])
        if cost[e] < z - eps and d < 4:
            return False
        if d < 4:
            close = True
        if cost[e] > worst:
            worst = cost[e]
        if cost[e] >= z - eps and d < msd:
            msd = d
            msim = e
    if nc < E and not close:
        pool[nc] = S
        cost[nc] = z
        cnt[0] = nc + 1
        return True
    if nc == E and z > worst + eps:
        return False
    if msim < 0:
        return False
    pool[msim] = S
    cost[msim] = z
    return True


@njit(cache=True)
def run_popstar(C, order, p, maxit, esize, seed, target, time_limit, max_cycles):
    """Time-limited PopStar.  max_cycles = 0: repeat cycles until the time limit.
    stats = [multistart iterations, path-relinkings, relinkings with a local minimum,
             post-optimization generations, cycles, improvements of the best by the multistart
             local search, by relinking in the multistart phase, by post-optimization,
             neighbourhood scans, time, time of first hit, relinking steps]
    The trace holds (time, value) of every improvement of the best solution; improvements
    found after the time limit are in the trace with their time, so that the value at the
    limit can be recovered exactly."""
    np.random.seed(seed)
    m = C.shape[0]
    t0 = K.now()
    perm = np.arange(m).astype(np.int64)
    counters = np.zeros(4, dtype=np.int64)
    pc = np.zeros(2, dtype=np.int64)
    stats = np.zeros(12, dtype=np.float64)
    trace = np.empty((6000, 2), dtype=np.float64)
    ntr = 0
    pool = np.empty((esize, p), dtype=np.int64)
    cost = np.empty(esize, dtype=np.float64)
    cnt = np.zeros(1, dtype=np.int64)
    gpool = np.empty((esize, p), dtype=np.int64)
    gcost = np.empty(esize, dtype=np.float64)
    gcnt = np.zeros(1, dtype=np.int64)
    npool = np.empty((esize, p), dtype=np.int64)
    ncost = np.empty(esize, dtype=np.float64)
    ncnt = np.zeros(1, dtype=np.int64)
    w = np.empty(esize, dtype=np.float64)
    best = INF
    bsol = np.empty(p, dtype=np.int64)
    t_hit = -1.0
    eps = 1e-6
    stop = False
    while not stop:
        # ---------------- multistart phase with path-relinking ----------------
        for it in range(maxit):
            if K.now() - t0 >= time_limit:
                stop = True
                break
            stats[0] += 1
            s0 = sample_greedy(C, p, perm)
            S, z, nr, dd = K.ls_full(s0, C, order, 100000000, counters)
            if z < best - eps:
                best = z; bsol[:] = S; stats[5] += 1
                tn = K.now() - t0
                if ntr < trace.shape[0]:
                    trace[ntr, 0] = tn; trace[ntr, 1] = best; ntr += 1
                if t_hit < 0 and best <= target + eps:
                    t_hit = tn
            # biased selection of an elite solution
            tot = 0.0
            for e in range(cnt[0]):
                w[e] = diff_sorted(S, pool[e])
                tot += w[e]
            if tot > 0 and K.now() - t0 < time_limit:
                u = np.random.random() * tot
                sel = 0
                acc = w[0]
                while acc < u and sel < cnt[0] - 1:
                    sel += 1
                    acc += w[sel]
                stats[1] += 1
                if z <= cost[sel]:        # "up": from the better to the worse
                    found, R = path_relink(S, pool[sel], C, order, pc)
                else:
                    found, R = path_relink(pool[sel], S, C, order, pc)
                if found == 1:
                    stats[2] += 1
                    R2, z2, nr2, dd2 = K.ls_full(R, C, order, 100000000, counters)
                    if z2 < best - eps:
                        best = z2; bsol[:] = R2; stats[6] += 1
                        tn = K.now() - t0
                        if ntr < trace.shape[0]:
                            trace[ntr, 0] = tn; trace[ntr, 1] = best; ntr += 1
                        if t_hit < 0 and best <= target + eps:
                            t_hit = tn
                    pool_add(pool, cost, cnt, R2, z2)
            pool_add(pool, cost, cnt, S, z)
        if stop:
            break
        # ---------------- post-optimization ----------------
        ng = cnt[0]
        if ng >= 2:
            gcnt[0] = ng
            gbest = INF
            for e in range(ng):
                gpool[e] = pool[e]; gcost[e] = cost[e]
                if cost[e] < gbest:
                    gbest = cost[e]
            while True:
                ncnt[0] = 0
                stats[3] += 1
                for i in range(gcnt[0]):
                    for j in range(i + 1, gcnt[0]):
                        if K.now() - t0 >= time_limit:
                            stop = True
                            break
                        if gcost[i] >= gcost[j]:      # "down": from the worse to the better
                            a = i; b = j
                        else:
                            a = j; b = i
                        stats[1] += 1
                        found, R = path_relink(gpool[a], gpool[b], C, order, pc)
                        if found == 1:
                            stats[2] += 1
                            R2, z2, nr2, dd2 = K.ls_full(R, C, order, 100000000, counters)
                            if z2 < best - eps:
                                best = z2; bsol[:] = R2; stats[7] += 1
                                tn = K.now() - t0
                                if ntr < trace.shape[0]:
                                    trace[ntr, 0] = tn; trace[ntr, 1] = best; ntr += 1
                                if t_hit < 0 and best <= target + eps:
                                    t_hit = tn
                            pool_add(npool, ncost, ncnt, R2, z2)
                        else:                          # no local minimum: one of the endpoints
                            if np.random.random() < 0.5:
                                pool_add(npool, ncost, ncnt, gpool[a], gcost[a])
                            else:
                                pool_add(npool, ncost, ncnt, gpool[b], gcost[b])
                    if stop:
                        break
                if stop:
                    break
                nb = INF
                for e in range(ncnt[0]):
                    if ncost[e] < nb:
                        nb = ncost[e]
                if nb < gbest - eps and ncnt[0] >= 2:
                    gbest = nb
                    gcnt[0] = ncnt[0]
                    for e in range(ncnt[0]):
                        gpool[e] = npool[e]; gcost[e] = ncost[e]
                else:
                    break
            pool_add(pool, cost, cnt, bsol, best)
        stats[4] += 1
        if max_cycles > 0 and stats[4] >= max_cycles:
            break
    stats[8] = counters[0]
    stats[9] = K.now() - t0
    stats[10] = t_hit
    stats[11] = pc[0]
    return best, bsol.copy(), stats, trace[:ntr]
