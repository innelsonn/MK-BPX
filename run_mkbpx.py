"""Run MK-BPX (or the VNS control) on one instance.

Examples
  python run_mkbpx.py pmed40                       OR-Library instance (p comes with the instance)
  python run_mkbpx.py fl1400 --p 100 --time 10     TSPLIB instance of data/tsplib
  python run_mkbpx.py u1432 --p 200 --algo VNS --seed 7 --runs 5
  python run_mkbpx.py my.tsp --p 50                any TSPLIB file with NODE_COORD_SECTION
  python run_mkbpx.py D.npy --p 30                 any symmetric distance matrix (customers = sites)

Options
  --p P        number of medians (required except for pmedX)
  --time T     time limit in seconds per run (default 10)
  --algo A     BPX (default), VNS, VNS20 or BPX4
  --seed S     seed of the first run (default 1); run r uses S + r
  --runs R     number of runs (default 1)
  --save FILE  write the best solution found (one site index per line, 0-based)
"""
import os, sys, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mkbpx import core as K, instances as inst

ALGOS = {'VNS': (0, 0, 1), 'BPX': (1, 0, 1), 'VNS20': (0, 20, 1), 'BPX4': (1, 0, 4)}   # use_px, k_max (0 = p), copies


def load(name):
    """Returns (distance matrix, known name or None)."""
    if name.endswith('.npy'):
        C = np.ascontiguousarray(np.load(name).astype(np.float64))
        if C.ndim != 2 or C.shape[0] != C.shape[1] or not np.array_equal(C, C.T):
            sys.exit('the matrix must be square and symmetric (customers = candidate sites)')
        return C, None
    if name.endswith('.tsp'):
        xy = []; go = False
        for ln in open(name):
            ln = ln.strip()
            if ln.startswith('NODE_COORD_SECTION'): go = True; continue
            if go and ln and ln != 'EOF':
                a = ln.split(); xy.append((float(a[1]), float(a[2])))
        xy = np.array(xy); d = xy[:, None, :] - xy[None, :, :]
        return np.ascontiguousarray(np.floor(np.sqrt((d ** 2).sum(-1)))), None
    return inst.load_matrix(name), name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('instance'); ap.add_argument('--p', type=int); ap.add_argument('--time', type=float, default=10.0)
    ap.add_argument('--algo', default='BPX', choices=sorted(ALGOS)); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--runs', type=int, default=1); ap.add_argument('--save')
    a = ap.parse_args()
    C, known = load(a.instance); n = C.shape[0]
    p = a.p if a.p else (inst.pmed_p(known) if known and known.startswith('pmed') else None)
    if not p or not 0 < p < n: sys.exit('give a valid number of medians with --p')
    ref = None
    if known:
        try: ref = inst.opt(known, p)
        except KeyError: pass
    use_px, kmax0, N = ALGOS[a.algo]; kmax = p if kmax0 == 0 else min(p, kmax0)
    print(f'{a.instance}: n = {n}, p = {p}, algorithm {a.algo}, {a.time:g} s per run' + (f', reference value {ref}' if ref else ''))
    order = K.build_order(C)
    K.run_main(C, order, min(5, p), 1, 0, 5, 2, 1, 0., 0.05); K.run_main(C, order, min(5, p), 0, 0, 5, 1, 1, 0., 0.05)   # compile
    best = None
    for r in range(a.runs):
        z, sol, st, tr = K.run_main(C, order, p, use_px, 0, kmax, N, a.seed + r, float(ref or 0), a.time)
        assert abs(K.eval_sol(sol, C) - z) < 1e-6 and len(set(sol.tolist())) == p
        gap = f', gap {100 * (z - ref) / ref:.4f} %' if ref else ''
        print(f'  run {r} (seed {a.seed + r}): cost {z:.0f}{gap}, {int(st[0])} iterations, {int(st[1])} improvements by acceptance, {int(st[2])} by crossover')
        if best is None or z < best[0]: best = (z, sol)
    print(f'best cost: {best[0]:.0f}')
    if a.save:
        np.savetxt(a.save, best[1], fmt='%d'); print('solution written to', a.save)


if __name__ == '__main__':
    main()
