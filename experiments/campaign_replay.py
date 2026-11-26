"""Exact replay of the main campaign (used to complete its improvement traces).

Usage:  python experiments/campaign_replay.py ALGO SET [--out DIR] [--main DIR] [--core K]

ALGO   VNS or BPX (as in campaign.py);  SET  S0, S1 or S2 (same pairs, seeds and time limits T).
--out  output directory (default results_replay/)
--main logs of the main campaign (default results/)
--core CPU core to which the process is pinned

A run of VNS or MK-BPX is determined by its seed; only the point at which the time limit stops it
depends on the machine.  The script repeats every run logged by the main campaign with its seed
and lets it go on until it has done at least as many iterations as the logged run (and at least
until 1.25 T).  The value after the logged number of iterations must be the logged value (column
`replay_ok`), and the trace stores the iteration of every improvement besides its time and value,
so experiments/complete_traces.py can rebuild the improvement trace of any logged run.

Columns: best_tl, best_tl3, best_end (values at T, at 1.25 T and at the end); iters_tl, iters_tl3,
iters_end; main_iters, main_best (the logged run); replay_best, replay_ok; t_replay (time at
which the logged number of iterations was reached on this machine); counters as in campaign.py;
cpu_ratio and attempts (a run whose CPU time is below 98.5 % of its elapsed time is repeated, at
most twice).  Complete pairs are skipped when the script is restarted.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'experiments'))
import json, csv, time
import numpy as np, pandas as pd
from mkbpx import core as K, instances as inst
import campaign as CP

HEADER = ('algo inst n p opt tl tl3 run seed best_tl best_tl3 best_end iters_tl iters_tl3 iters_end time_end '
          'main_iters main_best replay_best replay_ok t_replay imp_plain imp_px px_calls px_take comps comps_taken scans viol dsum '
          'cpu_ratio attempts').split()


def value_at(tr, t):
    """Best value found up to time t (the value of the initial local optimum if it came later)."""
    v = [b for a, b, _ in tr if a <= t]
    return min(v) if v else tr[0][1]


def horizon(setname, name, p, tl, pop):
    return 1.25 * tl


def main():
    argv = sys.argv[1:]; OUT = os.path.join(ROOT, 'results_replay'); MAIN = os.path.join(ROOT, 'results'); core = None
    for flag in ('--out', '--main', '--core'):
        if flag in argv:
            k = argv.index(flag); v = argv[k + 1]; del argv[k:k + 2]
            if flag == '--out': OUT = v
            elif flag == '--main': MAIN = v
            else: core = int(v)
    if len(argv) != 2 or argv[0] not in ('VNS', 'BPX') or argv[1] not in ('S0', 'S1', 'S2'):
        print(__doc__); sys.exit(1)
    algo, setname = argv
    if core is not None: os.sched_setaffinity(0, {core})
    os.makedirs(OUT, exist_ok=True)
    aid, use_px, kmax0, N = CP.ALGOS[algo]
    pop = None
    logged = {}
    mp = os.path.join(MAIN, f'{setname}_{algo}.csv')
    if os.path.exists(mp):
        for r in pd.read_csv(mp).itertuples(): logged[(r.inst, r.p, r.run)] = (int(r.iters), int(r.best))
    path = os.path.join(OUT, f'{setname}_{algo}.csv'); tpath = path[:-4] + '_trace.json'; opath = os.path.join(OUT, 'order_time.json')
    plan = [(nm, p, tl, R) for nm, p, tl, R in CP.pairs(setname)]
    done = set(); keep = []
    if os.path.exists(path):
        rows = list(csv.reader(open(path)))[1:]
        need = {(q[0], q[1]): q[3] for q in plan}; cnt = {}
        for r in rows: cnt[(r[1], int(r[3]))] = cnt.get((r[1], int(r[3])), 0) + 1
        done = {k for k, v in cnt.items() if v == need.get(k)}
        keep = [r for r in rows if (r[1], int(r[3])) in done]
    f = open(path, 'w', newline=''); w = csv.writer(f); w.writerow(HEADER); w.writerows(keep); f.flush()
    traces = json.load(open(tpath)) if os.path.exists(tpath) else {}
    cur = None
    for name, p, tl, R in plan:
        if (name, p) in done: continue
        C = inst.load_matrix(name)
        if cur != name:
            order = K.build_order(C); cur = name
            K.run_main_ext(C, order, min(5, p), 1, 0, 5, 1, 1, 0., 0.05, 0, 0, 0.0); K.run_main_ext(C, order, min(5, p), 0, 0, 5, 1, 1, 0., 0.05, 0, 0, 0.0)
            ts = []
            for _ in range(3):
                a = time.perf_counter(); K.build_order(C); ts.append(time.perf_counter() - a)
            ot = json.load(open(opath)) if os.path.exists(opath) else {}
            ot.setdefault(name, {})[algo] = round(float(np.median(ts)), 4); json.dump(ot, open(opath, 'w'), indent=1)
        o = inst.opt(name, p); kmax = p if kmax0 == 0 else min(p, kmax0)
        tl3 = horizon(setname, name, p, tl, pop)
        for r in range(R):
            seed = aid * 10**7 + CP.PAIRID[(name, p)] * 100 + r
            mi, mb = logged.get((name, p, r), (0, -1))
            for attempt in (1, 2, 3):
                c0 = time.process_time(); w0 = time.perf_counter()
                z, sol, st, tr = K.run_main_ext(C, order, p, use_px, 0, kmax, N, seed, float(o), tl3, 0, mi, tl)
                ratio = (time.process_time() - c0) / (time.perf_counter() - w0)
                if ratio >= 0.985: break
            assert abs(K.eval_sol(sol, C) - z) < 1e-6 and len(set(sol.tolist())) == p, (name, p, z, o)
            tr = [[float(a), int(round(b)), int(c)] for a, b, c in tr]
            rb = min([b for a, b, c in tr if c <= mi]) if mi else -1
            w.writerow([algo, name, C.shape[0], p, o, tl, round(tl3, 3), r, seed, int(value_at(tr, tl)), int(value_at(tr, tl3)), int(round(z)),
                        int(st[14]), int(st[12]), int(st[0]), round(st[8], 3), mi, mb, rb, int(rb == mb) if mi else -1, round(st[13], 3)]
                       + [int(st[k]) for k in (1, 2, 3, 4, 5, 6, 7, 10, 11)] + [round(ratio, 4), attempt])
            traces[f'{name}_{p}_{r}'] = [[round(a, 4), b, c] for a, b, c in tr]
            f.flush()
        json.dump(traces, open(tpath, 'w')); print(algo, name, p, round(tl3, 1), flush=True)
    f.close()


if __name__ == '__main__':
    main()
