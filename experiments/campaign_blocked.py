"""Third campaign: the original POPSTAR program against MK-BPX and VNS, blocked by instance-p pair.

Usage:  python experiments/campaign_blocked.py GROUP [--out DIR] [--pmm DIR] [--main DIR]

GROUP   test_small  fl1400, u1432, d2103            (10 runs)
        test_big    fl3795, rl5934                  (5 runs)
        dev         rl1304 (10 runs), pcb3038 (5 runs)
        orlib       pmed1-pmed40                    (30 runs)
--out   output directory (default results_orig/)
--pmm   directory of the distance matrices in the PMM format of POPSTAR (default data/pmm;
        missing files are written with tools/export_pmm.py from the matrices that MK-BPX uses)
--main  logs of the main campaign (default results/), only copied into the columns main_*

Why blocks.  The speed of a shared virtual machine drifts by 10-20 % over the hours that a
campaign lasts (and changes when the machine is restarted), which is of the order of the
differences that an equal-time comparison has to resolve.  Everything that is compared is
therefore measured within one block of a few minutes.  For each instance-p pair, in this order
and always with two processes running, one pinned to each of the first two cores:

  1. POPSTAR, published configuration (32 multistart iterations, 10 elite solutions): R runs,
     seed = run + 1; runs with an even index on core 0 and the others on core 1.
  2. POPSTAR scaled to the time limit T of Table 2 (not for OR-Library): k = floor((0.9 T - o - b) / a)
     multistart iterations, where a, b and o are the mean elapsed times of one multistart iteration,
     of the post-optimization and of the initialization in step 1.  Run only if k >= 40; otherwise
     the published configuration already uses the time limit and its runs stand for this one.
  3. MK-BPX and, on the test instances, VNS: the runs of the main campaign (same seeds, hence the
     same sequences of solutions), continued until H = max(T, 1.1 x longest POPSTAR run of the
     block), so that their improvement traces give the value at T and at the time POPSTAR needed.
     Run r of MK-BPX goes to core r mod 2 and run r of VNS to the other core.

Controls.  A POPSTAR run whose elapsed time exceeds its CPU time by more than 5 % + 0.15 s, or a
run of MK-BPX or VNS whose CPU time is below 98.5 % of its elapsed time (machine disturbed), is
repeated, at most twice (`attempts`).  Every solution written by POPSTAR is re-evaluated with the
distance matrix of this study.  A fixed computation (sorting the site lists of the instance) is
timed on both cores at the beginning and at the end of every block; blocks.csv keeps these
probes, which measure how much the speed of the machine changed inside the block.

A block is written to disk only when it is complete, and complete blocks are skipped when the
script is restarted.  Files:
P32_<SET>.csv, PT_<SET>.csv, PT_<SET>_scaling.csv, <SET>_BPX.csv, <SET>_VNS.csv, the *_trace.json
files, order_time.json and blocks.csv.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'experiments'))
import json, csv, time, subprocess, multiprocessing as mp
import numpy as np, pandas as pd
from mkbpx import instances as inst
import campaign as CP

GROUPS = {'test_small': ['fl1400', 'u1432', 'd2103'], 'test_big': ['fl3795', 'rl5934'], 'dev': ['rl1304', 'pcb3038'], 'orlib': [f'pmed{i}' for i in range(1, 41)]}
TEST = GROUPS['test_small'] + GROUPS['test_big']
ELITE = 10
HP = ('algo config inst n p opt tl graspit elite run seed best gap wall cpu init oracle prtime ms_end generations grasp '
      'user sys wall_total ms_iters po_relinks attempts core').split()
HB = ('algo inst n p opt tl tl3 run seed best_tl best_tl3 best_end iters_tl iters_tl3 iters_end time_end '
      'main_iters main_best replay_best replay_ok t_replay imp_plain imp_px px_calls px_take comps comps_taken scans viol dsum '
      'cpu_ratio attempts core').split()
HK = 'set inst p tl R t_start duration k scaled a b oracle wall32 wallT H probe_start_0 probe_start_1 probe_end_0 probe_end_1'.split()


def worker(core, name, qin, qout):
    """One process per core: runs POPSTAR (as a child pinned to the same core) and MK-BPX / VNS."""
    os.sched_setaffinity(0, {core})
    from mkbpx import core as K, popstar_orig as PO
    C = inst.load_matrix(name); order = K.build_order(C)
    K.run_main_ext(C, order, min(5, C.shape[0] // 4), 1, 0, 5, 1, 1, 0., 0.05, 0, 0, 0.0); K.run_main_ext(C, order, min(5, C.shape[0] // 4), 0, 0, 5, 1, 1, 0., 0.05, 0, 0, 0.0)

    def probe():
        a = time.perf_counter(); K.build_order(C); return time.perf_counter() - a
    qout.put(('ready', core, [probe() for _ in range(3)]))
    while True:
        job = qin.get()
        if job is None: break
        try:
            _work(job, core, C, order, K, PO, probe, qout)
        except Exception as e:                               # report instead of leaving the parent waiting
            qout.put(('error', core, repr(e)))


def _work(job, core, C, order, K, PO, probe, qout):
    if True:
        if job[0] == 'probe':
            qout.put(('probe', core, probe()))
        elif job[0] == 'pop':
            _, pmm, p, r, k = job
            for attempt in (1, 2, 3):
                x = PO.run(pmm, p, r + 1, graspit=k, elite=ELITE, core=core)
                if x['wall'] - x['cpu'] <= 0.05 * x['wall'] + 0.15: break
            sol = x.pop('solution'); z = x['best']
            x['ok'] = bool(len(sol) == p and len(set(sol.tolist())) == p and abs(K.eval_sol(sol, C) - z) < 1e-6 and abs(x['trace'][-1][1] - z) < 1e-6)
            x['attempt'] = attempt; x['core'] = core; x['run'] = r
            qout.put(('pop', core, x))
        else:
            _, algo, p, r, seed, o, H, T = job
            aid, use_px, kmax0, N = CP.ALGOS[algo]; kmax = p if kmax0 == 0 else min(p, kmax0)
            for attempt in (1, 2, 3):
                c0 = time.process_time(); w0 = time.perf_counter()
                z, sol, st, tr = K.run_main_ext(C, order, p, use_px, 0, kmax, N, seed, float(o), H, 0, 0, T)
                ratio = (time.process_time() - c0) / (time.perf_counter() - w0)
                if ratio >= 0.985: break
            ok = bool(abs(K.eval_sol(sol, C) - z) < 1e-6 and len(set(sol.tolist())) == p)
            qout.put(('run', core, dict(algo=algo, run=r, seed=seed, z=float(z), ok=ok, st=[float(v) for v in st], tr=[[float(a), int(round(b)), int(c)] for a, b, c in tr],
                                        ratio=float(ratio), attempt=attempt, core=core)))


def value_at(tr, t):
    v = [b for a, b, _ in tr if a <= t]
    return min(v) if v else tr[0][1]


def setof(name):
    return 'S0' if name.startswith('pmed') else ('S1' if name in ('fl1400', 'u1432', 'rl1304', 'd2103') else 'S2')


def append(path, header, rows):
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.writer(f)
        if new: w.writerow(header)
        w.writerows(rows)


def purge(path, keep):
    """Drop the rows of blocks that were not completed (after an interruption)."""
    if not os.path.exists(path): return
    rows = list(csv.reader(open(path))); h = rows[0]; i, j = h.index('inst'), h.index('p')
    good = [r for r in rows[1:] if (r[i], int(r[j])) in keep]
    if len(good) != len(rows) - 1:
        with open(path, 'w', newline='') as f: csv.writer(f).writerows([h] + good)


def main():
    argv = sys.argv[1:]; OUT = os.path.join(ROOT, 'results_orig'); MAIN = os.path.join(ROOT, 'results'); PMM = os.path.join(ROOT, 'data', 'pmm')
    for flag in ('--out', '--main', '--pmm'):
        if flag in argv:
            k = argv.index(flag); v = argv[k + 1]; del argv[k:k + 2]
            if flag == '--out': OUT = v
            elif flag == '--main': MAIN = v
            else: PMM = v
    if len(argv) != 1 or argv[0] not in GROUPS:
        print(__doc__); sys.exit(1)
    os.makedirs(OUT, exist_ok=True); os.makedirs(PMM, exist_ok=True)
    names = GROUPS[argv[0]]
    allpairs = {}
    for s in ('S0', 'S1', 'S2'):
        for nm, p, tl, R in CP.pairs(s): allpairs.setdefault(nm, []).append((p, tl, R))
    bpath = os.path.join(OUT, 'blocks.csv'); opath = os.path.join(OUT, 'order_time.json')
    done = set()
    if os.path.exists(bpath):
        done = {(r.inst, int(r.p)) for r in pd.read_csv(bpath).itertuples()}
    for s in ('S0', 'S1', 'S2'):                       # rows of interrupted blocks of this group
        for f in (f'P32_{s}.csv', f'PT_{s}.csv', f'PT_{s}_scaling.csv', f'{s}_BPX.csv', f'{s}_VNS.csv'):
            path = os.path.join(OUT, f)
            if os.path.exists(path):
                rows = list(csv.reader(open(path))); h = rows[0]; i, j = h.index('inst'), h.index('p')
                purge(path, {(r[i], int(r[j])) for r in rows[1:] if (r[i], int(r[j])) in done or r[i] not in names})
    logged = {}
    for s in ('S0', 'S1', 'S2'):
        for algo in ('BPX', 'VNS'):
            mp_ = os.path.join(MAIN, f'{s}_{algo}.csv')
            if os.path.exists(mp_):
                for r in pd.read_csv(mp_).itertuples(): logged[(algo, r.inst, r.p, r.run)] = (int(r.iters), int(r.best))
    ctx = mp.get_context('spawn')
    for name in names:
        todo = [(p, tl, R) for p, tl, R in allpairs[name] if (name, p) not in done]
        if not todo: continue
        s = setof(name)
        pmm = os.path.join(PMM, name + '.pmm')
        if not os.path.exists(pmm):
            subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 'export_pmm.py'), name, '--out', PMM], check=True, stdout=subprocess.DEVNULL)
        n = inst.load_matrix(name).shape[0]
        qin = [ctx.Queue(), ctx.Queue()]; qout = ctx.Queue()
        ws = [ctx.Process(target=worker, args=(c, name, qin[c], qout), daemon=True) for c in (0, 1)]
        for w in ws: w.start()
        pr = []
        for _ in ws: pr += qout.get()[2]
        ot = json.load(open(opath)) if os.path.exists(opath) else {}
        ot[name] = {'median': round(float(np.median(pr)), 4)}; json.dump(ot, open(opath, 'w'), indent=1)

        def collect(k):
            out = []
            for _ in range(k):
                m = qout.get()
                if m[0] == 'error': raise RuntimeError(f'worker on core {m[1]}: {m[2]}')
                out.append(m)
            return out

        def probes():
            for c in (0, 1): qin[c].put(('probe',))
            d = {m[1]: m[2] for m in collect(2)}
            return d[0], d[1]

        def popstar(p, R, k):
            for r in range(R): qin[r % 2].put(('pop', pmm, p, r, k))
            res = sorted((m[2] for m in collect(R)), key=lambda x: x['run'])
            assert all(x['ok'] for x in res), (name, p, k)
            return res

        for p, tl, R in todo:
            t0 = time.time(); o = inst.opt(name, p)
            ps = probes()
            r32 = popstar(p, R, 32)
            a = float(np.mean([(x['ms_end'] - x['oracle']) / x['ms_iters'] for x in r32])); b = float(np.mean([x['wall'] - x['ms_end'] for x in r32]))
            oc = float(np.mean([x['oracle'] for x in r32])); w32 = float(np.mean([x['wall'] for x in r32]))
            k = int(np.floor((0.9 * tl - oc - b) / a)) if s != 'S0' else 0
            scaled = int(s != 'S0' and k >= 40)
            rT = popstar(p, R, k) if scaled else []
            H = max(tl, 1.1 * max(x['wall'] for x in r32 + rT))
            algos = ['BPX'] + (['VNS'] if name in TEST else [])
            njobs = 0
            for r in range(R):
                for ai, algo in enumerate(algos):
                    aid = CP.ALGOS[algo][0]
                    qin[(r + ai) % 2].put(('run', algo, p, r, aid * 10**7 + CP.PAIRID[(name, p)] * 100 + r, o, H, tl)); njobs += 1
            runs = [m[2] for m in collect(njobs)]
            assert all(x['ok'] for x in runs), (name, p)
            pe = probes()
            # ---- the block is complete: write it
            for config, res in (('P32', r32), ('PT', rT)):
                if not res: continue
                rows = []; path = os.path.join(OUT, f'{config}_{s}.csv'); tpath = path[:-4] + '_trace.json'
                tr = json.load(open(tpath)) if os.path.exists(tpath) else {}
                for x in res:
                    z = x['best']
                    rows.append(['POPSTAR', config, name, n, p, o, tl, 32 if config == 'P32' else k, ELITE, x['run'], x['run'] + 1, int(round(z)), round(100 * (z - o) / o, 6),
                                 round(x['wall'], 4), round(x['cpu'], 4), round(x['init'], 4), round(x['oracle'], 4), round(x['prtime'], 4), round(x['ms_end'], 4), x['generations'],
                                 int(round(x['grasp'])), round(x['user'], 3), round(x['sys'], 3), round(x['wall_total'], 3), x['ms_iters'], x['po_relinks'], x['attempt'], x['core']])
                    tr[f"{name}_{p}_{x['run']}"] = [[round(u, 4), int(round(v))] for u, v in x['trace']]
                append(path, HP, rows); json.dump(tr, open(tpath, 'w'))
            if s != 'S0':
                append(os.path.join(OUT, f'PT_{s}_scaling.csv'), 'inst p tl a b oracle wall32 k scaled'.split(), [[name, p, tl, round(a, 5), round(b, 4), round(oc, 4), round(w32, 3), k, scaled]])
            for algo in algos:
                rows = []; path = os.path.join(OUT, f'{s}_{algo}.csv'); tpath = path[:-4] + '_trace.json'
                tr = json.load(open(tpath)) if os.path.exists(tpath) else {}
                for x in sorted((x for x in runs if x['algo'] == algo), key=lambda x: x['run']):
                    st = x['st']; mi, mb = logged.get((algo, name, p, x['run']), (0, -1))
                    rows.append([algo, name, n, p, o, tl, round(H, 3), x['run'], x['seed'], int(value_at(x['tr'], tl)), int(value_at(x['tr'], H)), int(round(x['z'])),
                                 int(st[14]) if st[14] >= 0 else int(st[0]), int(st[12]) if st[12] >= 0 else int(st[0]), int(st[0]), round(st[8], 3), mi, mb, -1, -1, -1]
                                + [int(st[q]) for q in (1, 2, 3, 4, 5, 6, 7, 10, 11)] + [round(x['ratio'], 4), x['attempt'], x['core']])
                    tr[f"{name}_{p}_{x['run']}"] = [[round(u, 4), v, c] for u, v, c in x['tr']]
                append(path, HB, rows); json.dump(tr, open(tpath, 'w'))
            append(bpath, HK, [[s, name, p, tl, R, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0)), round(time.time() - t0, 1), k, scaled, round(a, 5), round(b, 4), round(oc, 4),
                                round(w32, 3), round(float(np.mean([x['wall'] for x in rT])), 3) if rT else '', round(H, 3)] + [round(v, 4) for v in ps + pe]])
            print(time.strftime('%H:%M:%S'), name, p, 'k', k if scaled else 32, 'H', round(H, 1), 'probes', [round(v, 3) for v in ps + pe], flush=True)
        for c in (0, 1): qin[c].put(None)
        for w in ws: w.join()


if __name__ == '__main__':
    main()
