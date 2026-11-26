"""Analysis of the third campaign: the original POPSTAR program against MK-BPX and VNS at equal
elapsed time, on one machine, blocked by instance-p pair (experiments/campaign_blocked.py).

Usage:  python experiments/analyze_orig.py [DEST] [--results DIR] [--main DIR] [--sota DIR] [--replay DIR]

DEST       where numbers_orig.tex, numbers_orig.json, tables/tab_orig_*.tex and
           images/fig_orig_*.pdf are written (default: paper/)
--results  logs of the third campaign (default: results_orig/)
--main     logs of the main campaign (default: results/)
--sota     logs of the second campaign (default: results_sota/)
--replay   logs of the exact replay of the main campaign (default: results_replay/)

Equal-time rule.  POPSTAR has no time limit.  For each instance-p pair and each POPSTAR
configuration, t is the mean elapsed search time of the POPSTAR runs of the pair.  The value of a
run of MK-BPX or VNS is the best value it had found at time t, read from its improvement trace
after adding the time needed to sort the site lists (order_time.json), because the time of
POPSTAR includes the initialization of its distance oracle.

Configurations of POPSTAR
    P32   published: 32 multistart iterations, 10 elite solutions
    PT    multistart iterations scaled so that the run takes about 90 % of the time limit of
          Table 2; for the pairs in which this is not possible (PT_*_scaling.csv, scaled = 0) the
          published configuration already uses that time and its runs are used.

Reference value of a pair: the lowest of the published value and of the values found in any run
of the three campaigns (experiments/refvalues.py).  Deterministic (fixed bootstrap seed).
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'experiments'))
import json, glob
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import refvalues

_argv = sys.argv[1:]; RES = os.path.join(ROOT, 'results_orig'); MAIN = os.path.join(ROOT, 'results'); SOTA = os.path.join(ROOT, 'results_sota'); REPLAY = os.path.join(ROOT, 'results_replay')
for flag in ('--results', '--main', '--sota', '--replay'):
    if flag in _argv:
        _k = _argv.index(flag); _v = _argv[_k + 1]; del _argv[_k:_k + 2]
        if flag == '--results': RES = _v
        elif flag == '--main': MAIN = _v
        elif flag == '--replay': REPLAY = _v
        else: SOTA = _v
OUT = _argv[0] if _argv else os.path.join(ROOT, 'paper')
os.makedirs(f'{OUT}/tables', exist_ok=True); os.makedirs(f'{OUT}/images', exist_ok=True)
rng = np.random.default_rng(20261116)
TEST = ['fl1400', 'u1432', 'd2103', 'fl3795', 'rl5934']; DEV = ['rl1304', 'pcb3038']; SMALL = ['fl1400', 'u1432', 'd2103']; BIG = ['fl3795', 'rl5934']
ORL = [f'pmed{i}' for i in range(1, 41)]
LAB = {'POP': 'POPSTAR', 'BPX': 'MK-BPX', 'VNS': 'VNS'}
COL = {'POP': '#7b3294', 'BPX': '#0b3d91', 'VNS': '#d62728'}
num = {}
def N(k, v, fmt='{:.3f}'): num[k] = v if isinstance(v, str) else fmt.format(v)
def fP(P):
    if P < 1e-4:
        e = int(np.floor(np.log10(P))); return r'%.1f\times10^{%d}' % (P / 10 ** e, e)
    return f'{P:.4f}' if P < 0.01 else f'{P:.3f}'
def boot(x, B=20000):
    x = np.asarray(x, float); m = x[rng.integers(0, len(x), (B, len(x)))].mean(1); return np.percentile(m, [2.5, 97.5])

# ------------------------------------------------------------------ data
def read(pattern):
    fs = sorted(glob.glob(os.path.join(RES, pattern)))
    ds = [pd.read_csv(f) for f in fs]; ds = [d for d in ds if len(d)]
    tr = {}
    for f in fs:
        tf = f[:-4] + '_trace.json'
        if os.path.exists(tf): tr.update(json.load(open(tf)))
    return (pd.concat(ds, ignore_index=True) if ds else None), tr

pop32, tr32 = read('P32_S?.csv')
popT, trT = read('PT_S?.csv')
bpx, trB = read('S?_BPX.csv')
vns, trV = read('S?_VNS.csv')
_sf = sorted(glob.glob(os.path.join(RES, 'PT_S?_scaling.csv')))
scal = pd.concat([pd.read_csv(f) for f in _sf], ignore_index=True) if _sf else pd.DataFrame(columns=['inst', 'p', 'tl', 'a', 'b', 'oracle', 'wall32', 'k', 'scaled'])
order_time = json.load(open(os.path.join(RES, 'order_time.json')))
OT = {k: float(np.mean(list(v.values()))) for k, v in order_time.items()}
ref = refvalues.reference([MAIN, SOTA, RES])

def complete(d, cols=('inst', 'p')):
    """Keep the pairs with all their runs."""
    g = d.groupby(list(cols)).run.transform('count'); need = d.inst.map(lambda s: 30 if s.startswith('pmed') else (10 if s in ('fl1400', 'u1432', 'd2103', 'rl1304') else 5))
    return d[g == need].copy()
pop32 = complete(pop32); bpx = complete(bpx)
if vns is not None: vns = complete(vns)
if popT is not None: popT = complete(popT)

# the two POPSTAR configurations, pair by pair
CONF = {}
CONF['A'] = (pop32, tr32)
if popT is not None:
    sc = {(r.inst, r.p) for r in scal.itertuples() if r.scaled == 1}
    done = set(zip(popT.inst, popT.p))
    rest = pop32[[((i, p) not in sc) and not i.startswith('pmed') for i, p in zip(pop32.inst, pop32.p)]]
    missing = sc - done
    if missing: print('PT not complete yet for', sorted(missing))
    dT = pd.concat([popT, rest], ignore_index=True)
    trT2 = dict(trT)
    for r in rest.itertuples(): trT2[f'{r.inst}_{r.p}_{r.run}'] = tr32[f'{r.inst}_{r.p}_{r.run}']
    CONF['B'] = (dT, trT2)

def value_at(tr, t):
    """Best value at time t and whether a solution existed at that time."""
    v = [x[1] for x in tr if x[0] <= t]
    return (min(v), True) if v else (tr[0][1], False)

def refof(i, p): return float(ref[(i, p)])

def pair_table(conf):
    """One row per pair: POPSTAR and, at the mean POPSTAR time, MK-BPX and VNS."""
    dP, trP = CONF[conf]; rows = []
    for (i, p), g in dP.groupby(['inst', 'p'], sort=False):
        f = refof(i, p); t = g.wall.mean()
        row = dict(inst=i, p=p, n=int(g.n.iloc[0]), ref=f, tl=float(g.tl.iloc[0]), t=t, tmax=g.wall.max(), k=int(g.graspit.iloc[0]), R=len(g),
                   pop_gap=(100 * (g.best - f) / f).mean(), pop_hits=int((g.best <= f).sum()), cpu=g.cpu.mean(), gens=g.generations.mean())
        for key, d, tr in (('bpx', bpx, trB), ('vns', vns, trV)):
            if d is None: continue
            h = d[(d.inst == i) & (d.p == p)]
            if len(h) == 0: continue
            vals = []; nosol = 0; cover = True
            for r in h.itertuples():
                v, ok = value_at(tr[f'{i}_{p}_{r.run}'], t - OT[i]); vals.append(v); nosol += (not ok); cover &= (r.tl3 + OT[i] >= t - 1e-9)
            vals = np.array(vals, float)
            row[key + '_gap'] = (100 * (vals - f) / f).mean(); row[key + '_hits'] = int((vals <= f).sum()); row[key + '_R'] = len(h); row[key + '_nosol'] = nosol; row[key + '_cover'] = cover
            # value at the time limit of Table 2 (as in the other campaigns)
            row[key + '_gap_tl'] = (100 * (h.best_tl - f) / f).mean(); row[key + '_hits_tl'] = int((h.best_tl <= f).sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index(['inst', 'p'])

TAB = {c: pair_table(c) for c in CONF}
have_vns = vns is not None and 'vns_gap' in TAB['A']

def agg(t, key):
    R = t['R'] if key == 'pop' else t[key + '_R']
    return dict(pairs=len(t), gap=t[key + '_gap'].mean(), maxgap=t[key + '_gap'].max(), hit=100 * t[key + '_hits'].sum() / R.sum(), allopt=int((t[key + '_hits'] == R).sum()), runs=int(R.sum()))
EXACT = []
def cmp(t, a, b):
    """x = a, y = b; positive difference: b better."""
    t = t.dropna(subset=[a + '_gap', b + '_gap'])
    d = t[a + '_gap'].values - t[b + '_gap'].values; nz = d[d != 0]
    if len(nz) == 0: P = 1.0
    else:
        exact = len(np.unique(np.abs(np.round(nz, 9)))) == len(nz)
        P = stats.wilcoxon(nz, method='exact' if exact else 'approx').pvalue; EXACT.append(exact)
    ci = boot(d) if len(d) > 1 else (0, 0)
    return dict(P=P, w=int((d > 0).sum()), l=int((d < 0).sum()), t=int((d == 0).sum()), md=d.mean(), lo=ci[0], hi=ci[1], n=len(d), ax=agg(t, a), ay=agg(t, b))
def put(prefix, c, la='pop', lb='bpx'):
    N(prefix + 'Pairs', c['n'], '{:d}'); N(prefix + 'Runs', c['ax']['runs'], '{:d}')
    for l_, a in ((la, c['ax']), (lb, c['ay'])):
        N(prefix + l_ + 'Gap', a['gap'], '{:.4f}'); N(prefix + l_ + 'Hit', a['hit'], '{:.1f}'); N(prefix + l_ + 'All', a['allopt'], '{:d}'); N(prefix + l_ + 'Max', a['maxgap'], '{:.4f}')
    N(prefix + 'W', c['w'], '{:d}'); N(prefix + 'L', c['l'], '{:d}'); N(prefix + 'T', c['t'], '{:d}')
    N(prefix + 'SDiff', f"${c['md']:+.4f}$"); N(prefix + 'CI', f"$[{c['lo']:+.4f},\\,{c['hi']:+.4f}]$"); N(prefix + 'P', fP(c['P']))
    N(prefix + 'Better', LAB['BPX' if lb == 'bpx' else lb.upper()] if c['md'] > 0 else LAB['POP' if la == 'pop' else la.upper()])
    if c['ax']['gap'] > 0 and c['ay']['gap'] > 0: N(prefix + 'Ratio', max(c['ax']['gap'], c['ay']['gap']) / min(c['ax']['gap'], c['ay']['gap']), '{:.1f}')
def sub(t, insts, fn=None):
    return t[[(i in insts) and (fn is None or fn(p)) for i, p in t.index]]

# ------------------------------------------------------------------ comparisons
tests = []
CN = {'A': 'published configuration', 'B': 'scaled to the time limit'}
for c in CONF:
    t = TAB[c]
    for key, lab, insts, fn, prim in (('T', 'test set', TEST, None, True), ('D', 'development pairs', DEV, None, False), ('O', 'OR-Library', ORL, None, False),
                                      ('Ns', r'test set, $n\leq 2103$', SMALL, None, False), ('Nb', r'test set, $n\geq 3795$', BIG, None, False),
                                      ('Rlo', r'test set, $p\leq 20$', TEST, lambda p: p <= 20, False), ('Rmid', r'test set, $p=50$--$150$', TEST, lambda p: 50 <= p <= 150, False),
                                      ('Rhi', r'test set, $p=200$--$300$', TEST, lambda p: 200 <= p <= 300, False), ('Rtop', r'test set, $p\geq 400$', TEST, lambda p: p >= 400, False),
                                      ('Big', r'test set, $p\geq 100$', TEST, lambda p: p >= 100, False)):
        s = sub(t, insts, fn)
        if len(s) < 2 or 'bpx_gap' not in s or s.bpx_gap.isna().all(): continue
        cc = cmp(s, 'pop', 'bpx'); put('X' + c + key, cc)
        if key in ('T', 'D', 'O', 'Ns', 'Nb', 'Rlo', 'Rmid', 'Rhi', 'Rtop'): tests.append((c, key, lab, 'POPSTAR', 'MK-BPX', cc, prim))
        if have_vns and not s.vns_gap.isna().all():
            cv = cmp(s, 'pop', 'vns'); put('X' + c + key + 'V', cv, 'pop', 'vns')
            cb = cmp(s, 'vns', 'bpx'); put('X' + c + key + 'C', cb, 'vns', 'bpx')
            if key == 'T':
                tests.append((c, key + 'V', lab, 'POPSTAR', 'VNS', cv, False)); tests.append((c, key + 'C', lab, 'VNS', 'MK-BPX', cb, False))
        if key in ('T', 'D', 'Ns', 'Nb'):
            N('X' + c + key + 'Time', s.t.mean(), '{:.1f}'); N('X' + c + key + 'TimePct', 100 * (s.t / s.tl).mean(), '{:.0f}')
            N('X' + c + key + 'TimePctMin', 100 * (s.t / s.tl).min(), '{:.0f}'); N('X' + c + key + 'TimePctMax', 100 * (s.t / s.tl).max(), '{:.0f}')
            N('X' + c + key + 'LostList', ', '.join(f'{i} with $p={p}$' for (i, p) in s.index[(s.pop_gap < s.bpx_gap).values]) or 'none')
            N('X' + c + key + 'WonList', ', '.join(f'{i} with $p={p}$' for (i, p) in s.index[(s.pop_gap > s.bpx_gap).values]) or 'none')
# Holm over the two declared hypotheses of the third campaign (H7: published, H8: scaled; test set)
prim = [x for x in tests if x[6]]
ps = np.array([x[5]['P'] for x in prim]); o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
for r_, i_ in enumerate(o):
    run = max(run, (len(ps) - r_) * ps[i_]); adj[i_] = min(1, run)
HN = {'A': ('H7', 'Hseven'), 'B': ('H8', 'Height')}
L = [r'\begin{tabular}{llrrrrrr}', r'\toprule', r'Comparison & $x$ vs.\ $y$ & Pairs & $y$ better / worse / tie & Mean diff.\ (p.p.) & 95\% CI & $P$ & Holm $P$ \\', r'\midrule']
k_ = 0
for c in CONF:
    L.append(r'\multicolumn{8}{l}{\emph{POPSTAR, ' + CN[c] + r'}} \\')
    for cc_, key, lab, xa, ya, cc, pr in tests:
        if cc_ != c: continue
        if pr:
            h, hm = HN[c]; N(hm + 'P', fP(cc['P'])); N(hm + 'Padj', fP(adj[k_])); pa = '$' + fP(adj[k_]) + '$'; k_ += 1; lab2 = f'{h}: {lab}'
        else:
            pa = '--'; lab2 = lab
        L.append(f"\\quad {lab2[0].upper() + lab2[1:]} & {xa} vs.\\ {ya} & {cc['n']} & {cc['w']} / {cc['l']} / {cc['t']} & {cc['md']:+.4f} & [{cc['lo']:+.4f}, {cc['hi']:+.4f}] & ${fP(cc['P'])}$ & {pa} \\\\")
    L.append(r'\midrule')
L[-1] = r'\bottomrule'; L.append(r'\end{tabular}')
open(f'{OUT}/tables/tab_orig_tests.tex', 'w').write('\n'.join(L))

# ------------------------------------------------------------------ summary table
ALG3 = [('pop', 'POPSTAR'), ('bpx', 'MK-BPX')] + ([('vns', 'VNS')] if have_vns else [])
L = [r'\begin{tabular}{lllrrrrrrr}', r'\toprule', r'POPSTAR & Set & Algorithm & Pairs & Runs & Time (s) & Mean gap (\%) & Max mean gap (\%) & Optimal runs (\%) & All runs optimal \\', r'\midrule']
for c in CONF:
    t = TAB[c]; firstc = True
    for lab, insts, fn in (('Test set', TEST, None), (r'Test set, $p\geq 100$', TEST, lambda p: p >= 100), ('Development pairs', DEV, None), ('OR-Library', ORL, None)):
        s = sub(t, insts, fn)
        if len(s) == 0: continue
        first = True
        for key, la in ALG3:
            if key + '_gap' not in s: continue
            s2 = s.dropna(subset=[key + '_gap'])
            if len(s2) == 0: continue
            g = agg(s2, key)
            L.append(f"{({'A': 'Published', 'B': 'Scaled'}[c]) if firstc else ''} & {lab if first else ''} & {la} & {g['pairs']} & {g['runs']} & {s2.t.mean():.1f} & {g['gap']:.4f} & {g['maxgap']:.4f} & {g['hit']:.1f} & {g['allopt']}/{g['pairs']} \\\\"); first = False; firstc = False
        L.append(r'\addlinespace[2pt]')
    L[-1] = r'\midrule'
L[-1] = r'\bottomrule'; L.append(r'\end{tabular}')
open(f'{OUT}/tables/tab_orig_summary.tex', 'w').write('\n'.join(L))

# ------------------------------------------------------------------ by instance
L = [r'\begin{tabular}{lrrrrrrrrrrr}', r'\toprule',
     r' & & & \multicolumn{' + ('4' if have_vns else '3') + r'}{c}{Published configuration} & \multicolumn{' + ('5' if have_vns else '4') + r'}{c}{Scaled to the time limit} \\',
     r'\cmidrule(lr){4-' + ('7' if have_vns else '6') + r'}\cmidrule(lr){' + ('8-12' if have_vns else '7-10') + '}',
     r'Instance & $n$ & Pairs & Time (\%) & POPSTAR & MK-BPX' + (' & VNS' if have_vns else '') + r' & Iter. & Time (\%) & POPSTAR & MK-BPX' + (' & VNS' if have_vns else '') + r' \\', r'\midrule']
def g4(x): return ' & --' if np.isnan(x) else f' & {x:.4f}'
INAME = {'fl1400': 'Fla', 'fl3795': 'Flb', 'u1432': 'U', 'd2103': 'D', 'rl5934': 'Rlb', 'rl1304': 'Rla', 'pcb3038': 'Pcb'}
for ins in TEST + DEV:
    a = sub(TAB['A'], [ins])
    if len(a) == 0: continue
    if ins == DEV[0]: L.append(r'\midrule')
    row = f"{ins}{'$^{d}$' if ins in DEV else ''} & {int(a.n.iloc[0])} & {len(a)} & {100 * (a.t / a.tl).mean():.0f} & {a.pop_gap.mean():.4f} & {a.bpx_gap.mean():.4f}" + (g4(a.vns_gap.mean()) if have_vns else '')
    N('XAI' + INAME[ins] + 'pop', a.pop_gap.mean(), '{:.4f}'); N('XAI' + INAME[ins] + 'bpx', a.bpx_gap.mean(), '{:.4f}'); N('XAI' + INAME[ins] + 'Pct', 100 * (a.t / a.tl).mean(), '{:.0f}'); N('XAI' + INAME[ins] + 'Time', a.t.mean(), '{:.1f}')
    dd = a.pop_gap.values - a.bpx_gap.values; N('XAI' + INAME[ins] + 'W', int((dd > 0).sum()), '{:d}'); N('XAI' + INAME[ins] + 'L', int((dd < 0).sum()), '{:d}')
    if 'B' in TAB:
        b = sub(TAB['B'], [ins]).dropna(subset=['bpx_gap'])
        if len(b):
            kmin, kmax = int(b.k.min()), int(b.k.max())
            row += f" & {kmin if kmin == kmax else f'{kmin}--{kmax}'} & {100 * (b.t / b.tl).mean():.0f} & {b.pop_gap.mean():.4f} & {b.bpx_gap.mean():.4f}" + (g4(b.vns_gap.mean()) if have_vns else '')
            N('XBI' + INAME[ins] + 'pop', b.pop_gap.mean(), '{:.4f}'); N('XBI' + INAME[ins] + 'bpx', b.bpx_gap.mean(), '{:.4f}'); N('XBI' + INAME[ins] + 'Pct', 100 * (b.t / b.tl).mean(), '{:.0f}')
            dd = b.pop_gap.values - b.bpx_gap.values; N('XBI' + INAME[ins] + 'W', int((dd > 0).sum()), '{:d}'); N('XBI' + INAME[ins] + 'L', int((dd < 0).sum()), '{:d}')
        else: row += ' & & & & ' + ('& ' if have_vns else '')
    L.append(row + r' \\')
L += [r'\bottomrule', r'\end{tabular}']
open(f'{OUT}/tables/tab_orig_inst.tex', 'w').write('\n'.join(L))

# ------------------------------------------------------------------ per-pair tables
def bold(vals, i):
    s = f'{vals[i]:.4f}'
    return r'\textbf{' + s + '}' if vals[i] == min(vals) and max(vals) != min(vals) else s
def pairtable(insts, fname):
    L = [r'\begin{tabular}{lrr@{\hskip 10pt}rrrrr@{\hskip 10pt}rrrrrr}', r'\toprule',
         r' & & & \multicolumn{5}{c}{Published configuration (32 iterations)} & \multicolumn{6}{c}{Scaled to the time limit} \\', r'\cmidrule(lr){4-8}\cmidrule(lr){9-14}',
         r' & & & & \multicolumn{2}{c}{POPSTAR} & \multicolumn{2}{c}{MK-BPX} & & & \multicolumn{2}{c}{POPSTAR} & \multicolumn{2}{c}{MK-BPX} \\', r'\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){11-12}\cmidrule(lr){13-14}',
         r'Instance & $p$ & $f^*$ & $t$ (s) & Gap & Hits & Gap & Hits & Iter. & $t$ (s) & Gap & Hits & Gap & Hits \\', r'\midrule']
    for ins in insts:
        a = sub(TAB['A'], [ins])
        if len(a) == 0: continue
        for (i, p), r in a.iterrows():
            if np.isnan(r.bpx_gap): continue
            v = [r.pop_gap, r.bpx_gap]
            row = f"{ins}{'$^{d}$' if ins in DEV else ''} & {p} & {int(r.ref)} & {r.t:.1f} & {bold(v, 0)} & {r.pop_hits}/{r.R} & {bold(v, 1)} & {int(r.bpx_hits)}/{int(r.bpx_R)}"
            if 'B' in TAB and (i, p) in TAB['B'].index and not np.isnan(TAB['B'].bpx_gap[(i, p)]):
                q = TAB['B'].loc[(i, p)]; v = [q.pop_gap, q.bpx_gap]
                row += f" & {int(q.k)} & {q.t:.1f} & {bold(v, 0)} & {int(q.pop_hits)}/{int(q.R)} & {bold(v, 1)} & {int(q.bpx_hits)}/{int(q.bpx_R)}"
            else: row += ' & & & & & & '
            L.append(row + r' \\')
        L.append(r'\addlinespace[2pt]')
    L[-1] = r'\bottomrule'; L.append(r'\end{tabular}')
    open(f'{OUT}/tables/{fname}', 'w').write('\n'.join(L))
pairtable(SMALL, 'tab_orig_pairs1.tex')
pairtable(BIG + DEV, 'tab_orig_pairs2.tex')

# ------------------------------------------------------------------ descriptive numbers
tsp32 = pop32[~pop32.inst.str.startswith('pmed')]
N('XNRunsPop', len(pop32) + (len(popT) if popT is not None else 0), '{:d}'); N('XNRunsPopTsp', len(tsp32) + (len(popT) if popT is not None else 0), '{:d}')
N('XClockPop', 100 * ((tsp32.wall - tsp32.cpu) / tsp32.wall).mean(), '{:.1f}'); N('XClockPopMax', 100 * ((tsp32.wall - tsp32.cpu) / tsp32.wall).max(), '{:.1f}')
N('XSysPop', 100 * (tsp32.sys / (tsp32.user + tsp32.sys)).mean(), '{:.1f}')
N('XRepeatPop', int((pop32.attempts > 1).sum() + ((popT.attempts > 1).sum() if popT is not None else 0)), '{:d}')
tb = bpx[~bpx.inst.str.startswith('pmed')]
N('XCpuBpx', 100 * (1 - tb.cpu_ratio.mean()), '{:.1f}'); N('XRepeatBpx', int((bpx.attempts > 1).sum() + ((vns.attempts > 1).sum() if vns is not None else 0)), '{:d}')
N('XRepeatBpxTsp', int((tb.attempts > 1).sum() + ((vns[~vns.inst.str.startswith('pmed')].attempts > 1).sum() if vns is not None else 0)), '{:d}')
N('XRepeatPopTsp', int((tsp32.attempts > 1).sum() + ((popT.attempts > 1).sum() if popT is not None else 0)), '{:d}')
N('XNRunsOurs', len(bpx) + (len(vns) if vns is not None else 0), '{:d}')
N('XOrderMin', min(OT[i] for i in TEST + DEV if i in OT), '{:.2f}'); N('XOrderMax', max(OT[i] for i in TEST + DEV if i in OT), '{:.1f}')
N('XOracleMax', tsp32.oracle.max(), '{:.1f}')
tA = TAB['A']; tsA = tA[[not i.startswith('pmed') for i, _ in tA.index]]
N('XATimeMin', tsA.t.min(), '{:.1f}'); N('XATimeMax', tsA.t.max(), '{:.0f}'); N('XAOver', int((tsA.t > tsA.tl).sum()), '{:d}'); N('XAOverList', ', '.join(sorted({i for (i, p) in tsA.index[(tsA.t > tsA.tl).values]})) or 'none')
N('XAGens', tsp32.generations.mean(), '{:.1f}'); N('XAPostShare', 100 * ((tsp32.wall - tsp32.ms_end) / tsp32.wall).mean(), '{:.0f}')
for key in ('bpx', 'vns'):
    if key + '_nosol' in tsA: N('XANoSol' + key, int(tsA[key + '_nosol'].sum()), '{:d}')
    if key + '_cover' in tsA: N('XANoCover' + key, int((~tsA[key + '_cover'].astype(bool)).sum()), '{:d}')
if 'B' in TAB:
    tB = TAB['B']; s = scal
    N('XBScaled', int((s.scaled == 1).sum()), '{:d}'); N('XBNotScaled', int((s.scaled == 0).sum()), '{:d}'); N('XBKmin', int(s[s.scaled == 1].k.min()), '{:d}'); N('XBKmax', int(s[s.scaled == 1].k.max()), '{:d}')
    N('XBNotScaledList', ', '.join(sorted({r.inst for r in s.itertuples() if r.scaled == 0})) or 'none')
    sc_idx = [(r.inst, r.p) for r in s.itertuples() if r.scaled == 1 and (r.inst, r.p) in tB.index]
    q = tB.loc[sc_idx]
    N('XBTimePctScaled', 100 * (q.t / q.tl).mean(), '{:.0f}'); N('XBTimePctScaledMin', 100 * (q.t / q.tl).min(), '{:.0f}'); N('XBTimePctScaledMax', 100 * (q.t / q.tl).max(), '{:.0f}')
    N('XBOverScaled', int((q.t > q.tl).sum()), '{:d}')
    for key in ('bpx', 'vns'):
        if key + '_cover' in tB: N('XBNoCover' + key, int((~tB[key + '_cover'].dropna().astype(bool)).sum()), '{:d}')
    # does POPSTAR gain from the additional iterations?
    a_, b_ = TAB['A'].loc[sc_idx], q
    tt = [k for k in sc_idx if k[0] in TEST]
    if tt:
        N('XBGainA', a_.loc[tt].pop_gap.mean(), '{:.4f}'); N('XBGainB', b_.loc[tt].pop_gap.mean(), '{:.4f}'); N('XBGainPairs', len(tt), '{:d}')
        N('XBGainBpxA', a_.loc[tt].bpx_gap.mean(), '{:.4f}'); N('XBGainBpxB', b_.loc[tt].bpx_gap.mean(), '{:.4f}')
# time to the reference value in the pairs that all runs of both solve (published configuration)
def first_hit(tr, f, shift=0.0):
    t = [x[0] for x in tr if x[1] <= f]; return (t[0] + shift) if t else np.nan
both = [k for k in tsA.index if tsA.pop_hits[k] == tsA.R[k] and tsA.bpx_hits[k] == tsA.bpx_R[k]]
if both:
    tp = []; tb_ = []
    for (i, p) in both:
        f = refof(i, p)
        tp.append(np.mean([first_hit(tr32[f'{i}_{p}_{r}'], f) for r in range(int(tsA.R[(i, p)]))]))
        tb_.append(np.mean([first_hit(trB[f'{i}_{p}_{r}'], f, OT[i]) for r in range(int(tsA.bpx_R[(i, p)]))]))
    N('XATtPairs', len(both), '{:d}'); N('XATtPop', np.mean(tp), '{:.2f}'); N('XATtBpx', np.mean(tb_), '{:.2f}'); N('XATtBpxFaster', int((np.array(tb_) < np.array(tp)).sum()), '{:d}')
# OR-Library
oA = sub(tA, ORL)
if len(oA) and 'bpx_gap' in oA and not oA.bpx_gap.isna().all():
    oA = oA.dropna(subset=['bpx_gap'])
    N('XOInst', len(oA), '{:d}'); N('XOpopHit', 100 * oA.pop_hits.sum() / oA.R.sum(), '{:.1f}'); N('XObpxHit', 100 * oA.bpx_hits.sum() / oA.bpx_R.sum(), '{:.1f}')
    N('XOpopGap', oA.pop_gap.mean(), '{:.4f}'); N('XObpxGap', oA.bpx_gap.mean(), '{:.4f}'); N('XOpopAll', int((oA.pop_hits == oA.R).sum()), '{:d}'); N('XObpxAll', int((oA.bpx_hits == oA.bpx_R).sum()), '{:d}')
    N('XOTimeMax', oA.t.max(), '{:.2f}'); N('XOTimeMean', oA.t.mean(), '{:.2f}'); N('XOTimePct', 100 * (oA.t / oA.tl).mean(), '{:.0f}')
    N('XObpxHitTl', 100 * oA.bpx_hits_tl.sum() / oA.bpx_R.sum(), '{:.1f}'); N('XOpopRuns', int(oA.R.sum()), '{:d}')
    N('XONoSol', int(oA.bpx_nosol.sum()), '{:d}')

# ------------------------------------------------------------------ original program versus re-implementation (second campaign)
try:
    sp = pd.concat([pd.read_csv(os.path.join(SOTA, f'{s}_POPSTAR.csv')) for s in ('S1', 'S2')]); stp = {}
    for s in ('S1', 'S2'): stp.update(json.load(open(os.path.join(SOTA, f'{s}_POPSTAR_trace.json'))))
    rows = []
    for (i, p), g in sp.groupby(['inst', 'p']):
        f = refof(i, p); v = [min([b for a, b in stp[f'{i}_{p}_{r.run}'] if a <= r.tl] or [stp[f'{i}_{p}_{r.run}'][0][1]]) for r in g.itertuples()]
        rows.append(dict(inst=i, p=p, re_gap=(100 * (np.array(v, float) - f) / f).mean()))
    re = pd.DataFrame(rows).set_index(['inst', 'p'])
    if 'B' in TAB:
        j = TAB['B'].join(re, how='inner'); jt = sub(j, TEST)
        d = jt.re_gap.values - jt.pop_gap.values; nz = d[d != 0]
        N('XReGap', jt.re_gap.mean(), '{:.4f}'); N('XReOrigGap', jt.pop_gap.mean(), '{:.4f}'); N('XReW', int((d > 0).sum()), '{:d}'); N('XReL', int((d < 0).sum()), '{:d}'); N('XReT', int((d == 0).sum()), '{:d}')
        N('XReP', fP(stats.wilcoxon(nz).pvalue) if len(nz) else '1.000'); N('XRePairs', len(jt), '{:d}')
        for nm, ii in (('S', SMALL), ('B', BIG)):
            q = sub(j, ii); N('XRe' + nm + 'Gap', q.re_gap.mean(), '{:.4f}'); N('XRe' + nm + 'OrigGap', q.pop_gap.mean(), '{:.4f}'); N('XRe' + nm + 'OrigPct', 100 * (q.t / q.tl).mean(), '{:.0f}')
except FileNotFoundError:
    pass

# ------------------------------------------------------------------ replication of the main campaign on the third machine, exact replay
def gaps_tl(d):
    f = np.array([refof(i, p) for i, p in zip(d.inst, d.p)]); return 100 * (d.best_tl.values - f) / f
for key, d, algo in (('B', bpx, 'BPX'), ('V', vns, 'VNS')):
    if d is None: continue
    dm = pd.concat([pd.read_csv(os.path.join(MAIN, f'{s}_{algo}.csv')) for s in ('S0', 'S1', 'S2') if os.path.exists(os.path.join(MAIN, f'{s}_{algo}.csv'))])
    m = d.merge(dm[['inst', 'p', 'run', 'best', 'iters']], on=['inst', 'p', 'run'], suffixes=('', '_m'))
    mt = m[m.inst.isin(TEST)]
    if len(mt):
        fm = np.array([refof(i, p) for i, p in zip(mt.inst, mt.p)])
        gm = pd.Series(100 * (mt.best.values - fm) / fm).groupby([mt.inst.values, mt.p.values]).mean(); gn = pd.Series(gaps_tl(mt)).groupby([mt.inst.values, mt.p.values]).mean()
        N('XRep' + key + 'Pairs', len(gm), '{:d}'); N('XRep' + key + 'Runs', len(mt), '{:d}'); N('XRep' + key + 'MainGap', gm.mean(), '{:.4f}'); N('XRep' + key + 'NewGap', gn.mean(), '{:.4f}')
        N('XRep' + key + 'Same', 100 * (mt.best_tl == mt.best).mean(), '{:.1f}')
        q = (mt.groupby(['inst', 'p']).iters_tl.mean() / mt.groupby(['inst', 'p']).iters.mean())
        N('XRep' + key + 'Iter', q.mean(), '{:.2f}'); N('XRep' + key + 'IterMin', q.min(), '{:.2f}'); N('XRep' + key + 'IterMax', q.max(), '{:.2f}')
        N('XRep' + key + 'MainHit', 100 * (mt.best.values <= fm).mean(), '{:.1f}'); N('XRep' + key + 'NewHit', 100 * (mt.best_tl.values <= fm).mean(), '{:.1f}')
    _rf = sorted(glob.glob(os.path.join(REPLAY, f'S?_{algo}.csv')))
    rp = pd.concat([pd.read_csv(f) for f in _rf], ignore_index=True) if _rf else d.iloc[:0]
    rp = rp[rp.replay_ok >= 0]
    N('XReplay' + key + 'Runs', len(rp), '{:d}'); N('XReplay' + key + 'Ok', int((rp.replay_ok == 1).sum()), '{:d}')
if vns is not None:
    # the primary comparison of the paper (VNS against MK-BPX at the time limit of Table 2), on the third machine
    rows = []
    for (i, p), g in bpx[bpx.inst.isin(TEST)].groupby(['inst', 'p']):
        h = vns[(vns.inst == i) & (vns.p == p)]
        if len(h) == 0: continue
        f = refof(i, p); rows.append(dict(inst=i, p=p, R=len(g), bpx_R=len(g), vns_R=len(h), bpx_gap=(100 * (g.best_tl - f) / f).mean(), vns_gap=(100 * (h.best_tl - f) / f).mean(), bpx_hits=int((g.best_tl <= f).sum()), vns_hits=int((h.best_tl <= f).sum()), t=0, tl=1))
    if rows:
        c = cmp(pd.DataFrame(rows).set_index(['inst', 'p']), 'vns', 'bpx'); put('XHone', c, 'vns', 'bpx')
N('XReplayRuns', int(num.get('XReplayBRuns', '0')) + int(num.get('XReplayVRuns', '0')), '{:d}'); N('XReplayOk', int(num.get('XReplayBOk', '0')) + int(num.get('XReplayVOk', '0')), '{:d}')
pth = os.path.join(MAIN, 'trace_completion.json')
if os.path.exists(pth):
    tc = json.load(open(pth)); N('XTraceAdded', tc['added'], '{:d}'); N('XTraceChecked', tc['checked_existing'], '{:d}'); N('XTraceMismatch', tc['mismatch'], '{:d}')
# new reference values
IM = refvalues.improved([MAIN, SOTA], [RES])
N('XNImproved', len(IM), '{:d}')
N('XRefSentence', ('The third campaign found a better value than the reference of the earlier campaigns for ' + '; '.join(f'{i} with $p={p}$ ({new} against {old}, found by {who})' for i, p, old, new, who in IM)
                   + '. The gaps of all campaigns are computed with these values.') if IM else 'No run of the third campaign found a value better than the reference values of the earlier campaigns.')

# ------------------------------------------------------------------ blocks: drift of the machine inside a block and over the campaign
blk = pd.read_csv(os.path.join(RES, 'blocks.csv')); bt = blk[~blk.inst.str.startswith('pmed')]
PC = ['probe_start_0', 'probe_start_1', 'probe_end_0', 'probe_end_1']
N('XBlocks', len(blk), '{:d}'); N('XBlocksTsp', len(bt), '{:d}'); N('XBlockMin', bt.duration.min() / 60, '{:.1f}'); N('XBlockMax', bt.duration.max() / 60, '{:.1f}'); N('XBlockMean', bt.duration.mean() / 60, '{:.1f}')
N('XCampHours', (pd.to_datetime(blk.t_start).max() - pd.to_datetime(blk.t_start).min()).total_seconds() / 3600 + blk.duration.iloc[-1] / 3600, '{:.1f}')
ch = np.concatenate([(bt.probe_end_0 / bt.probe_start_0 - 1).values, (bt.probe_end_1 / bt.probe_start_1 - 1).values])
N('XProbeIn', 100 * np.abs(ch).mean(), '{:.1f}'); N('XProbeInMed', 100 * np.median(np.abs(ch)), '{:.1f}')
span = bt.groupby('inst').apply(lambda g: g[PC].values.max() / g[PC].values.min() - 1)
N('XProbeSpan', 100 * span.max(), '{:.0f}'); N('XProbeSpanMean', 100 * span.mean(), '{:.0f}')
if popT is not None:
    o32 = pop32.groupby(['inst', 'p']).oracle.mean(); oT = popT.groupby(['inst', 'p']).oracle.mean(); q = (oT / o32).dropna()
    N('XOracleIn', 100 * (q - 1).abs().mean(), '{:.1f}'); N('XOracleInMax', 100 * (q - 1).abs().max(), '{:.1f}'); N('XOracleBlocks', len(q), '{:d}')
    ospan = pd.concat([pop32, popT]).groupby('inst').apply(lambda g: g.groupby('p').oracle.mean().max() / g.groupby('p').oracle.mean().min() - 1)
    N('XOracleSpan', 100 * ospan[[i for i in ospan.index if not i.startswith('pmed')]].max(), '{:.0f}')
if 'core' in bpx:
    cb = tb.groupby(['inst', 'p', 'core']).iters_tl.mean().unstack('core'); N('XCoreRatio', (cb[1] / cb[0]).mean(), '{:.3f}')
    cp_ = tsp32.groupby(['inst', 'p', 'core']).cpu.mean().unstack('core'); N('XCoreRatioPop', (cp_[1] / cp_[0]).mean(), '{:.3f}')

# ------------------------------------------------------------------ figures
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
FL = 2e-4
ncf = len(CONF)
fig, axs = plt.subplots(ncf, 1, figsize=(7.4, 3.0 * ncf + 0.3), squeeze=False)
for ax, c in zip(axs[:, 0], CONF):
    t = TAB[c]; pos = 0; ticks = []; labs = []; bounds = []
    for ins in TEST + DEV:
        s = sub(t, [ins]).dropna(subset=['bpx_gap'])
        if len(s) == 0: continue
        start = pos
        for k, r in s.iterrows():
            a, b = max(r.pop_gap, FL), max(r.bpx_gap, FL)
            ax.plot([pos, pos], [a, b], color='0.6', lw=0.8, zorder=1)
            if have_vns and not np.isnan(r.vns_gap): ax.plot(pos, max(r.vns_gap, FL), 'o', color=COL['VNS'], ms=2.6, zorder=1, mfc='none', mew=0.7)
            ax.plot(pos, a, 'D', color=COL['POP'], ms=3.6, zorder=2); ax.plot(pos, b, 's', color=COL['BPX'], ms=3.6, zorder=3); pos += 1
        ticks.append((start + pos - 1) / 2); labs.append(ins + ('*' if ins in DEV else '')); bounds.append(pos - 0.5); pos += 1
    for bnd in bounds[:-1]: ax.axvline(bnd + 0.5, color='0.9', lw=0.8)
    ax.set_yscale('log'); ax.set_ylim(FL * 0.7, 1.5); ax.set_yticks([FL, 1e-3, 1e-2, 1e-1, 1]); ax.set_yticklabels(['0', '0.001', '0.01', '0.1', '1'])
    ax.set_xticks(ticks); ax.set_xticklabels(labs, fontsize=8); ax.set_ylabel('Mean gap (%)')
    ax.set_title({'A': '(a) POPSTAR in its published configuration; MK-BPX and VNS at the same elapsed time', 'B': '(b) POPSTAR scaled to the time limit of Table 2; MK-BPX and VNS at the same elapsed time'}[c], fontsize=8, loc='left')
axs[0, 0].plot([], [], 'D', color=COL['POP'], label='POPSTAR (original program)'); axs[0, 0].plot([], [], 's', color=COL['BPX'], label='MK-BPX')
if have_vns: axs[0, 0].plot([], [], 'o', color=COL['VNS'], mfc='none', label='VNS')
fig.tight_layout(rect=(0, 0, 1, 0.96)); hs, ls_ = axs[0, 0].get_legend_handles_labels()
fig.legend(hs, ls_, frameon=False, ncol=3, loc='upper center', fontsize=8)
fig.savefig(f'{OUT}/images/fig_orig_pairs.pdf'); plt.close(fig)

def curve(trs, f, ts, shift=0.0):
    """Mean gap at each time over the runs that already have a solution (nan before all have one)."""
    out = []
    for t in ts:
        v = []
        for tr in trs:
            w = [x[1] for x in tr if x[0] + shift <= t]
            v.append(min(w) if w else np.nan)
        out.append(np.nan if np.isnan(v).any() else 100 * (np.mean(v) - f) / f)
    return np.array(out)
cand = [('d2103', 300), ('d2103', 500), ('fl3795', 300), ('rl5934', 400), ('u1432', 300), ('fl1400', 300)]
pick = [c for c in cand if c in TAB['A'].index and not np.isnan(TAB['A'].bpx_gap[c])][:4]
if pick:
    import matplotlib.ticker as mt
    fig, axs = plt.subplots(1, len(pick), figsize=(7.4, 2.8))
    for ax, (ins, p) in zip(np.atleast_1d(axs), pick):
        f = refof(ins, p); rA = TAB['A'].loc[(ins, p)]; hb = bpx[(bpx.inst == ins) & (bpx.p == p)]
        tend = float(hb.tl3.iloc[0]) + OT[ins]; ts = np.linspace(tend * 0.02, tend, 120)
        ax.plot(ts, np.maximum(curve([trB[f'{ins}_{p}_{r}'] for r in hb.run], f, ts, OT[ins]), 1e-4), color=COL['BPX'], lw=1.5, label='MK-BPX')
        if have_vns:
            hv = vns[(vns.inst == ins) & (vns.p == p)]
            if len(hv): ax.plot(ts, np.maximum(curve([trV[f'{ins}_{p}_{r}'] for r in hv.run], f, ts, OT[ins]), 1e-4), color=COL['VNS'], lw=1.0, ls=':', label='VNS')
        ax.plot(ts, np.maximum(curve([tr32[f'{ins}_{p}_{r}'] for r in range(int(rA.R))], f, ts), 1e-4), color=COL['POP'], lw=1.3, ls='--', label='POPSTAR, published')
        ax.plot([rA.t], [max(rA.pop_gap, 1e-4)], 'D', color=COL['POP'], ms=4)
        if 'B' in TAB and (ins, p) in TAB['B'].index and int(TAB['B'].k[(ins, p)]) != 32:
            rB = TAB['B'].loc[(ins, p)]
            ax.plot(ts, np.maximum(curve([CONF['B'][1][f'{ins}_{p}_{r}'] for r in range(int(rB.R))], f, ts), 1e-4), color='#c2a5cf', lw=1.3, ls='-.', label='POPSTAR, scaled')
            ax.plot([rB.t], [max(rB.pop_gap, 1e-4)], 'D', color='#c2a5cf', ms=4)
        ax.axvline(float(rA.tl), color='0.8', lw=0.8, zorder=0)
        ax.set_yscale('log'); ax.set_title(f'{ins}, $p={p}$', fontsize=8); ax.set_xlabel('Time (s)')
        ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f'{v:g}')); ax.yaxis.set_minor_formatter(mt.NullFormatter())
    np.atleast_1d(axs)[0].set_ylabel('Mean gap (%)')
    hs, ls_ = [], []
    for ax in np.atleast_1d(axs):
        for h_, l_ in zip(*ax.get_legend_handles_labels()):
            if l_ not in ls_: hs.append(h_); ls_.append(l_)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.90, bottom=0.30, wspace=0.42)
    fig.legend(hs, ls_, fontsize=7.5, frameon=False, ncol=4, loc='lower center', bbox_to_anchor=(0.5, 0.0))
    fig.savefig(f'{OUT}/images/fig_orig_anytime.pdf'); plt.close(fig)

N('XAllExact', 'yes' if all(EXACT) else 'no')
with open(f'{OUT}/numbers_orig.tex', 'w') as fo:
    for k, v in sorted(num.items()):
        assert k.isalpha(), k
        fo.write('\\newcommand{\\m%s}{%s}\n' % (k, v))
json.dump(num, open(f'{OUT}/numbers_orig.json', 'w'), indent=1)
for c in TAB: TAB[c].to_csv(os.path.join(RES, f'summary_pairs_{c}.csv'))
print('OK', len(num), 'numbers; all Wilcoxon exact:', all(EXACT))
