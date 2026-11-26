"""Analysis of the second campaign (PopStar versus MK-BPX, run side by side).

Usage:  python experiments/analyze_sota.py [DEST] [--results DIR] [--main DIR]

DEST       where numbers_sota.tex, numbers_sota.json, tables/tab_sota_*.tex and
           images/fig_sota_*.pdf are written (default: paper/)
--results  logs of the second campaign (default: results_sota/)
--main     logs of the main campaign, used for the reference values and for the comparison of
           the two sets of MK-BPX runs (default: results/)

The value of a run is the best value found within the time limit, read from the improvement
trace, for both algorithms.  The reference value of a pair is the published value or the best
value found in any run of either campaign, whichever is lower.  Deterministic (fixed seed).
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import json, glob
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_argv = sys.argv[1:]; RES = os.path.join(ROOT, 'results_sota'); MAIN = os.path.join(ROOT, 'results')
for flag in ('--results', '--main'):
    if flag in _argv:
        _k = _argv.index(flag)
        if flag == '--results': RES = _argv[_k + 1]
        else: MAIN = _argv[_k + 1]
        del _argv[_k:_k + 2]
OUT = _argv[0] if _argv else os.path.join(ROOT, 'paper')
os.makedirs(f'{OUT}/tables', exist_ok=True); os.makedirs(f'{OUT}/images', exist_ok=True)
rng = np.random.default_rng(20261111)
TEST = ['fl1400', 'u1432', 'd2103', 'fl3795', 'rl5934']; DEV = ['rl1304', 'pcb3038']
ORL = [f'pmed{i}' for i in range(1, 41)]
LAB = {'POPSTAR': 'PopStar', 'BPX': 'MK-BPX'}
COL = {'POPSTAR': '#7b3294', 'BPX': '#0b3d91'}

def load(dirname, algo, sets=('S0', 'S1', 'S2')):
    ds = []; tr = {}
    for s in sets:
        f = os.path.join(dirname, f'{s}_{algo}.csv')
        if not os.path.exists(f): continue
        d = pd.read_csv(f)
        if len(d) == 0: continue
        ds.append(d)
        tf = f[:-4] + '_trace.json'
        if os.path.exists(tf): tr.update(json.load(open(tf)))
    return (pd.concat(ds, ignore_index=True) if ds else None), tr

def at_limit(d, tr):
    """Best value found within the time limit (first value of the trace if none)."""
    out = []
    for r in d.itertuples():
        t = tr[f'{r.inst}_{r.p}_{r.run}']
        v = [b for a, b in t if a <= r.tl]
        out.append(min(v) if v else t[0][1])
    return np.array(out, float)

runs = {}; trace = {}
for a in ('POPSTAR', 'BPX'):
    d, tr = load(RES, a)
    if d is None: continue
    d = d.copy(); d['val'] = at_limit(d, tr); d['over'] = d.time - d.tl
    runs[a] = d; trace[a] = tr
long = {}; ltrace = {}
for a in ('POPSTAR', 'BPX'):
    d, tr = load(RES, a, ('S3',))
    if d is None: continue
    d = d.copy(); d['val'] = at_limit(d, tr); long[a] = d; ltrace[a] = tr
main_bpx, main_tr = load(MAIN, 'BPX')
main_bpx = main_bpx.copy(); main_bpx['val'] = main_bpx.best.astype(float)     # main campaign: value returned by the run
rerun_best = runs['BPX'].copy(); rerun_best['val'] = rerun_best.best.astype(float)
# reference values: published, main campaign (all algorithms, including long runs), second campaign
allmain = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(MAIN, 'S?_*.csv')))])
pub = allmain.groupby(['inst', 'p']).opt.first()
ref_main = pd.concat([pub, allmain.groupby(['inst', 'p']).best.min()], axis=1).min(axis=1)
new_min = pd.concat([d.groupby(['inst', 'p']).best.min() for d in list(runs.values()) + list(long.values())], axis=1).min(axis=1)
ref = pd.concat([ref_main, new_min], axis=1).min(axis=1)
IMPR = [(i, p, int(ref_main[(i, p)]), int(ref[(i, p)])) for (i, p) in ref.index if ref[(i, p)] < ref_main[(i, p)]]
for d in list(runs.values()) + list(long.values()) + [main_bpx, rerun_best]:
    r = ref.reindex(pd.MultiIndex.from_arrays([d.inst, d.p])).values
    d['ref'] = r; d['gap'] = 100 * (d.val - r) / r

def pairtab(d):
    g = d.groupby(['inst', 'p'], sort=False)
    t = pd.DataFrame({'n': g.n.first(), 'ref': g.ref.first(), 'R': g.gap.count(), 'mgap': g.gap.mean(), 'hits': g.gap.apply(lambda x: int((x == 0).sum())),
                      'iters': g.iters.mean(), 'tl': g.tl.first()})
    full = t.R.groupby(level=0).transform('max')
    return t[t.R == full]
PT = {a: pairtab(d) for a, d in runs.items()}
PTM = pairtab(main_bpx); PTB = pairtab(rerun_best)
idx_all = PT['POPSTAR'].index.intersection(PT['BPX'].index)
num = {}
def N(k, v, fmt='{:.3f}'): num[k] = v if isinstance(v, str) else fmt.format(v)
def fP(P):
    if P < 1e-4:
        e = int(np.floor(np.log10(P))); return r'%.1f\times10^{%d}' % (P / 10 ** e, e)
    return f'{P:.4f}' if P < 0.01 else f'{P:.3f}'
def boot(x, B=20000):
    x = np.asarray(x, float); m = x[rng.integers(0, len(x), (B, len(x)))].mean(1); return np.percentile(m, [2.5, 97.5])
def sub(t, insts, idx=None):
    i = t.index if idx is None else idx
    return t.loc[[k for k in i if k[0] in insts and k in t.index]]
def agg(t):
    return dict(pairs=len(t), gap=t.mgap.mean(), hit=100 * t.hits.sum() / max(1, t.R.sum()), allopt=int((t.hits == t.R).sum()), found=int((t.hits > 0).sum()), runs=int(t.R.sum()), maxgap=t.mgap.max())
EXACT = []
def cmp(x, y):
    """positive difference: y better."""
    d = x.mgap.values - y.mgap.values; nz = d[d != 0]
    if len(nz) == 0: P = 1.0
    else:
        exact = len(np.unique(np.abs(np.round(nz, 9)))) == len(nz)
        P = stats.wilcoxon(nz, method='exact' if exact else 'approx').pvalue; EXACT.append(exact)
    ci = boot(d) if len(d) > 1 else (0, 0)
    return dict(P=P, w=int((d > 0).sum()), l=int((d < 0).sum()), t=int((d == 0).sum()), md=d.mean(), lo=ci[0], hi=ci[1], n=len(d), ax=agg(x), ay=agg(y))
def put(prefix, c):
    N(prefix + 'Pairs', c['n'], '{:d}'); N(prefix + 'Runs', c['ax']['runs'], '{:d}')
    for la, a in (('pop', c['ax']), ('bpx', c['ay'])):
        N(prefix + la + 'Gap', a['gap'], '{:.4f}'); N(prefix + la + 'Hit', a['hit'], '{:.1f}'); N(prefix + la + 'All', a['allopt'], '{:d}'); N(prefix + la + 'Max', a['maxgap'], '{:.4f}'); N(prefix + la + 'Found', a['found'], '{:d}')
    N(prefix + 'W', c['w'], '{:d}'); N(prefix + 'L', c['l'], '{:d}'); N(prefix + 'T', c['t'], '{:d}')
    N(prefix + 'Diff', abs(c['md']), '{:.4f}'); N(prefix + 'Lo', min(abs(c['lo']), abs(c['hi'])), '{:.4f}'); N(prefix + 'Hi', max(abs(c['lo']), abs(c['hi'])), '{:.4f}'); N(prefix + 'P', fP(c['P']))
    N(prefix + 'Sign', 'lower' if c['md'] > 0 else 'higher')
    N(prefix + 'CI', f"$[{c['lo']:+.4f},\\,{c['hi']:+.4f}]$"); N(prefix + 'SDiff', f"${c['md']:+.4f}$")
    if c['ay']['gap'] > 0 and c['ax']['gap'] > 0:
        N(prefix + 'Ratio', max(c['ax']['gap'], c['ay']['gap']) / min(c['ax']['gap'], c['ay']['gap']), '{:.1f}')

X, Y = PT['POPSTAR'].loc[idx_all], PT['BPX'].loc[idx_all]
tests = []
sets = [('T', 'H5', 'PopStar, test set', TEST, True), ('O', 'H6', 'PopStar, OR-Library', ORL, True), ('D', '', 'PopStar, development pairs', DEV, False)]
C = {}
for key, h, lab, insts, prim in sets:
    x, y = sub(X, insts), sub(Y, insts)
    if len(x) == 0: continue
    C[key] = cmp(x, y); put('P' + key, C[key]); tests.append((h, lab, C[key], prim))
    N('P' + key + 'LostList', ', '.join(f'{i} with $p={p}$' for (i, p) in x.index[(x.mgap < y.mgap).values]) or 'none')
# by range of p (TSPLIB-based pairs)
tsp_idx = [k for k in idx_all if not k[0].startswith('pmed')]
RNG = [('$p\\leq 20$', lambda p: p <= 20, 'Lo'), ('$p=50$--$150$', lambda p: 50 <= p <= 150, 'Mid'), ('$p=200$--$300$', lambda p: 200 <= p <= 300, 'Hi'), ('$p\\geq 400$', lambda p: p >= 400, 'Top')]
for lab, fn, key in RNG:
    ii = [k for k in tsp_idx if fn(k[1]) and k[0] in TEST]
    if len(ii) < 2: continue
    c = cmp(X.loc[ii], Y.loc[ii]); put('PR' + key, c); tests.append(('', f'Test set, {lab}', c, False))
if tsp_idx:
    ii = [k for k in tsp_idx if k[0] in TEST and k[1] >= 100]
    if ii:
        c = cmp(X.loc[ii], Y.loc[ii]); put('PBig', c)
    gi = pd.DataFrame({'x': X.loc[tsp_idx].mgap.groupby(level=0).mean(), 'y': Y.loc[tsp_idx].mgap.groupby(level=0).mean()})
    N('PInstW', int((gi.y < gi.x).sum()), '{:d}'); N('PInstL', int((gi.y > gi.x).sum()), '{:d}'); N('PInstN', len(gi), '{:d}')
    N('PInstP', f"{stats.binomtest(int((gi.y < gi.x).sum()), int((gi.y != gi.x).sum()), 0.5).pvalue:.3f}" if (gi.y != gi.x).any() else '1.000')
    INAME = {'fl1400': 'Fla', 'fl3795': 'Flb', 'u1432': 'U', 'd2103': 'D', 'rl5934': 'Rlb', 'rl1304': 'Rla', 'pcb3038': 'Pcb'}
    for ins in gi.index:
        N('PI' + INAME[ins] + 'pop', gi.x[ins], '{:.4f}'); N('PI' + INAME[ins] + 'bpx', gi.y[ins], '{:.4f}')
        dd = X.loc[[k_ for k_ in tsp_idx if k_[0] == ins]].mgap.values - Y.loc[[k_ for k_ in tsp_idx if k_[0] == ins]].mgap.values
        N('PI' + INAME[ins] + 'W', int((dd > 0).sum()), '{:d}'); N('PI' + INAME[ins] + 'L', int((dd < 0).sum()), '{:d}'); N('PI' + INAME[ins] + 'T', int((dd == 0).sum()), '{:d}')
    for lab, key, fn in ((r'Test set, $n\leq 2103$', 'PNs', lambda k_: k_[0] in ('fl1400', 'u1432', 'd2103')), (r'Test set, $n\geq 3795$', 'PNb', lambda k_: k_[0] in ('fl3795', 'rl5934'))):
        ii = [k_ for k_ in tsp_idx if fn(k_)]
        if len(ii) >= 2:
            c = cmp(X.loc[ii], Y.loc[ii]); put(key, c); tests.append(('', lab, c, False))
# Holm over the declared hypotheses of the second campaign
prim = [t for t in tests if t[3]]
ps = np.array([t[2]['P'] for t in prim]); o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
for r, i in enumerate(o):
    run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1, run)
names = {'H5': 'Hfive', 'H6': 'Hsix'}
L = [r'\begin{tabular}{llrrrrrr}', r'\toprule', r'Hypothesis & $x$ vs.\ $y$ & Pairs & $y$ better / worse / tie & Mean diff.\ (p.p.) & 95\% CI & $P$ & Holm $P$ \\', r'\midrule']
k = 0; ruled = False
for h, lab, c, pr in tests:
    if pr:
        N(names[h] + 'P', fP(c['P'])); N(names[h] + 'Padj', fP(adj[k])); pa = '$' + fP(adj[k]) + '$'; k += 1
    else:
        pa = '--'
        if not ruled: L.append(r'\midrule'); ruled = True
    L.append(f"{(h + ': ') if h else ''}{lab} & PopStar vs.\\ MK-BPX & {c['n']} & {c['w']} / {c['l']} / {c['t']} & {c['md']:+.4f} & [{c['lo']:+.4f}, {c['hi']:+.4f}] & ${fP(c['P'])}$ & {pa} \\\\")
# replication of MK-BPX: main campaign versus second campaign
ri = [k_ for k_ in PTB.index if k_ in PTM.index]
rt = [k_ for k_ in ri if k_[0] in TEST]
if rt:
    c = cmp(PTM.loc[rt], PTB.loc[rt])
    L.append(f"Replication, test set & MK-BPX (main) vs.\\ MK-BPX (second) & {c['n']} & {c['w']} / {c['l']} / {c['t']} & {c['md']:+.4f} & [{c['lo']:+.4f}, {c['hi']:+.4f}] & ${fP(c['P'])}$ & -- \\\\")
    N('RepPairs', c['n'], '{:d}'); N('RepMainGap', c['ax']['gap'], '{:.4f}'); N('RepNewGap', c['ay']['gap'], '{:.4f}'); N('RepMainHit', c['ax']['hit'], '{:.1f}'); N('RepNewHit', c['ay']['hit'], '{:.1f}')
    N('RepW', c['w'], '{:d}'); N('RepL', c['l'], '{:d}'); N('RepT', c['t'], '{:d}'); N('RepP', fP(c['P']))
    N('RepIter', (PT['BPX'].loc[rt].iters / PTM.loc[rt].iters).mean(), '{:.2f}')
    q = (PT['BPX'].loc[rt].iters / PTM.loc[rt].iters); N('RepIterMin', q.min(), '{:.2f}'); N('RepIterMax', q.max(), '{:.2f}')
    m = rerun_best.merge(main_bpx, on=['inst', 'p', 'run'], suffixes=('', '_m')); m = m[m.inst.isin(TEST)]
    N('RepSameRuns', 100 * (m.val == m.val_m).mean(), '{:.1f}'); N('RepRuns', len(m), '{:d}')
L += [r'\bottomrule', r'\end{tabular}']
open(f'{OUT}/tables/tab_sota_tests.tex', 'w').write('\n'.join(L))

# ------------------------------------------------------------ summary table
L = [r'\begin{tabular}{llrrrrrr}', r'\toprule', r'Set & Algorithm & Pairs & Runs & Mean gap (\%) & Max mean gap (\%) & Optimal runs (\%) & All runs optimal \\', r'\midrule']
for lab, insts, extra in (('Test set', TEST, None), (r'Test set, $p\geq 100$', TEST, lambda p: p >= 100), ('Development pairs', DEV, None), ('OR-Library', ORL, None)):
    ii = [k_ for k_ in idx_all if k_[0] in insts and (extra is None or extra(k_[1]))]
    if not ii: continue
    first = True
    for a in ('POPSTAR', 'BPX'):
        g = agg(PT[a].loc[ii])
        L.append(f"{lab if first else ''} & {LAB[a]} & {g['pairs']} & {g['runs']} & {g['gap']:.4f} & {g['maxgap']:.4f} & {g['hit']:.1f} & {g['allopt']}/{g['pairs']} \\\\"); first = False
    L.append(r'\midrule')
L[-1] = r'\bottomrule'; L.append(r'\end{tabular}')
open(f'{OUT}/tables/tab_sota_summary.tex', 'w').write('\n'.join(L))

# ------------------------------------------------------------ by instance
if tsp_idx:
    L = [r'\begin{tabular}{lrrrrrrr}', r'\toprule', r' & & & \multicolumn{2}{c}{Mean gap (\%)} & \multicolumn{2}{c}{Optimal runs (\%)} & MK-BPX better / \\', r'Instance & $n$ & Pairs & PopStar & MK-BPX & PopStar & MK-BPX & worse / tie \\', r'\midrule']
    for ins in TEST + DEV:
        ks = [k_ for k_ in tsp_idx if k_[0] == ins]
        if not ks: continue
        if ins == DEV[0]: L.append(r'\midrule')
        x, y = X.loc[ks], Y.loc[ks]; dd = x.mgap.values - y.mgap.values
        L.append(f"{ins}{'$^{d}$' if ins in DEV else ''} & {int(x.n.iloc[0])} & {len(ks)} & {x.mgap.mean():.4f} & {y.mgap.mean():.4f} & {100 * x.hits.sum() / x.R.sum():.1f} & {100 * y.hits.sum() / y.R.sum():.1f} & {int((dd > 0).sum())} / {int((dd < 0).sum())} / {int((dd == 0).sum())} \\\\")
    L += [r'\bottomrule', r'\end{tabular}']
    open(f'{OUT}/tables/tab_sota_inst.tex', 'w').write('\n'.join(L))

# ------------------------------------------------------------ per-pair tables
def pairtable(insts, fname):
    L = [r'\begin{tabular}{lrrrrrr}', r'\toprule', r'Instance & $p$ & $f^*$ & \multicolumn{2}{c}{PopStar} & \multicolumn{2}{c}{MK-BPX} \\', r'\cmidrule(lr){4-5}\cmidrule(lr){6-7}', r' & & & Gap & Hits & Gap & Hits \\', r'\midrule']
    for ins in insts:
        ks = [k_ for k_ in idx_all if k_[0] == ins]
        if not ks: continue
        for k_ in ks:
            row = f"{ins}{'$^{d}$' if ins in DEV else ''} & {k_[1]} & {int(X.ref[k_])}"
            vals = [X.mgap[k_], Y.mgap[k_]]; mn = min(vals)
            for t, v in ((X, vals[0]), (Y, vals[1])):
                s = f'{v:.4f}'
                if v == mn and vals[0] != vals[1]: s = r'\textbf{' + s + '}'
                row += f" & {s} & {int(t.hits[k_])}/{int(t.R[k_])}"
            L.append(row + r' \\')
        L.append(r'\addlinespace[2pt]')
    L[-1] = r'\bottomrule'; L.append(r'\end{tabular}')
    open(f'{OUT}/tables/{fname}', 'w').write('\n'.join(L))
pairtable(['fl1400', 'u1432', 'd2103'], 'tab_sota_pairs1.tex')
pairtable(['fl3795', 'rl5934', 'rl1304', 'pcb3038'], 'tab_sota_pairs2.tex')

# ------------------------------------------------------------ behaviour of PopStar and overruns
dp = runs['POPSTAR']; db = runs['BPX']
tp = dp[~dp.inst.str.startswith('pmed')]; tb = db[~db.inst.str.startswith('pmed')]
if len(tp):
    imp = tp.imp_ms.sum() + tp.imp_pr.sum() + tp.imp_po.sum()
    N('PopImpMs', 100 * tp.imp_ms.sum() / imp, '{:.1f}'); N('PopImpPr', 100 * tp.imp_pr.sum() / imp, '{:.1f}'); N('PopImpPo', 100 * tp.imp_po.sum() / imp, '{:.1f}')
    N('PopPrFound', 100 * tp.pr_found.sum() / tp.pr_calls.sum(), '{:.1f}'); N('PopCyclesMin', tp.groupby(['inst', 'p']).cycles.mean().min(), '{:.1f}'); N('PopCyclesMax', tp.groupby(['inst', 'p']).cycles.mean().max(), '{:.1f}')
    N('PopItersMin', tp.groupby(['inst', 'p']).iters.mean().min(), '{:.0f}'); N('PopItersMax', tp.groupby(['inst', 'p']).iters.mean().max(), '{:.0f}')
    N('PopNoPost', int((tp.groupby(['inst', 'p']).gens.max() == 0).sum()), '{:d}')
    bigp = tp[tp.inst.isin(['fl3795', 'rl5934'])]; bigb = tb[tb.inst.isin(['fl3795', 'rl5934'])]
    if len(bigp): N('PopItersBig', bigp.iters.mean(), '{:.0f}'); N('BpxItersBig', bigb.iters.mean(), '{:.0f}'); N('PopCyclesBig', bigp.cycles.mean(), '{:.1f}')
    N('PopOver', 100 * (tp.over / tp.tl).mean(), '{:.1f}'); N('BpxOver', 100 * (tb.over / tb.tl).mean(), '{:.1f}')
    N('PopOverMax', 100 * (tp.over / tp.tl).max(), '{:.1f}'); N('BpxOverMax', 100 * (tb.over / tb.tl).max(), '{:.1f}')
    N('PopLate', int((tp.best < tp.val).sum()), '{:d}'); N('BpxLate', int((tb.best < tb.val).sum()), '{:d}')
N('PNImproved', len(IMPR), '{:d}')
if IMPR:
    who = []
    for i, p, old, new in IMPR:
        al = [LAB[a] for a in ('POPSTAR', 'BPX') if any(((d_.inst == i) & (d_.p == p) & (d_.best < old)).any() for d_ in ([runs[a]] + ([long[a]] if a in long else [])))]
        who.append(f"{i} with $p={p}$ ({new} against {old}, found by {' and '.join(al)})")
    N('PRefSentence', 'The second campaign found a better value than the reference of the main campaign for ' + '; '.join(who) + '.')
else:
    N('PRefSentence', 'No run of the second campaign found a value better than the reference values of the main campaign.')
N('PMaxit', int(dp.maxit.iloc[0]), '{:d}'); N('PElite', int(dp.elite.iloc[0]), '{:d}')
# time to reach the reference value, pairs in which both reached it in every run
both = [k_ for k_ in tsp_idx if X.hits[k_] == X.R[k_] and Y.hits[k_] == Y.R[k_]]
def first_hit(a, k_):
    d = runs[a]; d = d[(d.inst == k_[0]) & (d.p == k_[1])]
    ts = []
    for r in d.itertuples():
        t = [x for x, v in trace[a][f'{r.inst}_{r.p}_{r.run}'] if v <= r.ref]
        ts.append(t[0])
    return np.mean(ts)
if both:
    tx = np.array([first_hit('POPSTAR', k_) for k_ in both]); ty = np.array([first_hit('BPX', k_) for k_ in both])
    N('PTtPairs', len(both), '{:d}'); N('PTtPop', tx.mean(), '{:.2f}'); N('PTtBpx', ty.mean(), '{:.2f}'); N('PTtBpxFaster', int((ty < tx).sum()), '{:d}')

orl_idx = [k_ for k_ in idx_all if k_[0].startswith('pmed') and X.hits[k_] == X.R[k_] and Y.hits[k_] == Y.R[k_]]
if orl_idx:
    ox = np.array([first_hit('POPSTAR', k_) for k_ in orl_idx]); oy = np.array([first_hit('BPX', k_) for k_ in orl_idx])
    tls = np.array([X.tl[k_] for k_ in orl_idx])
    N('POTtPairs', len(orl_idx), '{:d}'); N('POTtPop', 100 * (ox / tls).mean(), '{:.0f}'); N('POTtBpx', 100 * (oy / tls).mean(), '{:.0f}'); N('POTtBpxFaster', int((oy < ox).sum()), '{:d}'); N('POTtPopFaster', int((ox < oy).sum()), '{:d}')
# ------------------------------------------------------------ long runs (five times the time limit)
if 'POPSTAR' in long and 'BPX' in long:
    def lt(d):
        g = d.groupby(['inst', 'p'], sort=False); t = pd.DataFrame({'R': g.gap.count(), 'mgap': g.gap.mean(), 'hits': g.gap.apply(lambda x: int((x == 0).sum()))}); return t[t.R == 5]
    lx, ly = lt(long['POPSTAR']), lt(long['BPX']); li = lx.index.intersection(ly.index); lx, ly = lx.loc[li], ly.loc[li]
    sx, sy = X.loc[li], Y.loc[li]
    cl = cmp(lx, ly); put('PL', cl); cs = cmp(sx, sy); put('PLs', cs)
    N('PLLostList', ', '.join(f'{i} with $p={p}$' for (i, p) in li[(lx.mgap < ly.mgap).values]) or 'none')
    cx = cmp(sx, lx); N('PLpopImp', sx.mgap.mean() / lx.mgap.mean() if lx.mgap.mean() > 0 else 0, '{:.1f}'); N('PLbpxImp', sy.mgap.mean() / ly.mgap.mean() if ly.mgap.mean() > 0 else 0, '{:.1f}')
    L = [r'\begin{tabular}{lrrrrrrrr}', r'\toprule', r' & \multicolumn{4}{c}{Time limit of Table~\ref{tab:instances}} & \multicolumn{4}{c}{Five times that limit} \\', r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}',
         r' & \multicolumn{2}{c}{PopStar} & \multicolumn{2}{c}{MK-BPX} & \multicolumn{2}{c}{PopStar} & \multicolumn{2}{c}{MK-BPX} \\', r'Instance & Gap & Opt.\ (\%) & Gap & Opt.\ (\%) & Gap & Opt.\ (\%) & Gap & Opt.\ (\%) \\', r'\midrule']
    for ins in ['fl1400', 'u1432', 'd2103']:
        if ins not in li.get_level_values(0): continue
        r = ins
        for t in (sx, sy, lx, ly): r += f" & {t.loc[ins].mgap.mean():.4f} & {100 * t.loc[ins].hits.sum() / t.loc[ins].R.sum():.0f}"
        L.append(r + r' \\')
        for la, t in (('pop', lx), ('bpx', ly)): N('PL' + ''.join(ch for ch in ins if ch.isalpha()) + la, t.loc[ins].mgap.mean(), '{:.4f}')
    r = r'\midrule' + '\nAll'
    for t in (sx, sy, lx, ly): r += f" & {t.mgap.mean():.4f} & {100 * t.hits.sum() / t.R.sum():.0f}"
    L += [r + r' \\', r'\bottomrule', r'\end{tabular}']
    open(f'{OUT}/tables/tab_sota_long.tex', 'w').write('\n'.join(L))
    tests_long = (cl, cs)

# ------------------------------------------------------------ development table of PopStar
pth = os.path.join(RES, 'dev_popstar.csv')
if os.path.exists(pth):
    dv = pd.read_csv(pth)
    L = [r'\begin{tabular}{lrrrrrrr}', r'\toprule', r' & \multicolumn{3}{c}{rl1304 (10 s)} & \multicolumn{3}{c}{pcb3038 (40 s)} & \\', r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}', r'Iterations per cycle & $p=100$ & 300 & 500 & $p=100$ & 300 & 500 & Mean \\', r'\midrule']
    for mi in sorted(dv.maxit.unique(), reverse=True):
        v = [dv[(dv.maxit == mi) & (dv.inst == ins) & (dv.p == p)].gap_at_tl.mean() for ins in ('rl1304', 'pcb3038') for p in (100, 300, 500)]
        L.append(f"{mi}{' (default, retained)' if mi == 32 else ''} & " + ' & '.join(f'{x:.4f}' for x in v) + f' & {np.mean(v):.4f} \\\\')
        N('PDev' + {32: 'A', 16: 'B', 8: 'C'}[mi], np.mean(v), '{:.4f}')
    L += [r'\bottomrule', r'\end{tabular}']
    open(f'{OUT}/tables/tab_sota_dev.tex', 'w').write('\n'.join(L))
pth = os.path.join(RES, 'verify_popstar.json')
if os.path.exists(pth):
    v_ = json.load(open(pth)); assert v_['mismatch'] == 0; N('PVerPairs', v_['pairs'], '{:d}'); N('PVerMin', v_['with_local_minimum'], '{:d}')
pth = os.path.join(RES, 'fidelity_popstar.csv')
if os.path.exists(pth):
    fd = pd.read_csv(pth)
    for cls, key in (('fl1400', 'Fl'), ('orlib', 'Or')):
        s = fd[fd.cls == cls]
        if len(s) == 0: continue
        g = s.groupby('p' if cls == 'fl1400' else 'inst')
        N('PFid' + key + 'Dev', g.dev.mean().mean(), '{:.3f}' if cls == 'fl1400' else '{:.4f}'); N('PFid' + key + 'Pub', g.pub_dev.first().mean(), '{:.3f}'); N('PFid' + key + 'N', len(g), '{:d}'); N('PFid' + key + 'Runs', int(g.dev.count().iloc[0]), '{:d}')
        N('PFid' + key + 'Time', s.time.mean(), '{:.1f}'); N('PFid' + key + 'Better', int((g.dev.mean() <= g.pub_dev.first() + 1e-9).sum()), '{:d}')
        if cls == 'orlib': N('PFidOrHit', 100 * (s.dev == 0).mean(), '{:.1f}'); N('PFidOrMedOpt', int((g.dev.median() == 0).sum()), '{:d}')
        if cls == 'fl1400': N('PFidFlNeg', int((g.dev.min() < -1e-3).sum()), '{:d}')

# ------------------------------------------------------------ figures
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
if tsp_idx:
    fig, ax = plt.subplots(figsize=(7.4, 3.4)); FL = 2e-4
    pos = 0; ticks = []; labs = []; bounds = []
    for ins in TEST + DEV:
        ks = [k_ for k_ in tsp_idx if k_[0] == ins]
        if not ks: continue
        start = pos
        for k_ in ks:
            a, b_ = max(X.mgap[k_], FL), max(Y.mgap[k_], FL)
            ax.plot([pos, pos], [a, b_], color='0.6', lw=0.8, zorder=1)
            ax.plot(pos, a, 'D', color=COL['POPSTAR'], ms=3.6, zorder=2); ax.plot(pos, b_, 's', color=COL['BPX'], ms=3.6, zorder=3)
            pos += 1
        ticks.append((start + pos - 1) / 2); labs.append(ins + ('*' if ins in DEV else '')); bounds.append(pos - 0.5); pos += 1
    for bnd in bounds[:-1]: ax.axvline(bnd + 0.5, color='0.9', lw=0.8)
    ax.set_yscale('log'); ax.set_ylim(FL * 0.7, 1); ax.set_yticks([FL, 1e-3, 1e-2, 1e-1, 1]); ax.set_yticklabels(['0', '0.001', '0.01', '0.1', '1'])
    ax.set_xticks(ticks); ax.set_xticklabels(labs, fontsize=8); ax.set_ylabel('Mean gap to the reference value (%)')
    ax.plot([], [], 'D', color=COL['POPSTAR'], label='PopStar'); ax.plot([], [], 's', color=COL['BPX'], label='MK-BPX'); ax.legend(frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=8)
    fig.tight_layout(); fig.savefig(f'{OUT}/images/fig_sota_pairs.pdf'); plt.close(fig)
    def value_at(tr, t):
        b = tr[0][1]
        for a, v in tr:
            if a <= t: b = v
            else: break
        return b
    cand = [('d2103', 300), ('d2103', 500), ('fl3795', 300), ('rl5934', 400), ('d2103', 200), ('fl3795', 500), ('rl5934', 200), ('d2103', 100)]
    pick = [c for c in cand if c in idx_all][:4]
    if pick:
        import matplotlib.ticker as mt
        fig, axs = plt.subplots(1, len(pick), figsize=(7.4, 2.7))
        for ax, (ins, p) in zip(np.atleast_1d(axs), pick):
            tl = X.tl[(ins, p)]; opt = X.ref[(ins, p)]; ts = np.linspace(tl * 0.05, tl, 60)
            for a in ('POPSTAR', 'BPX'):
                R = int(PT[a].R[(ins, p)])
                Yv = [np.mean([100 * (value_at(trace[a][f'{ins}_{p}_{r}'], t) - opt) / opt if trace[a][f'{ins}_{p}_{r}'][0][0] <= t else np.nan for r in range(R)]) for t in ts]
                ax.plot(ts, np.maximum(Yv, 1e-4), color=COL[a], label=LAB[a], lw=1.5, ls='-' if a == 'BPX' else '--')
            ax.set_yscale('log'); ax.set_title(f'{ins}, $p={p}$', fontsize=8); ax.set_xlabel('Time (s)')
            ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f'{v:g}')); ax.yaxis.set_minor_formatter(mt.NullFormatter())
        np.atleast_1d(axs)[0].set_ylabel('Mean gap (%)')
        hs, ls_ = np.atleast_1d(axs)[0].get_legend_handles_labels()
        fig.subplots_adjust(left=0.09, right=0.99, top=0.90, bottom=0.30, wspace=0.42)
        fig.legend(hs, ls_, fontsize=7.5, frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(0.5, 0.0))
        fig.savefig(f'{OUT}/images/fig_sota_anytime.pdf'); plt.close(fig)
N('PAllExact', 'yes' if all(EXACT) else 'no'); print('all Wilcoxon exact:', all(EXACT))
with open(f'{OUT}/numbers_sota.tex', 'w') as f:
    for k_, v_ in sorted(num.items()):
        assert k_.isalpha(), k_
        f.write('\\newcommand{\\m%s}{%s}\n' % (k_, v_))
json.dump(num, open(f'{OUT}/numbers_sota.json', 'w'), indent=1)
print('OK', len(num)); print(json.dumps({k: v for k, v in num.items() if k.startswith(('PT', 'PO', 'PD', 'Rep', 'Hf', 'Hs', 'PBig'))}, indent=0))
