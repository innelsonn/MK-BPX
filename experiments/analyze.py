"""Analysis of the campaign: every number, table and figure of the paper.

Usage:  python experiments/analyze.py [DEST] [--results DIR]

DEST       where numbers.tex, numbers.json, tables/ and images/ are written (default: paper/)
--results  directory with the run-level logs (default: results/)
The analysis is deterministic (fixed bootstrap seed).
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

_argv = sys.argv[1:]; RES = os.path.join(ROOT, 'results')
if '--results' in _argv:
    _k = _argv.index('--results'); RES = _argv[_k + 1]; del _argv[_k:_k + 2]
OUT = _argv[0] if _argv else os.path.join(ROOT, 'paper')
REFDIR = os.path.join(ROOT, 'data', 'reference')
os.makedirs(f'{OUT}/tables', exist_ok=True); os.makedirs(f'{OUT}/images', exist_ok=True)
rng = np.random.default_rng(20261110)
TEST = ['fl1400', 'u1432', 'd2103', 'fl3795', 'rl5934']; DEV = ['rl1304', 'pcb3038']; SMALL = ['fl1400', 'u1432', 'd2103']
ALG = ['VNS', 'VNS20', 'BPX', 'BPX4']
LAB = {'VNS': 'VNS', 'VNS20': r'VNS$_{20}$', 'BPX': 'MK-BPX', 'BPX4': r'MK-BPX$_{4}$'}
COL = {'VNS': '#d62728', 'VNS20': '#ff9f1c', 'BPX': '#0b3d91', 'BPX4': '#2ca02c'}
runs = {}; trace = {}; long = {}
for f in sorted(glob.glob(os.path.join(RES, 'S?_*.csv'))):
    s, a = os.path.basename(f)[:-4].split('_', 1)
    d = pd.read_csv(f)
    if len(d) == 0: continue
    if s == 'S3':
        long[a] = d; continue
    runs[a] = pd.concat([runs[a], d]) if a in runs else d
    tf = f[:-4] + '_trace.json'
    if os.path.exists(tf): trace.setdefault(a, {}).update(json.load(open(tf)))
# reference value: published optimal / best-known value, or a better value found in our runs
allr = pd.concat(list(runs.values()) + list(long.values()))
ref = allr.groupby(['inst', 'p']).agg(pub=('opt', 'first'), ours=('best', 'min'))
ref['ref'] = ref[['pub', 'ours']].min(axis=1)
IMPROVED = ref[ref.ours < ref.pub]
for d in list(runs.values()) + list(long.values()):
    r = ref.ref.reindex(pd.MultiIndex.from_arrays([d.inst, d.p])).values
    d['opt'] = r; d['gap'] = 100 * (d.best - r) / r
def pairtab(a):
    d = runs[a]; g = d.groupby(['inst', 'p'], sort=False)
    t = pd.DataFrame({'n': g.n.first(), 'opt': g.opt.first(), 'R': g.gap.count(), 'mgap': g.gap.mean(), 'hits': g.gap.apply(lambda x: int((x == 0).sum())),
                      'best': g.best.min(), 'iters': g.iters.mean(), 'tl': g.tl.first(), 'imp_px': g.imp_px.sum(), 'imp_plain': g.imp_plain.sum(),
                      'px_calls': g.px_calls.sum(), 'px_take': g.px_take.sum(), 'comps': g.comps.sum(), 'taken': g.comps_taken.sum(), 'dsum': g.dsum.sum(), 'viol': g.viol.sum()})
    # keep only complete pairs
    full = t.R.groupby(level=0).transform('max')
    return t[t.R == full]
PT = {a: pairtab(a) for a in runs}
num = {}
def N(k, v, fmt='{:.3f}'): num[k] = v if isinstance(v, str) else fmt.format(v)
def fP(P):
    if P < 1e-4:
        e = int(np.floor(np.log10(P))); return r'%.1f\times10^{%d}' % (P / 10 ** e, e)
    return f'{P:.4f}' if P < 0.01 else f'{P:.3f}'
def boot(x, B=20000):
    x = np.asarray(x, float); m = x[rng.integers(0, len(x), (B, len(x)))].mean(1); return np.percentile(m, [2.5, 97.5])
def sel(a, insts):
    t = PT[a]; return t[[i in insts for i in t.index.get_level_values(0)]]
def common(a, b, insts):
    x, y = sel(a, insts), sel(b, insts); idx = x.index.intersection(y.index); return x.loc[idx], y.loc[idx]
def agg(t):
    return dict(pairs=len(t), gap=t.mgap.mean(), hit=100 * t.hits.sum() / t.R.sum(), allopt=int((t.hits == t.R).sum()), found=int((t.hits > 0).sum()), runs=int(t.R.sum()), maxgap=t.mgap.max())
EXACT = []
def cmp(a, b, insts):
    """x = a, y = b: positive difference means b better."""
    x, y = common(a, b, insts); d = x.mgap.values - y.mgap.values; nz = d[d != 0]
    if len(nz) == 0: P = 1.0; exact = True
    else:
        exact = len(np.unique(np.abs(np.round(nz, 9)))) == len(nz)
        P = stats.wilcoxon(nz, method='exact' if exact else 'approx').pvalue
        EXACT.append(exact)
    ci = boot(d) if len(d) > 1 else (0, 0)
    return dict(P=P, w=int((d > 0).sum()), l=int((d < 0).sum()), t=int((d == 0).sum()), md=d.mean(), lo=ci[0], hi=ci[1], n=len(d), ax=agg(x), ay=agg(y))
ORL = [f'pmed{i}' for i in range(1, 41)]
tests = []
def put(prefix, la, lb, a):
    N(prefix + 'Pairs', a['n'], '{:d}'); N(prefix + la + 'Gap', a['ax']['gap']); N(prefix + lb + 'Gap', a['ay']['gap'])
    N(prefix + la + 'Hit', a['ax']['hit'], '{:.1f}'); N(prefix + lb + 'Hit', a['ay']['hit'], '{:.1f}')
    N(prefix + la + 'All', a['ax']['allopt'], '{:d}'); N(prefix + lb + 'All', a['ay']['allopt'], '{:d}')
    N(prefix + la + 'Found', a['ax']['found'], '{:d}'); N(prefix + lb + 'Found', a['ay']['found'], '{:d}')
    N(prefix + la + 'Max', a['ax']['maxgap']); N(prefix + lb + 'Max', a['ay']['maxgap']); N(prefix + 'Runs', a['ax']['runs'], '{:d}')
    N(prefix + 'W', a['w'], '{:d}'); N(prefix + 'L', a['l'], '{:d}'); N(prefix + 'T', a['t'], '{:d}')
    N(prefix + 'Diff', abs(a['md'])); N(prefix + 'Lo', min(abs(a['lo']), abs(a['hi']))); N(prefix + 'Hi', max(abs(a['lo']), abs(a['hi']))); N(prefix + 'P', fP(a['P']))
    if a['ay']['gap'] > 0: N(prefix + 'Ratio', a['ax']['gap'] / a['ay']['gap'], '{:.1f}')
have = lambda *al: all(a in PT for a in al)
if have('VNS', 'BPX'):
    T = cmp('VNS', 'BPX', TEST); put('T', 'vns', 'bpx', T); tests.append(('H1', 'Crossover, test set', 'VNS', 'BPX', T, True))
    x_, y_ = common('VNS', 'BPX', TEST); N('TBothAll', int(((x_.hits == x_.R) & (y_.hits == y_.R)).sum()), '{:d}')
    N('TLostList', ', '.join(f'{i} with $p={p}$' for (i, p) in x_.index[(x_.mgap < y_.mgap).values]))
    N('TvnsHits', int(x_.hits.sum()), '{:d}'); N('TbpxHits', int(y_.hits.sum()), '{:d}')
    xa, ya = common('VNS', 'BPX', TEST + DEV); gi = pd.DataFrame({'v': xa.mgap.groupby(level=0).mean(), 'b': ya.mgap.groupby(level=0).mean()})
    N('InstW', int((gi.b < gi.v).sum()), '{:d}'); N('InstN', len(gi), '{:d}'); N('TInstW', int((gi.loc[[i for i in TEST if i in gi.index]].b < gi.loc[[i for i in TEST if i in gi.index]].v).sum()), '{:d}')
    N('InstP', f"{stats.binomtest(int((gi.b < gi.v).sum()), len(gi), 0.5).pvalue:.3f}")
    Dv = cmp('VNS', 'BPX', DEV)
    if Dv['n'] > 0: put('D', 'vns', 'bpx', Dv)
    if len(sel('BPX', ORL)) > 0:
        O = cmp('VNS', 'BPX', ORL); put('O', 'vns', 'bpx', O); tests.append(('H2', 'Crossover, OR-Library', 'VNS', 'BPX', O, True))
    A = cmp('VNS', 'BPX', TEST + DEV); put('A', 'vns', 'bpx', A)
if have('VNS20', 'BPX'):
    S3 = cmp('VNS20', 'BPX', SMALL); put('S', 'vnst', 'bpx', S3); tests.append(('H3', 'Tuned control', 'VNS20', 'BPX', S3, True))
    S5 = cmp('VNS', 'VNS20', SMALL); put('V', 'vns', 'vnst', S5)
if have('BPX4', 'BPX'):
    S4 = cmp('BPX', 'BPX4', SMALL); put('F', 'bpx', 'bpxf', S4); tests.append(('H4', 'Four elite copies', 'BPX', 'BPX4', S4, True))
if have('VNS', 'BPX') and Dv['n'] > 0: tests.append(('', 'Crossover, development pairs', 'VNS', 'BPX', Dv, False))
if have('VNS20', 'VNS'): tests.append(('', 'Shaking range of VNS', 'VNS', 'VNS20', S5, False))
prim = [t for t in tests if t[5]]
ps = np.array([t[4]['P'] for t in prim]); o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
for r, i in enumerate(o):
    run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1, run)
names = {'H1': 'Hone', 'H2': 'Htwo', 'H3': 'Hthree', 'H4': 'Hfour'}
L = [r'\begin{tabular}{llrrrrrr}', r'\toprule', r'Hypothesis & $x$ vs.\ $y$ & Pairs & $y$ better / worse / tie & Mean diff.\ (p.p.) & 95\% CI & $P$ & Holm $P$ \\', r'\midrule']
k = 0
for t in tests:
    h, lab, a, b, c, pr = t
    if pr:
        N(names[h] + 'P', fP(c['P'])); N(names[h] + 'Padj', fP(adj[k])); pa = '$' + fP(adj[k]) + '$'; k += 1
    else:
        pa = '--'
        if lab.startswith('Crossover, dev'): L.append(r'\midrule')
    L.append(f"{(h + ': ') if h else ''}{lab} & {LAB[a]} vs.\\ {LAB[b]} & {c['n']} & {c['w']} / {c['l']} / {c['t']} & {c['md']:+.4f} & [{c['lo']:+.4f}, {c['hi']:+.4f}] & ${fP(c['P'])}$ & {pa} \\\\")
L += [r'\bottomrule', r'\end{tabular}']
open(f'{OUT}/tables/tab_tests.tex', 'w').write('\n'.join(L))

# ------------------------------------------------------------ summary table
L = [r'\begin{tabular}{llrrrrrr}', r'\toprule', r'Set & Algorithm & Pairs & Runs & Mean gap (\%) & Max mean gap (\%) & Optimal runs (\%) & All runs optimal \\', r'\midrule']
def block(label, insts, algs, ref='BPX'):
    idx = None
    for a in algs:
        if a in PT:
            i = sel(a, insts).index; idx = i if idx is None else idx.intersection(i)
    if idx is None or len(idx) == 0: return
    first = True
    for a in algs:
        if a not in PT: continue
        g = agg(sel(a, insts).loc[idx])
        L.append(f"{label if first else ''} & {LAB[a]} & {g['pairs']} & {g['runs']} & {g['gap']:.4f} & {g['maxgap']:.4f} & {g['hit']:.1f} & {g['allopt']}/{g['pairs']} \\\\"); first = False
    L.append(r'\midrule')
block('Test set', TEST, ['VNS', 'BPX'])
block(r'Test set, $n\leq 2103$', SMALL, ['VNS', 'VNS20', 'BPX', 'BPX4'])
block('Development pairs', DEV, ['VNS', 'BPX'])
block('OR-Library', ORL, ['VNS', 'BPX'])
L[-1] = r'\bottomrule'; L.append(r'\end{tabular}')
open(f'{OUT}/tables/tab_summary.tex', 'w').write('\n'.join(L))

# ------------------------------------------------------------ per-pair tables
def pairtable(insts, algs, fname):
    algs = [a for a in algs if a in PT]
    L = [r'\begin{tabular}{lrr' + 'rr' * len(algs) + '}', r'\toprule', 'Instance & $p$ & $f^*$' + ''.join(r' & \multicolumn{2}{c}{%s}' % LAB[a] for a in algs) + r' \\',
         ''.join(r'\cmidrule(lr){%d-%d}' % (4 + 2 * j, 5 + 2 * j) for j in range(len(algs))), ' & & ' + ' & Gap & Hits' * len(algs) + r' \\', r'\midrule']
    for ins in insts:
        if ins not in PT[algs[0]].index.get_level_values(0): continue
        for p in PT[algs[0]].loc[ins].index:
            row = f"{ins}{'$^{d}$' if ins in DEV else ''} & {p} & {int(PT[algs[0]].loc[(ins, p)].opt)}"
            vals = [PT[a].mgap.get((ins, p), np.nan) for a in algs]; mn = np.nanmin(vals)
            for a, v in zip(algs, vals):
                if np.isnan(v): row += ' & -- & --'
                else:
                    s = f'{v:.4f}'
                    if v == mn and np.sum(np.array(vals) == mn) < np.sum(~np.isnan(vals)): s = r'\textbf{' + s + '}'
                    row += f" & {s} & {int(PT[a].hits[(ins, p)])}/{int(PT[a].R[(ins, p)])}"
            L.append(row + r' \\')
        L.append(r'\addlinespace[2pt]')
    L[-1] = r'\bottomrule'; L.append(r'\end{tabular}')
    open(f'{OUT}/tables/{fname}', 'w').write('\n'.join(L))
if have('VNS', 'BPX'):
    pairtable(['fl1400', 'u1432', 'd2103', 'rl1304'], ['VNS', 'VNS20', 'BPX', 'BPX4'], 'tab_pairs1.tex')
    pairtable(['fl3795', 'rl5934', 'pcb3038'], ['VNS', 'BPX'], 'tab_pairs2.tex')

# ------------------------------------------------------------ mechanics
if 'BPX' in PT:
    b = sel('BPX', TEST + DEV).copy(); b['pv'] = b.index.get_level_values(1)
    N('BpxSharePct', 100 * b.imp_px.sum() / (b.imp_px.sum() + b.imp_plain.sum()), '{:.1f}')
    N('PxTakePct', 100 * b.px_take.sum() / b.px_calls.sum(), '{:.1f}'); N('Viol', int(b.viol.sum() + (sel('BPX', ORL).viol.sum() if len(sel('BPX', ORL)) else 0)), '{:d}')
    N('PxCalls', '{:,}'.format(int(b.px_calls.sum() + (sel('BPX', ORL).px_calls.sum() if len(sel('BPX', ORL)) else 0))).replace(',', '{,}'))
    grp = [('$p\\leq 20$', b.pv <= 20, 'Lo'), ('$p=50$--$150$', (b.pv >= 50) & (b.pv <= 150), 'Mid'), ('$p=200$--$300$', (b.pv >= 200) & (b.pv <= 300), 'Hi'), ('$p\\geq 400$', b.pv >= 400, 'Top')]
    v = sel('VNS', TEST + DEV).loc[b.index]
    L = [r'\begin{tabular}{lrrrrrrrr}', r'\toprule',
         r' & & \multicolumn{2}{c}{Mean gap (\%)} & Iterations & Calls with a & Components & Facilities & Improvements \\',
         r'Range of $p$ & Pairs & VNS & MK-BPX & ratio & selection (\%) & per call & differing & by BPX (\%) \\', r'\midrule']
    for lab, m, key in grp:
        if m.sum() == 0: continue
        bb = b[m]; vv = v[m]
        share = 100 * bb.imp_px.sum() / max(1, bb.imp_px.sum() + bb.imp_plain.sum()); take = 100 * bb.px_take.sum() / bb.px_calls.sum()
        L.append(f"{lab} & {len(bb)} & {vv.mgap.mean():.4f} & {bb.mgap.mean():.4f} & {(bb.iters / vv.iters).mean():.2f} & {take:.1f} & {bb.comps.sum() / bb.px_calls.sum():.1f} & {bb.dsum.sum() / bb.px_calls.sum():.1f} & {share:.1f} \\\\")
        N('Share' + key, share, '{:.1f}'); N('Take' + key, take, '{:.1f}'); N('Comps' + key, bb.comps.sum() / bb.px_calls.sum(), '{:.1f}'); N('Gapvns' + key, vv.mgap.mean(), '{:.4f}'); N('Gapbpx' + key, bb.mgap.mean(), '{:.4f}')
        N('IterRatio' + key, (bb.iters / vv.iters).mean(), '{:.2f}'); N('NPairs' + key, len(bb), '{:d}')
    L += [r'\bottomrule', r'\end{tabular}']
    open(f'{OUT}/tables/tab_mech.tex', 'w').write('\n'.join(L))
    N('IterRatio', (b.iters / v.iters).mean(), '{:.2f}')
    tt = sel('BPX', TEST); tv = sel('VNS', TEST).loc[tt.index]; big = tt.index.get_level_values(1) >= 100
    N('TbigPairs', int(big.sum()), '{:d}'); N('TbigvnsGap', tv.mgap[big].mean(), '{:.4f}'); N('TbigbpxGap', tt.mgap[big].mean(), '{:.4f}')
    N('TbigvnsHit', 100 * tv.hits[big].sum() / tv.R[big].sum(), '{:.1f}'); N('TbigbpxHit', 100 * tt.hits[big].sum() / tt.R[big].sum(), '{:.1f}')
    N('TsmallPairs', int((~big).sum()), '{:d}'); N('TsmallvnsHit', 100 * tv.hits[~big].sum() / tv.R[~big].sum(), '{:.1f}'); N('TsmallbpxHit', 100 * tt.hits[~big].sum() / tt.R[~big].sum(), '{:.1f}')
    for ins in ('rl5934', 'fl3795'):
        if ins in tt.index.get_level_values(0):
            N(ins[:2].capitalize() + 'vnsGap', tv.loc[ins].mgap.mean(), '{:.4f}'); N(ins[:2].capitalize() + 'bpxGap', tt.loc[ins].mgap.mean(), '{:.4f}')

# ------------------------------------------------------------ OR-Library by class + earlier study
if have('VNS', 'BPX') and len(sel('BPX', ORL)) > 0:
    CL = {}; i = 1
    for _n, _k in ((100, 5), (200, 5), (300, 5), (400, 5), (500, 5), (600, 5), (700, 4), (800, 3), (900, 3)):
        for j in range(_k): CL[f'pmed{i}'] = ['$p=5$', '$p=10$', r'$p\approx n/10$', r'$p\approx n/5$', r'$p\approx n/3$'][j]; i += 1
    ov, ob = common('VNS', 'BPX', ORL)
    old = {}
    pth = os.path.join(REFDIR, 'earlier_study_orlib_runs.csv')
    if os.path.exists(pth):
        dd = pd.read_csv(pth); dd['gap'] = 100 * (dd.best - dd.opt) / dd.opt
        for a, nm in (('HGA', 'HGA-EBMKI'), ('OLDVNS', 'VNS-Whitaker')):
            g = dd[dd.algo == nm].groupby('inst')
            old[a] = pd.DataFrame({'mgap': g.gap.mean(), 'hits': g.gap.apply(lambda x: (x == 0).sum()), 'R': g.gap.count()})
    L = [r'\begin{tabular}{lr' + 'rr' * (2 + len(old)) + '}', r'\toprule',
         'Design class & Inst.' + (r' & \multicolumn{2}{c}{HGA-EBMKI$^{a}$}' if 'HGA' in old else '') + (r' & \multicolumn{2}{c}{VNS$^{a}$}' if 'OLDVNS' in old else '') + r' & \multicolumn{2}{c}{VNS} & \multicolumn{2}{c}{MK-BPX} \\',
         ' & ' + ' & Gap & Opt.\\ (\\%)' * (2 + len(old)) + r' \\', r'\midrule']
    cls = pd.Series([CL[i] for i in ov.index.get_level_values(0)], index=ov.index)
    def rowfor(mask, lab):
        r = f'{lab} & {int(mask.sum())}'
        for a in ('HGA', 'OLDVNS'):
            if a in old:
                t = old[a].loc[[i for i in ov.index.get_level_values(0)[mask]]]; r += f" & {t.mgap.mean():.4f} & {100 * t.hits.sum() / t.R.sum():.1f}"
        for t in (ov[mask], ob[mask]): r += f" & {t.mgap.mean():.4f} & {100 * t.hits.sum() / t.R.sum():.1f}"
        return r + r' \\'
    for c in ['$p=5$', '$p=10$', r'$p\approx n/10$', r'$p\approx n/5$', r'$p\approx n/3$']: L.append(rowfor((cls == c).values, c))
    L += [r'\midrule', rowfor(np.ones(len(ov), bool), 'All'), r'\bottomrule', r'\end{tabular}']
    open(f'{OUT}/tables/tab_orlib.tex', 'w').write('\n'.join(L))
    for a, key in (('HGA', 'Oldhga'), ('OLDVNS', 'Oldvns')):
        if a in old: N(key + 'Gap', old[a].mgap.mean()); N(key + 'Hit', 100 * old[a].hits.sum() / old[a].R.sum(), '{:.1f}')
    hard = (cls.isin([r'$p\approx n/5$', r'$p\approx n/3$'])).values
    N('OhardvnsHit', 100 * ov.hits[hard].sum() / ov.R[hard].sum(), '{:.1f}'); N('OhardbpxHit', 100 * ob.hits[hard].sum() / ob.R[hard].sum(), '{:.1f}'); N('OhardPairs', int(hard.sum()), '{:d}')
    rb = runs['BPX'][runs['BPX'].inst.isin(ORL)]; rv = runs['VNS'][runs['VNS'].inst.isin(ORL)]
    N('OvnsHits', int(ov.hits.sum()), '{:d}'); N('ObpxHits', int(ob.hits.sum()), '{:d}')
    N('ObpxMinHits', int(ob.hits.min()), '{:d}'); N('OvnsMinHits', int(ov.hits.min()), '{:d}')
    N('ObpxWorst', rb.gap.max()); N('OvnsWorst', rv.gap.max())

N('NImproved', len(IMPROVED), '{:d}')
tsp = ref[[not i.startswith('pmed') for i in ref.index.get_level_values(0)]]
N('NRefReached', int((tsp.ours == tsp.pub).sum()), '{:d}'); N('NTspPairs', len(tsp), '{:d}')
if len(IMPROVED):
    who = {}
    for (i, p), r in IMPROVED.iterrows():
        who[(i, p)] = [LAB[a] for a in ALG if a in runs and ((runs[a].inst == i) & (runs[a].p == p) & (runs[a].best < r.pub)).any()]
    N('RefSentence', 'For ' + '; '.join(f"{i} with $p={p}$ our runs found a better value than the published one ({int(r.ours)} against {int(r.pub)}; algorithms that improved on the published value: {', '.join(who[(i, p)])})" for (i, p), r in IMPROVED.iterrows()) + ', and the better value is used as reference.')
else:
    N('RefSentence', 'No run found a value better than the published one.')

# ------------------------------------------------------------ long runs (five times the time limit)
if 'VNS' in long and 'BPX' in long:
    def lt(d):
        g = d.groupby(['inst', 'p'], sort=False); t = pd.DataFrame({'R': g.gap.count(), 'mgap': g.gap.mean(), 'hits': g.gap.apply(lambda x: int((x == 0).sum()))}); return t[t.R == 5]
    lv, lb = lt(long['VNS']), lt(long['BPX']); idx = lv.index.intersection(lb.index); lv, lb = lv.loc[idx], lb.loc[idx]
    sv, sb = PT['VNS'].loc[idx], PT['BPX'].loc[idx]
    N('LPairs', len(idx), '{:d}'); N('LvnsGap', lv.mgap.mean(), '{:.4f}'); N('LbpxGap', lb.mgap.mean(), '{:.4f}'); N('LsvnsGap', sv.mgap.mean(), '{:.4f}'); N('LsbpxGap', sb.mgap.mean(), '{:.4f}')
    N('LvnsHit', 100 * lv.hits.sum() / lv.R.sum(), '{:.1f}'); N('LbpxHit', 100 * lb.hits.sum() / lb.R.sum(), '{:.1f}'); N('LsvnsHit', 100 * sv.hits.sum() / sv.R.sum(), '{:.1f}'); N('LsbpxHit', 100 * sb.hits.sum() / sb.R.sum(), '{:.1f}')
    def wx(d):
        nz = d[d != 0]; ex = len(np.unique(np.abs(np.round(nz, 9)))) == len(nz); EXACT.append(ex); return stats.wilcoxon(nz, method='exact' if ex else 'approx').pvalue
    d = lv.mgap.values - lb.mgap.values; N('LW', int((d > 0).sum()), '{:d}'); N('LL', int((d < 0).sum()), '{:d}'); N('LT', int((d == 0).sum()), '{:d}'); N('LP', fP(wx(d)))
    d2 = lv.mgap.values - sb.mgap.values; N('XW', int((d2 > 0).sum()), '{:d}'); N('XL', int((d2 < 0).sum()), '{:d}'); N('XT', int((d2 == 0).sum()), '{:d}'); N('XP', fP(wx(d2)))
    N('LRatio', lv.mgap.mean() / lb.mgap.mean(), '{:.1f}'); N('LsRatio', sv.mgap.mean() / sb.mgap.mean(), '{:.1f}')
    N('LLostList', ', '.join(f'{i} with $p={p}$' for (i, p) in idx[d < 0])); N('XLostList', ', '.join(f'{i} with $p={p}$' for (i, p) in idx[d2 < 0]))
    L = [r'\begin{tabular}{lrrrrrrrr}', r'\toprule', r' & \multicolumn{4}{c}{Time limit of Table~\ref{tab:instances}} & \multicolumn{4}{c}{Five times that limit} \\', r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}',
         r' & \multicolumn{2}{c}{VNS} & \multicolumn{2}{c}{MK-BPX} & \multicolumn{2}{c}{VNS} & \multicolumn{2}{c}{MK-BPX} \\', r'Instance & Gap & Opt.\ (\%) & Gap & Opt.\ (\%) & Gap & Opt.\ (\%) & Gap & Opt.\ (\%) \\', r'\midrule']
    for ins in ['fl1400', 'u1432', 'd2103']:
        if ins not in idx.get_level_values(0): continue
        r = ins
        for t in (sv, sb, lv, lb): r += f" & {t.loc[ins].mgap.mean():.4f} & {100 * t.loc[ins].hits.sum() / t.loc[ins].R.sum():.0f}"
        L.append(r + r' \\')
    r = r'\midrule' + '\nAll'
    for t in (sv, sb, lv, lb): r += f" & {t.mgap.mean():.4f} & {100 * t.hits.sum() / t.R.sum():.0f}"
    L += [r + r' \\', r'\bottomrule', r'\end{tabular}']
    open(f'{OUT}/tables/tab_long.tex', 'w').write('\n'.join(L))
if os.path.exists(os.path.join(RES, 'verify.json')):
    v_ = json.load(open(os.path.join(RES, 'verify.json')))
    N('VerPairs', v_['pairs'], '{:d}'); N('VerSubsets', '{:,}'.format(v_['subsets']).replace(',', '{,}')); N('VerBalanced', '{:,}'.format(v_['balanced']).replace(',', '{,}')); N('VerStrict', v_['strict'], '{:d}'); N('VerViol', v_['bound_viol'] + v_['dp_mismatch'], '{:d}'); N('VerTight', v_['tight'], '{:d}')
if os.path.exists(os.path.join(RES, 'speed_benchmark.csv')):
    sp = pd.read_csv(os.path.join(RES, 'speed_benchmark.csv')); assert sp.both_local_opt.all()
    L = [r'\begin{tabular}{lrrr@{\hskip 18pt}lrrr}', r'\toprule', r'Instance & $n$ & $p$ & Ratio & Instance & $n$ & $p$ & Ratio \\', r'\midrule']
    rows = [f"{r.inst} & {r.n} & {r.p} & {r.ratio:.1f}" for r in sp.itertuples()]
    h = (len(rows) + 1) // 2
    for j in range(h): L.append(rows[j] + ' & ' + (rows[j + h] if j + h < len(rows) else ' & & & ') + r' \\')
    L += [r'\bottomrule', r'\end{tabular}']
    open(f'{OUT}/tables/tab_speed.tex', 'w').write('\n'.join(L))
    big = sp[sp.p >= 50]; N('SpMin', sp.ratio.min(), '{:.1f}'); N('SpMax', sp.ratio.max(), '{:.0f}'); N('SpBigMin', big.ratio.min(), '{:.1f}'); N('SpBigMax', big.ratio.max(), '{:.0f}'); N('SpCases', len(sp), '{:d}')
# ------------------------------------------------------------ example numbers
if os.path.exists(os.path.join(RES, 'example.json')):
    e = json.load(open(os.path.join(RES, 'example.json')))
    N('ExZS', int(e['zS']), '{:d}'); N('ExZT', int(e['zT']), '{:d}'); N('ExQ', e['q'], '{:d}'); N('ExNC', e['nc'], '{:d}'); N('ExGain', f"${int(e['bound'] - e['zS'])}$"); N('ExZC', int(e['zc']), '{:d}')

# ------------------------------------------------------------ figures
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
if have('VNS', 'BPX'):
    x, y = common('VNS', 'BPX', TEST + DEV)
    order = [(i, p) for i in TEST + DEV for p in sorted(set(x.index.get_level_values(1))) if (i, p) in x.index]
    fig, ax = plt.subplots(figsize=(7.4, 3.4)); FL = 2e-4
    pos = 0; ticks = []; labs = []; bounds = []
    for ins in TEST + DEV:
        ps_ = [p for (i, p) in order if i == ins]
        if not ps_: continue
        start = pos
        for p in ps_:
            a, b_ = max(x.mgap[(ins, p)], FL), max(y.mgap[(ins, p)], FL)
            ax.plot([pos, pos], [a, b_], color='0.6', lw=0.8, zorder=1)
            ax.plot(pos, a, 'D', color=COL['VNS'], ms=3.6, zorder=2); ax.plot(pos, b_, 's', color=COL['BPX'], ms=3.6, zorder=3)
            pos += 1
        ticks.append((start + pos - 1) / 2); labs.append(ins + ('*' if ins in DEV else '')); bounds.append(pos - 0.5); pos += 1
    for bnd in bounds[:-1]: ax.axvline(bnd + 0.5, color='0.9', lw=0.8)
    ax.set_yscale('log'); ax.set_ylim(FL * 0.7, 1); ax.set_yticks([FL, 1e-3, 1e-2, 1e-1, 1]); ax.set_yticklabels(['0', '0.001', '0.01', '0.1', '1'])
    ax.set_xticks(ticks); ax.set_xticklabels(labs, fontsize=8); ax.set_ylabel('Mean gap to the reference value (%)')
    ax.plot([], [], 'D', color=COL['VNS'], label='VNS'); ax.plot([], [], 's', color=COL['BPX'], label='MK-BPX'); ax.legend(frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=8)
    fig.tight_layout(); fig.savefig(f'{OUT}/images/fig_pairs.pdf'); plt.close(fig)

def value_at(tr, t):
    b = tr[0][1]
    for a, v in tr:
        if a <= t: b = v
        else: break
    return b
if have('VNS', 'BPX'):
    cand = [('u1432', 300), ('d2103', 400), ('fl3795', 300), ('rl5934', 400), ('fl1400', 200), ('d2103', 200)]
    cand = [('d2103', 300), ('d2103', 500), ('fl3795', 300), ('rl5934', 400), ('d2103', 200), ('fl3795', 500), ('rl5934', 200), ('d2103', 100)]
    ok = lambda c, a: a in trace and all(f'{c[0]}_{c[1]}_{r}' in trace[a] for r in range(int(PT[a].R[c]))) if (a in PT and c in PT[a].index) else False
    pick = [c for c in cand if ok(c, 'VNS') and ok(c, 'BPX')][:4]
    if pick:
        fig, axs = plt.subplots(1, len(pick), figsize=(7.4, 2.7))
        for ax, (ins, p) in zip(np.atleast_1d(axs), pick):
            tl = PT['BPX'].tl[(ins, p)]; opt = PT['BPX'].opt[(ins, p)]; ts = np.linspace(tl * 0.05, tl, 60)
            for a in ['VNS', 'VNS20', 'BPX4', 'BPX']:
                if not ok((ins, p), a): continue
                R = int(PT[a].R[(ins, p)]); Y = []
                for t in ts:
                    Y.append(np.mean([100 * (value_at(trace[a][f'{ins}_{p}_{r}'], t) - opt) / opt for r in range(R)]))
                ax.plot(ts, np.maximum(Y, 1e-4), color=COL[a], label=LAB[a], lw=1.5 if a in ('VNS', 'BPX') else 1.0, ls='-' if a in ('VNS', 'BPX') else '--')
            ax.set_yscale('log'); ax.set_title(f'{ins}, $p={p}$', fontsize=8); ax.set_xlabel('Time (s)')
            import matplotlib.ticker as mt
            ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f'{v:g}')); ax.yaxis.set_minor_formatter(mt.NullFormatter())
        np.atleast_1d(axs)[0].set_ylabel('Mean gap (%)')
        hs, ls_ = np.atleast_1d(axs)[0].get_legend_handles_labels()
        fig.subplots_adjust(left=0.09, right=0.99, top=0.90, bottom=0.30, wspace=0.42)
        fig.legend(hs, ls_, fontsize=7.5, frameon=False, ncol=4, loc='lower center', bbox_to_anchor=(0.5, 0.0))
        fig.savefig(f'{OUT}/images/fig_anytime.pdf'); plt.close(fig)
N('AllExact', 'yes' if all(EXACT) else 'no'); print('all Wilcoxon exact:', all(EXACT), EXACT)
with open(f'{OUT}/numbers.tex', 'w') as f:
    for k_, v_ in sorted(num.items()):
        assert k_.isalpha(), k_
        f.write('\\newcommand{\\m%s}{%s}\n' % (k_, v_))
json.dump(num, open(f'{OUT}/numbers.json', 'w'), indent=1)
print('OK', len(num))
