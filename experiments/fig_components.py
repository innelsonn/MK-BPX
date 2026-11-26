"""Figure 1 of the paper: components of BPX on a real pair of local optima (rl1304, p = 100).

Usage:  python experiments/fig_components.py [SEED] [plot] [DEST]
The paper uses  python experiments/fig_components.py 9 plot  (writes results/example.json and
DEST/images/fig_components.pdf, DEST = paper/ by default). Without "plot" it only prints the
components found for that seed."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import json, numpy as np
from numba import njit
from mkbpx import core as K, instances as inst
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
name, p, kk = 'rl1304', 100, 20
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
PLOT = len(sys.argv) > 2
C = inst.load_matrix(name); order = K.build_order(C); n = C.shape[0]
xy = inst.read_coords(name)
cnt = np.zeros(4, np.int64)
@njit
def make(C, order, p, kk, seed, cnt):
    np.random.seed(seed); m = C.shape[0]
    perm = np.arange(m).astype(np.int64); pp = np.empty(p, np.int64); pm = np.empty(m, np.int64)
    s = K.random_subset(m, p, perm)
    S, zS, nS, dS = K.ls_full(s, C, order, 10**8, cnt)
    # a few accepted kicks so that S is a good local optimum
    for it in range(300):
        c = K.kick(S, m, 1 + it % 3, pp, pm); S2, z2, n2, d2 = K.ls_full(c, C, order, 10**8, cnt)
        if z2 < zS: S, zS, nS, dS = S2, z2, n2, d2
    c = K.kick(S, m, kk, pp, pm); T, zT, nT, dT = K.ls_full(c, C, order, 10**8, cnt)
    par = np.empty(m, np.int64); cid = np.empty(m, np.int64); tag = np.zeros(m, np.int8); info = np.zeros(4)
    ch = K.px(S, zS, nS, dS, T, zT, nT, dT, m, par, tag, cid, info, 0)
    ch2, zc, nc_, dc_ = K.ls_full(ch, C, order, 10**8, cnt)
    return S, zS, nS, dS, T, zT, nT, dT, ch, info, zc
S, zS, nS, dS, T, zT, nT, dT, ch, info, zc = make(C, order, p, kk, seed, cnt)
setS, setT = set(S.tolist()), set(T.tolist()); D = sorted(setS - setT); A = sorted(setT - setS)
par = {x: x for x in D + A}
def find(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
for i in range(n):
    s, t = nS[i], nT[i]
    if s in par and t in par and s not in setT and t not in setS: par[find(s)] = find(t)
comp = {}
for x in D + A: comp.setdefault(find(x), {'D': [], 'A': [], 'delta': 0.0})['D' if x in setS else 'A'].append(x)
for i in range(n):
    s, t = nS[i], nT[i]
    if s == t: continue
    if s in par and s not in setT: comp[find(s)]['delta'] += dT[i] - dS[i]
    elif t in par and t not in setS: comp[find(t)]['delta'] += dT[i] - dS[i]
comps = list(comp.values())
print('zS', zS, 'zT', zT, 'q', len(D), 'components', len(comps), 'info', info, 'child after LS', zc)
print([(len(c['D']), len(c['A']), c['delta']) for c in comps])
import itertools
bestv, bestsub = 0.0, ()
for r in range(1, len(comps) + 1):
    for sub in itertools.combinations(range(len(comps)), r):
        if sum(len(comps[j]['A']) - len(comps[j]['D']) for j in sub) == 0:
            v = sum(comps[j]['delta'] for j in sub)
            if v < bestv - 1e-9: bestv, bestsub = v, sub
for j, c in enumerate(comps): c['take'] = j in bestsub
assert abs(zS + bestv - info[2]) < 1e-6
if not PLOT: sys.exit()
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.scatter(xy[:, 0], xy[:, 1], s=1.5, color='0.78', zorder=1)
common = sorted(setS & setT)
ax.scatter(xy[common, 0], xy[common, 1], s=14, color='0.35', marker='s', zorder=2, label='Common facilities')
first = {'imp': True, 'wor': True}
for jc, c in enumerate(sorted(comps, key=lambda c: np.mean(xy[c['D'] + c['A']][:, 0]))):
    good = c['take']; col = '#1a7f37' if good else '#c8553d'
    pts = c['D'] + c['A']
    for d in c['D']:
        for a in c['A']:
            ax.plot([xy[d, 0], xy[a, 0]], [xy[d, 1], xy[a, 1]], color=col, lw=0.9, zorder=3)
    ax.scatter(xy[c['D'], 0], xy[c['D'], 1], s=36, facecolors='white', edgecolors=col, linewidths=1.4, zorder=4)
    ax.scatter(xy[c['A'], 0], xy[c['A'], 1], s=36, color=col, zorder=4)
    cx, cy = xy[pts].mean(0)
    ax.annotate(f"$\\Delta={c['delta']:+.0f}$\n$|D_c|={len(c['D'])}$, $|A_c|={len(c['A'])}$", (cx, cy), textcoords='offset points', xytext=[(-100, 14), (12, -30), (14, 12), (14, 12)][min(jc, 3)], fontsize=7, bbox=dict(fc='white', ec='none', alpha=0.85, pad=1), color=col, zorder=5)
import matplotlib.lines as ml
h = [ml.Line2D([], [], marker='s', ls='', color='0.35', ms=5, label='Facilities common to $S$ and $T$'),
     ml.Line2D([], [], marker='o', ls='', mfc='white', mec='k', ms=6, label='Only in $S$ (set $D$)'),
     ml.Line2D([], [], marker='o', ls='', color='k', ms=6, label='Only in $T$ (set $A$)'),
     ml.Line2D([], [], color='#1a7f37', lw=1.5, label='Component taken from $T$'),
     ml.Line2D([], [], color='#c8553d', lw=1.5, label='Component kept from $S$')]
ax.legend(handles=h, fontsize=7, frameon=True, loc='upper left', bbox_to_anchor=(1.01, 1.0))
ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_visible(False)
DEST = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ROOT, 'paper')
os.makedirs(os.path.join(DEST, 'images'), exist_ok=True)
fig.tight_layout(); fig.savefig(os.path.join(DEST, 'images', 'fig_components.pdf'))
json.dump({'seed': seed, 'zS': float(zS), 'zT': float(zT), 'q': len(D), 'nc': len(comps), 'taken': int(info[1]), 'bound': float(info[2]), 'zc': float(zc), 'nimp': int(sum(c['delta'] < 0 for c in comps)),
           'sumneg': float(sum(c['delta'] for c in comps if c['delta'] < 0)), 'sumpos': float(sum(c['delta'] for c in comps if c['delta'] >= 0))}, open(os.path.join(ROOT, 'results', 'example.json'), 'w'))
