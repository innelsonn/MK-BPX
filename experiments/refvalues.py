"""Reference values shared by the three analyses.

The reference value f* of an instance-p pair is the published optimal or best-known value
(mkbpx/instances.py) or, if lower, the best value found in any run of the campaigns whose log
directories are given.  Every value found by a run was re-evaluated from scratch when it was
logged, so a value below the published one is a feasible solution, not an error.

reference(dirs)            -> Series indexed by (inst, p)
improved(old_dirs, new)    -> [(inst, p, old reference, new reference, who found it), ...]
"""
import os, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import pandas as pd
from mkbpx import instances as inst

LABEL = {'VNS': 'VNS', 'VNS20': 'VNS$_{20}$', 'BPX': 'MK-BPX', 'BPX4': 'MK-BPX$_{4}$', 'POPSTAR': 'POPSTAR'}


def _runs(d):
    """All runs logged in directory d: columns inst, p, best, algo."""
    out = []
    files = [f for pat in ('S?_*.csv', 'P32_S?.csv', 'PT_S?.csv') for f in glob.glob(os.path.join(d, pat))]      # run-level logs only
    for f in sorted(files):
        x = pd.read_csv(f)
        if 'inst' not in x or 'p' not in x or 'run' not in x or len(x) == 0:
            continue
        col = 'best' if 'best' in x else ('best_end' if 'best_end' in x else None)
        if col is None:
            continue
        y = x[['inst', 'p']].copy(); y['best'] = x[col].astype(float)
        y['algo'] = x['algo'] if 'algo' in x else os.path.basename(f)
        out.append(y)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=['inst', 'p', 'best', 'algo'])


def reference(dirs):
    r = pd.concat([_runs(d) for d in dirs if os.path.isdir(d)], ignore_index=True)
    best = r.groupby(['inst', 'p']).best.min()
    pub = pd.Series({k: float(inst.opt(*k)) for k in best.index})
    pub.index = pd.MultiIndex.from_tuples(pub.index, names=['inst', 'p'])
    return pd.concat([pub, best], axis=1).min(axis=1)


def improved(old_dirs, new_dirs):
    old = reference(old_dirs); r = pd.concat([_runs(d) for d in new_dirs if os.path.isdir(d)], ignore_index=True)
    out = []
    for (i, p), g in r.groupby(['inst', 'p']):
        o = old.get((i, p), float(inst.opt(i, p)))
        if g.best.min() < o:
            who = sorted({LABEL.get(a, a) for a in g[g.best < o].algo})
            out.append((i, int(p), int(o), int(g.best.min()), ' and '.join(who)))
    return out
