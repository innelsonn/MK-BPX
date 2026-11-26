"""Rebuilds data/orlib/pmed_distances.npz from the original OR-Library files in data/orlib_raw.

Each file: first line "n edges p"; then one edge "i j cost" per line (1-based). As prescribed by the
library, a repeated edge keeps its last cost; distances are all-pairs shortest paths (Floyd-Warshall).

Usage:  python tools/build_orlib_matrices.py [--check]     (--check compares with the shipped file)
"""
import os, sys
import numpy as np
from scipy.sparse.csgraph import floyd_warshall
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = {}
for i in range(1, 41):
    tok = open(os.path.join(ROOT, 'data', 'orlib_raw', f'pmed{i}.txt')).read().split()
    n, m = int(tok[0]), int(tok[1]); W = np.full((n, n), np.inf); np.fill_diagonal(W, 0.0)
    for k in range(m):
        a, b, c = int(tok[3 + 3 * k]) - 1, int(tok[4 + 3 * k]) - 1, float(tok[5 + 3 * k])
        W[a, b] = c; W[b, a] = c
    D = floyd_warshall(W, directed=False); assert np.isfinite(D).all()
    out[f'pmed{i}'] = D.astype(np.int32)
dst = os.path.join(ROOT, 'data', 'orlib', 'pmed_distances.npz')
if '--check' in sys.argv:
    with np.load(dst) as z:
        bad = [k for k in out if not np.array_equal(out[k], z[k])]
    print('identical to the shipped matrices' if not bad else f'DIFFERENT: {bad}')
else:
    np.savez_compressed(dst, **out); print('written', dst)
