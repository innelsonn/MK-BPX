"""Export an instance of the study to the PMM format of POPSTAR (full distance matrix).

Usage:  python tools/export_pmm.py NAME [NAME ...] --out DIR

NAME is pmed1..pmed40 or one of the TSPLIB instances of data/tsplib. The matrix written is
exactly the one used by MK-BPX (shortest-path distances for OR-Library, Euclidean distance
rounded down for TSPLIB), so that both programs minimise the same objective function.
Format (https://mauricio.resende.info/popstar/input.html):
    p <n> <m>
    a <user> <facility> <distance>      (n*m lines, labels from 1)
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from mkbpx import instances as inst


def export(name, outdir):
    C = inst.load_matrix(name)
    n = C.shape[0]
    assert np.array_equal(C, np.floor(C)), 'non-integer distances'
    path = os.path.join(outdir, name + '.pmm')
    with open(path, 'w') as f:
        f.write(f'p {n} {n}\n')
        fac = np.arange(1, n + 1, dtype=np.int32)
        for lo in range(0, n, 256):                       # chunks of users, to bound memory
            hi = min(n, lo + 256)
            u = np.repeat(np.arange(lo + 1, hi + 1, dtype=np.int32), n)
            df = pd.DataFrame({'a': 'a', 'u': u, 'f': np.tile(fac, hi - lo), 'd': C[lo:hi].astype(np.int64).ravel()})
            df.to_csv(f, sep=' ', header=False, index=False)
    return path


if __name__ == '__main__':
    argv = sys.argv[1:]
    out = '.'
    if '--out' in argv:
        k = argv.index('--out'); out = argv[k + 1]; del argv[k:k + 2]
    if not argv:
        print(__doc__); sys.exit(1)
    os.makedirs(out, exist_ok=True)
    for nm in argv:
        print(export(nm, out), flush=True)
