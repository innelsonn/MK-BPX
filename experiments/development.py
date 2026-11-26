"""Development runs (Table 3 of the paper): mean gap over R runs on the six development pairs.

Usage:  python experiments/development.py "CFG;CFG;..." R

CFG = use_px,neutral,kmax,N   for the final code (mkbpx.core.run_main)
      use_px  0 = VNS, 1 = MK-BPX
      neutral 1 = components with zero difference are also accepted
      kmax    <= 1: fraction of p;  > 1: absolute value
      N       number of copies of the elite solution
CFG = pop,N,G,fmode           for the prototype mkbpx.prototypes.run_pop (rows with two and four
                              elite solutions: pop,2,1,0 and pop,4,1,0; every five rounds: pop,2,5,1)

Rows of the table, R = 2:
  python experiments/development.py "0,0,1,1;0,0,20,1;1,0,1,1;1,0,0.2,1;1,0,20,1;1,1,1,1;pop,2,1,0;pop,4,1,0" 2
Seeds are 100, 101, ...; time limits are 10 s (rl1304) and 40 s (pcb3038).
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from mkbpx import core as K, instances as inst, prototypes as P
if len(sys.argv) < 3:
    print(__doc__); sys.exit(1)
cfgs = [tuple(c.split(',')) for c in sys.argv[1].split(';')]; R = int(sys.argv[2])
DEV = [('rl1304', 100, 10), ('rl1304', 300, 10), ('rl1304', 500, 10), ('pcb3038', 100, 40), ('pcb3038', 300, 40), ('pcb3038', 500, 40)]
res = {c: [] for c in cfgs}
for name, p, tl in DEV:
    C = inst.load_matrix(name); order = K.build_order(C); o = inst.opt(name, p)
    K.run_main(C, order, 5, 1, 1, 5, 2, 1, 0., 0.2); P.run_pop(C, order, 5, 2, 2, 0, 5, 1, 0., 0.2)   # compile
    for c in cfgs:
        g = []
        for s in range(R):
            if c[0] == 'pop':
                z = P.run_pop(C, order, p, int(c[1]), int(c[2]), int(c[3]), p, 100 + s, float(o), tl)[0]
            else:
                kf = float(c[2]); kmax = max(2, int(round(kf * p))) if kf <= 1 else int(kf)
                z = K.run_main(C, order, p, int(c[0]), int(c[1]), kmax, int(c[3]), 100 + s, float(o), tl)[0]
            g.append(100 * (z - o) / o)
        res[c].append(np.mean(g)); print(name, p, ','.join(c), '%.4f' % np.mean(g), flush=True)
for c in cfgs: print('CFG', ','.join(c), ' '.join('%.4f' % x for x in res[c]), '| mean %.4f' % np.mean(res[c]))
