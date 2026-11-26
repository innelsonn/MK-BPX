"""Instances and reference values.

OR-Library pmed1-pmed40: shortest-path distance matrices stored in data/orlib/pmed_distances.npz
(built from the original files by tools/build_orlib_matrices.py).
TSPLIB: Euclidean distance rounded down to an integer (plain floor in double precision), the
convention of Duran-Mateluna, Ales & Elloumi (2023), whose optimal / best-known values are in TSP.

load_matrix(name) -> distance matrix (float64, C-contiguous);  opt(name, p) -> reference value."""
import os, numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get('MKBPX_DATA', os.path.join(_ROOT, 'data'))
ORLIB = os.path.join(DATA, 'orlib', 'pmed_distances.npz')
TSPDIR = os.path.join(DATA, 'tsplib')
# OR-Library instances: number -> (n, p, optimal value)
PMED = {1:(100,5,5819),2:(100,10,4093),3:(100,10,4250),4:(100,20,3034),5:(100,33,1355),6:(200,5,7824),7:(200,10,5631),8:(200,20,4445),9:(200,40,2734),10:(200,67,1255),11:(300,5,7696),12:(300,10,6634),13:(300,30,4374),14:(300,60,2968),15:(300,100,1729),16:(400,5,8162),17:(400,10,6999),18:(400,40,4809),19:(400,80,2845),20:(400,133,1789),21:(500,5,9138),22:(500,10,8579),23:(500,50,4619),24:(500,100,2961),25:(500,167,1828),26:(600,5,9917),27:(600,10,8307),28:(600,60,4498),29:(600,120,3033),30:(600,200,1989),31:(700,5,10086),32:(700,10,9297),33:(700,70,4700),34:(700,140,3013),35:(800,5,10400),36:(800,10,9934),37:(800,80,5057),38:(900,5,11060),39:(900,10,9423),40:(900,90,5128)}
# optimal values (Duran-Mateluna, Ales & Elloumi, 2023, Tables 3-4)
TSP = {
 'rl1304': {5:3099073,10:2134295,20:1412108,50:795012,100:491639,200:268573,300:177326,400:128332,500:97024},
 'fl1400': {5:174877,10:100601,20:57191,50:28486,100:15962,200:8806,300:6109,400:4648,500:3764},
 'u1432': {5:1210126,10:849759,20:588766,50:362072,100:243793,200:159887,300:123689,400:103979,500:93200},
 'd2103': {5:1005136,10:687321,20:482926,50:302221,100:194664,200:117753,300:90471,400:75324,500:64006},
 'pcb3038': {5:1777835,10:1211704,20:839494,50:506339,100:351500,150:280128,200:237399,300:186833,400:156276,500:134798},
 'fl3795': {5:1052627,10:520940,20:319722,50:150940,100:88299,150:65868,200:53928,300:39586,400:31354,500:25976},
 'rl5934': {10:9792218,20:6716215,50:4030007,200:1805530,300:1392419,400:1143940,500:972799},
}
_c = {}
def load_matrix(name):
    if name in _c: return _c[name]
    if name.startswith('pmed'):
        with np.load(ORLIB) as z:
            C = np.ascontiguousarray(z[name].astype(np.float64))
    else:
        xy = []; go = False
        for ln in open(os.path.join(TSPDIR, name + '.tsp')):
            ln = ln.strip()
            if ln.startswith('NODE_COORD_SECTION'): go = True; continue
            if ln == 'EOF' or not ln: continue
            if go:
                a = ln.split(); xy.append((float(a[1]), float(a[2])))
        xy = np.array(xy); d = xy[:, None, :] - xy[None, :, :]
        C = np.floor(np.sqrt((d ** 2).sum(-1)))   # plain floor in double precision, the convention of the reference values
        C = np.ascontiguousarray(C)
    _c.clear(); _c[name] = C
    return C
def opt(name, p):
    if name.startswith('pmed'): return PMED[int(name[4:])][2]
    return TSP[name][p]
def pmed_p(name): return PMED[int(name[4:])][1]
def read_coords(name):
    """Coordinates of a TSPLIB instance (n x 2 array)."""
    xy = []; go = False
    for ln in open(os.path.join(TSPDIR, name + '.tsp')):
        ln = ln.strip()
        if ln.startswith('NODE_COORD_SECTION'): go = True; continue
        if go and ln and ln != 'EOF':
            a = ln.split(); xy.append((float(a[1]), float(a[2])))
    return np.array(xy)
