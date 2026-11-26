"""Driver for the original POPSTAR program of Resende & Werneck (2004).

The program itself is not part of this repository: tools/get_popstar.sh downloads the authors'
distribution (https://mauricio.resende.info/popstar/downloads.html) and compiles it unchanged.
This module only launches the executable and reads what it prints.

What is measured
    POPSTAR reports `cputime`, the user CPU time of the search (multistart phase, post-optimization
    and the initialization of its distance oracle; reading the instance is excluded).  A user-time
    clock cannot be compared with the elapsed-time clock of MK-BPX, so the driver also measures
    elapsed time: POPSTAR writes the line "Fixed:..." to stderr (unbuffered) as soon as the
    instance has been read and its oracle initialized, and the elapsed search time of a run is

        wall = (time at which the process ends) - (time of the "Fixed:" line) + oracletime

    i.e. it covers exactly what `cputime` covers.

Improvement trace
    In its first cycle POPSTAR writes one line to stderr per multistart iteration and one per
    path-relinking of the post-optimization phase, each with the value of the solution obtained.
    The driver stamps every line with the elapsed search time, which gives the time and value of
    every improvement of the best solution without touching the program.

Memory allocator
    POPSTAR allocates and frees its auxiliary matrices in every local search.  With the default
    settings of the GNU C library these blocks are returned to the operating system and mapped
    again each time, which on the instances with more than 3000 nodes added as much kernel time as
    the search itself.  The driver therefore sets GLIBC_TUNABLES so that freed memory stays in the
    process (MALLOC_ENV below).  This changes nothing in the algorithm and only makes POPSTAR faster.
"""
import os, re, subprocess, time, tempfile
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BIN = os.path.join(ROOT, 'third_party', 'popstar', 'popstar')
MALLOC_ENV = 'glibc.malloc.mmap_max=0:glibc.malloc.trim_threshold=1099511627776:glibc.malloc.top_pad=268435456'

_IT = re.compile(r'^\s*(\d+): \[\S+\] (\S+) -> (\S+)(.*)$')          # multistart iteration
_PO = re.compile(r'^[> ] \(\d+:\S+\) x \(\d+:\S+\) \[\d+\]: (.*)$')    # post-optimization relink
_NUM = re.compile(r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?')


def binary():
    b = os.environ.get('POPSTAR_BIN', DEFAULT_BIN)
    if not os.path.exists(b):
        raise FileNotFoundError(f'POPSTAR executable not found at {b}; run tools/get_popstar.sh or set POPSTAR_BIN')
    return b


def run(pmm, p, seed, graspit=32, elite=10, core=None, tuned_malloc=True, keep_solution=True, extra=()):
    """One run of POPSTAR.  Returns a dict with
        best        value reported by the program (bestsol)
        wall        elapsed search time (see module docstring), seconds
        cpu         `cputime` reported by the program (user CPU time of the search)
        init        `inittime` (user time spent reading the instance, not part of wall or cpu)
        oracle, prtime, generations, grasp (value before post-optimization)
        user, sys   CPU times of the whole process, including reading the instance
        wall_total  elapsed time of the whole process
        trace       [[elapsed search time, value], ...] for every improvement of the best solution
        ms_iters    multistart iterations seen on stderr, po_relinks: relinks of the post-optimization
        ms_end      elapsed search time at the end of the multistart phase
        solution    sorted 0-based indices of the open facilities (if keep_solution)
    """
    env = dict(os.environ)
    if tuned_malloc:
        env['GLIBC_TUNABLES'] = MALLOC_ENV
    cmd = [binary(), pmm, '-p', str(p), '-seed', str(seed), '-graspit', str(graspit), '-elite', str(elite)] + list(extra)
    solfile = None
    if keep_solution:
        fd, solfile = tempfile.mkstemp(suffix='.sol'); os.close(fd)
        cmd += ['-output', solfile]
    pre = (lambda: os.sched_setaffinity(0, {core})) if core is not None else None
    r0 = os.times()
    t0 = time.perf_counter()
    pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, preexec_fn=pre, text=True, bufsize=1)
    t_ready = None; best = float('inf'); trace = []; ms = 0; po = 0; ms_end = None
    for line in pr.stderr:
        t = time.perf_counter()
        if t_ready is None:
            if line.startswith('Fixed:'):
                t_ready = t
            continue
        m = _IT.match(line)
        if m:
            ms += 1
            nums = _NUM.findall(m.group(4))
            v = min(float(m.group(3)), float(nums[-1])) if nums else float(m.group(3))
            ms_end = t - t_ready
        else:
            m = _PO.match(line)
            if not m:
                continue
            po += 1
            v = float(_NUM.findall(m.group(1))[-1])
        if v < best - 1e-9:
            best = v; trace.append([t - t_ready, v])
    out = pr.stdout.read()
    pr.wait()
    t1 = time.perf_counter(); r1 = os.times()
    if pr.returncode != 0 or t_ready is None:
        raise RuntimeError(f'POPSTAR failed ({pr.returncode}): {" ".join(cmd)}')
    kv = {}
    for ln in out.splitlines():
        a = ln.split(None, 1)
        if len(a) == 2:
            kv.setdefault(a[0], a[1])
    oracle = float(kv.get('oracletime', 0.0))
    res = dict(best=float(kv['bestsol']), wall=t1 - t_ready + oracle, cpu=float(kv['cputime']), init=float(kv['inittime']),
               oracle=oracle, prtime=float(kv.get('prtime', 0.0)), generations=int(kv.get('generations', 0)),
               grasp=float(kv.get('grasp', 'nan')), user=r1.children_user - r0.children_user, sys=r1.children_system - r0.children_system,
               wall_total=t1 - t0, ms_iters=ms, po_relinks=po, ms_end=(ms_end + oracle) if ms_end is not None else float('nan'),
               trace=[[a + oracle, b] for a, b in trace], version=kv.get('version', ''))
    if keep_solution:
        fac = [int(ln.split()[1]) - 1 for ln in open(solfile) if ln.startswith('f ')]
        os.unlink(solfile)
        res['solution'] = np.array(sorted(fac), dtype=np.int64)
    return res
