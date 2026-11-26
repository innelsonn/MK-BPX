"""Runs the whole study: campaign (two processes side by side, as in the paper), verification,
speed benchmark, figure and analysis. About 6.5 hours on two cores.

Usage:  python experiments/run_all.py [--out DIR]
With the default DIR (results/) the shipped logs are complete, so the campaign steps are skipped;
give a new directory (e.g. --out results_new) to repeat the experiments from scratch.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import subprocess
out = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else os.path.join(ROOT, 'results')
py = sys.executable; E = os.path.join(ROOT, 'experiments')
def both(a, b, s):
    ps = [subprocess.Popen([py, os.path.join(E, 'campaign.py'), x, s, '--out', out]) for x in (a, b)]
    for q in ps:
        if q.wait() != 0: sys.exit('campaign failed')
for a, b, s in (('VNS', 'BPX', 'S1'), ('VNS', 'BPX', 'S2'), ('VNS', 'BPX', 'S0'), ('VNS20', 'BPX4', 'S1'), ('VNS', 'BPX', 'S3')):
    both(a, b, s)
if os.path.abspath(out) == os.path.join(ROOT, 'results'):
    for step in (['verify_bpx.py'], ['speed_benchmark.py'], ['fig_components.py', '9', 'plot']):
        subprocess.check_call([py, os.path.join(E, step[0])] + step[1:])
    subprocess.check_call([py, os.path.join(E, 'analyze.py')])
else:
    import shutil
    for f in ('verify.json', 'speed_benchmark.csv', 'example.json'):
        if not os.path.exists(os.path.join(out, f)): shutil.copy(os.path.join(ROOT, 'results', f), out)
    subprocess.check_call([py, os.path.join(E, 'analyze.py'), os.path.join(out, 'paper'), '--results', out])
