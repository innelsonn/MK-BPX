"""Complete the improvement traces of the main campaign by exact replay.

Usage:  python experiments/complete_traces.py [--main DIR] [--replay DIR]

The main campaign stored the time and value of every improvement of every run, except for the
runs of VNS and MK-BPX on fl1400, u1432 and rl1304 (270 runs of each), whose traces were not
saved.  A run is determined by its seed, so it can be repeated exactly: the replay campaign
(experiments/campaign_replay.py) re-ran every logged run with the same seed for at least the logged
number of iterations and recorded the iteration of every improvement.  This script

  1. checks, for every logged run, that the value after the logged number of iterations is the
     logged value (column replay_ok);
  2. checks, for every run whose trace was saved, that the saved sequence of values is the
     sequence of the replay;
  3. writes the missing traces into DIR/S1_VNS_trace.json and DIR/S1_BPX_trace.json.

The values of a reconstructed trace are exact.  Its times were measured on the machine of the
replay and are rescaled by (logged duration of the run) / (time the replay needed for
the same number of iterations), so that they refer to the clock of the original run; they are
estimates, accurate to the extent that the two machines differ by a constant factor during the
run.  DIR/trace_completion.json records what was added and checked, with the list of the
reconstructed traces, so that calling the script again gives the same report.

--main    logs of the main campaign (default results/)
--replay  logs of the replay (default results_replay/)
"""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd


def main():
    argv = sys.argv[1:]; MAIN = os.path.join(ROOT, 'results'); REP = os.path.join(ROOT, 'results_replay')
    for flag in ('--main', '--replay'):
        if flag in argv:
            k = argv.index(flag); v = argv[k + 1]; del argv[k:k + 2]
            if flag == '--main': MAIN = v
            else: REP = v
    rep = {'method': 'exact replay with the logged seed and number of iterations (experiments/campaign_replay.py)', 'added': 0, 'checked_existing': 0,
           'mismatch': 0, 'replayed_runs': 0, 'replay_ok': 0, 'files': {}, 'reconstructed': {}}
    ppath = os.path.join(MAIN, 'trace_completion.json')
    prev = json.load(open(ppath)) if os.path.exists(ppath) else {}
    for setname in ('S0', 'S1', 'S2'):
        for algo in ('VNS', 'BPX'):
            mp = os.path.join(MAIN, f'{setname}_{algo}.csv'); rp = os.path.join(REP, f'{setname}_{algo}.csv')
            if not (os.path.exists(mp) and os.path.exists(rp)): continue
            dm = pd.read_csv(mp); dr = pd.read_csv(rp); dr = dr[dr.replay_ok >= 0]
            tr_new = json.load(open(rp[:-4] + '_trace.json'))
            tpath = mp[:-4] + '_trace.json'; tr_old = json.load(open(tpath)) if os.path.exists(tpath) else {}
            m = dm.merge(dr[['inst', 'p', 'run', 'main_iters', 'replay_best', 'replay_ok', 't_replay']], on=['inst', 'p', 'run'])
            assert (m.main_iters == m.iters).all()
            added = checked = bad = 0; scale = []; new_keys = []
            before = set(prev.get('reconstructed', {}).get(f'{setname}_{algo}', []))     # traces added by an earlier call
            for r in m.itertuples():
                key = f'{r.inst}_{r.p}_{r.run}'
                seq = [(t, v) for t, v, it in tr_new[key] if it <= r.iters]
                ok = r.replay_ok == 1 and seq and seq[-1][1] == r.best
                f = r.time / r.t_replay if r.t_replay > 0 else 1.0
                if key in tr_old and key not in before:
                    checked += 1
                    if not ok or [v for _, v in seq] != [v for _, v in tr_old[key]]: bad += 1
                elif ok:
                    if key not in tr_old: new_keys.append(key)
                    elif [v for _, v in seq] != [v for _, v in tr_old[key]]: bad += 1
                    tr_old[key] = [[round(t * f, 4), v] for t, v in seq]; added += 1; scale.append(f)
                else:
                    bad += 1
            if new_keys: json.dump(tr_old, open(tpath, 'w'))
            rep['reconstructed'][f'{setname}_{algo}'] = sorted(before | set(new_keys))
            rep['files'][f'{setname}_{algo}'] = dict(logged_runs=len(dm), replayed=len(m), replay_ok=int((m.replay_ok == 1).sum()), traces_added=added, traces_checked=checked,
                                                   mismatch=bad, time_scale_mean=round(sum(scale) / len(scale), 4) if scale else None,
                                                   time_scale_min=round(min(scale), 4) if scale else None, time_scale_max=round(max(scale), 4) if scale else None)
            rep['added'] += added; rep['checked_existing'] += checked; rep['mismatch'] += bad; rep['replayed_runs'] += len(m); rep['replay_ok'] += int((m.replay_ok == 1).sum())
            print(f'{setname}_{algo}: {len(m)} runs replayed, {int((m.replay_ok == 1).sum())} with the logged value; {added} traces added, {checked} existing traces checked, {bad} mismatches')
    json.dump(rep, open(os.path.join(MAIN, 'trace_completion.json'), 'w'), indent=1)
    if rep['mismatch']: sys.exit('MISMATCH: see trace_completion.json')
    print('OK')


if __name__ == '__main__':
    main()
