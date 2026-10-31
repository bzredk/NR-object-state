"""History-level inference summaries and a separate serial CPU latency suite."""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import time
import traceback

import numpy as np

from .experiments import read_json, write_json, source_hashes


GROUP = ('split', 'method', 'channels', 'window', 'action')
METRICS = ('force_rmse_mN', 'marker_rmse_um', 'force_coverage90',
           'force_interval_width_mN', 'force_contact_coverage90',
           'force_contact_interval_width_mN', 'force_crps_mN', 'noisy_force_coverage90',
           'noisy_force_interval_width_mN', 'noisy_force_crps_mN',
           'coefficient_coverage90', 'posterior_physics_violation_rate',
           'physics_violation_rate', 'posterior_prior_logit_sd_ratio',
           'posterior_prior_mean_shift', 'final_kl')


def normalized_row(row):
    row = dict(row)
    # Early records used WindowProblem.record()'s integer observation channels.
    if isinstance(row.get('channels'), list):
        row['channels'] = 'lateral' if 3 in row['channels'] else 'basic'
    if 'window' in row:
        row['window'] = str(row['window'])
        if row['window'] in ('0', '0.00'):
            row['window'] = '0.0'
    if isinstance(row.get('physics'), dict):
        row['physics_violation_rate'] = float(not row['physics']['passed'])
    return row


def bootstrap(values, label, resamples):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {'mean': None, 'ci95': None, 'independent_histories': 0}
    interval = None
    if len(values) >= 2:
        seed = int.from_bytes(hashlib.sha256(str(label).encode()).digest()[:8], 'little')
        rng = np.random.default_rng(seed)
        means = values[rng.integers(0, len(values), (resamples, len(values)))].mean(1)
        interval = np.quantile(means, [.025, .975]).tolist()
    return {'mean': float(values.mean()), 'ci95': interval,
            'history_sd': float(values.std(ddof=1)) if len(values) > 1 else None,
            'independent_histories': len(values)}


def metric_cells(rows, metric):
    """History -> noise repeat -> training seed, without inventing sample units."""
    values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        value = row.get(metric)
        if value is not None and np.isfinite(float(value)):
            values[row['history_id']][int(row['noise_repeat'])][int(row.get('training_seed', -1))].append(float(value))
    return {h: {n: {s: float(np.mean(v)) for s, v in seeds.items()}
                for n, seeds in repeats.items()} for h, repeats in values.items()}


def history_values(rows, metric):
    values = metric_cells(rows, metric)
    result = {}
    for history, repeats in values.items():
        by_seed = defaultdict(list)
        for seeds in repeats.values():
            for seed, value in seeds.items():
                by_seed[seed].append(value)
        result[history] = float(np.mean([np.mean(v) for v in by_seed.values()]))
    return result


def paired_values(left, right, metric='force_rmse_mN'):
    """Positive gain means left error minus right error on shared observations."""
    left, right = metric_cells(left, metric), metric_cells(right, metric)
    pairs = []
    for history in sorted(left.keys() & right.keys()):
        repeats = []
        for noise in sorted(left[history].keys() & right[history].keys()):
            a, b = left[history][noise], right[history][noise]
            # Shared learned seeds are paired when both sides are learned.
            if all(s >= 0 for s in a) and all(s >= 0 for s in b):
                shared = sorted(a.keys() & b.keys())
                if not shared:
                    continue
                av, bv = np.mean([a[s] for s in shared]), np.mean([b[s] for s in shared])
            else:
                shared = sorted(a.keys() & b.keys())
                av, bv = np.mean(list(a.values())), np.mean(list(b.values()))
            repeats.append({'noise_repeat': noise, 'gain_mN': float(av-bv),
                            'left_error_mN': float(av), 'right_error_mN': float(bv),
                            'left_training_seeds': sorted(a), 'right_training_seeds': sorted(b),
                            'matched_training_seeds': shared})
        if repeats:
            pairs.append({'history_id': history,
                          'gain_mN': float(np.mean([r['gain_mN'] for r in repeats])),
                          'matched_noise_repeats': len(repeats), 'noise_values': repeats})
    return pairs, {'left_histories': len(left), 'right_histories': len(right),
                   'left_only_histories': sorted(left.keys()-right.keys()),
                   'right_only_histories': sorted(right.keys()-left.keys())}


def matches(row, selector):
    return all(row.get(k) == v for k, v in selector.items())


def failure_counts(failures, selector):
    # An inference failure is one recorded event, not one event per future branch.
    condition = {k: v for k, v in selector.items() if k != 'action'}
    found = [f for f in failures if matches(f, condition)
             and (not f.get('action') or f['action'] == selector.get('action'))]
    return {'recorded_failure_events': len(found),
            'failed_histories': len({f['history_id'] for f in found}),
            'inference_failure_events': sum(not f.get('action') for f in found),
            'future_action_failure_events': sum(bool(f.get('action')) for f in found)}


def summarize(run):
    """Write descriptive means and complete-history percentile bootstrap intervals."""
    run = Path(run)
    spec = read_json(run/'study.json')
    resamples = int(spec.get('bootstrap_resamples', 2000))
    rows, failures, diagnostics, files = [], [], [], []
    observed_jobs = defaultdict(set)
    for path in sorted((run/'evaluation').rglob('result.json')):
        record = read_json(path)
        observed_jobs[record['split']].add((record['history_id'],int(record['repeat'])))
        rows.extend(normalized_row(r) for r in record['rows'])
        failures.extend(normalized_row(f) for f in record['failures'])
        files.append(str(path.relative_to(run)).replace('\\', '/'))
        source = path.with_name('inference.json')
        if source.exists():
            diagnostics.extend(normalized_row(d) for d in read_json(source))
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in GROUP)].append(row)
    groups = []
    for key, found in sorted(grouped.items()):
        selector = dict(zip(GROUP, key))
        metrics = {}
        for metric in METRICS:
            values = history_values(found, metric)
            if values:
                metrics[metric] = bootstrap(list(values.values()), (key, metric), resamples)
        seeds = sorted({int(r.get('training_seed', -1)) for r in found if int(r.get('training_seed', -1)) >= 0})
        by_seed = []
        for seed in seeds:
            values = history_values([r for r in found if int(r.get('training_seed', -1)) == seed], 'force_rmse_mN')
            by_seed.append({'training_seed': seed, **bootstrap(list(values.values()), (key, seed), resamples)})
        seed_means = [s['mean'] for s in by_seed]
        groups.append({**selector, 'successful_prediction_rows': len(found),
                       'independent_histories': len({r['history_id'] for r in found}),
                       'noise_repeats_by_history': {h: len({r['noise_repeat'] for r in found if r['history_id'] == h})
                                                    for h in sorted({r['history_id'] for r in found})},
                       'metrics': metrics, **failure_counts(failures, selector),
                       'training_seed_variability': {'seed_results': by_seed,
                           'sd_of_seed_means_mN': float(np.std(seed_means, ddof=1)) if len(seed_means)>1 else None,
                           'scope': 'Training-seed variation is separate from independent physical-history uncertainty.'}})
    conditions = [
        ('current lateral benefit', {'channels':'basic','window':'0.0'}, {'channels':'lateral','window':'0.0'}),
        ('0.2s lateral history benefit', {'channels':'lateral','window':'0.0'}, {'channels':'lateral','window':'0.2'}),
        ('0.8s lateral history benefit', {'channels':'lateral','window':'0.0'}, {'channels':'lateral','window':'0.8'}),
        ('earlier contact vs recent wait', {'channels':'basic','window':'0.2'}, {'channels':'basic','window':'contact'})]
    contrasts = []
    splits = sorted({r['split'] for r in rows})
    actions = sorted({r['action'] for r in rows})
    methods = sorted({r['method'] for r in rows}
                     - {'True state','Projected truth','Full replay'})
    def contrast(label, left, right):
        pairs, support = paired_values([r for r in rows if matches(r,left)],
                                       [r for r in rows if matches(r,right)])
        if pairs or support['left_histories'] or support['right_histories']:
            contrasts.append({'contrast': label, 'split': left['split'], 'action': left['action'],
                'method': left['method'] if left['method']==right['method'] else right['method'],
                'left': left, 'right': right, 'gain_definition': 'left RMSE minus right RMSE; positive favors right',
                'gain_mN': bootstrap([p['gain_mN'] for p in pairs], (label,left,right), resamples),
                'matched_histories': len(pairs), 'support': support, 'history_gains': pairs,
                'left_failures': failure_counts(failures,left), 'right_failures': failure_counts(failures,right)})
    for split in splits:
        for action in actions:
            common = {'split':split,'action':action}
            for method in methods:
                for label, left, right in conditions:
                    contrast(label, {**common,'method':method,**left}, {**common,'method':method,**right})
            for channels in ('basic','lateral'):
                for window in ('0.0','0.2','0.8','contact'):
                    for prior, label in [('RME prior','VI vs neural center'), ('RME prior mean','VI vs same-summary prior')]:
                        shared = {**common,'channels':channels,'window':window}
                        contrast(label, {**shared,'method':prior}, {**shared,'method':'RME-inspired VI'})
    differences = []
    for split in splits:
        for action in actions:
            for label, _, _ in conditions[:3]:
                for baseline in ('TRF calibrated','TRF published'):
                    for method in ('GRU','RME-inspired VI'):
                        selected = {m: next((c for c in contrasts if c['split']==split and c['action']==action
                                              and c['contrast']==label and c['method']==m), None)
                                    for m in (baseline,method)}
                        if not all(selected.values()):
                            continue
                        left = {p['history_id']:p for p in selected[baseline]['history_gains']}
                        right = {p['history_id']:p for p in selected[method]['history_gains']}
                        pairs = []
                        for h in sorted(left.keys() & right.keys()):
                            a = {n['noise_repeat']:n['gain_mN'] for n in left[h]['noise_values']}
                            b = {n['noise_repeat']:n['gain_mN'] for n in right[h]['noise_values']}
                            shared = sorted(a.keys() & b.keys())
                            if shared:
                                pairs.append({'history_id':h, 'gain_difference_mN':float(np.mean([b[n]-a[n] for n in shared])),
                                              'matched_noise_repeats':len(shared)})
                        if pairs:
                            differences.append({'split':split,'action':action,'contrast':label,
                                'method':method,'baseline':baseline,
                                'definition':'method information gain minus TRF information gain; positive favors method',
                                'difference_mN':bootstrap([p['gain_difference_mN'] for p in pairs],
                                                          (split,action,label,method,baseline),resamples),
                                'matched_histories':len(pairs),'history_differences':pairs})
    projected = defaultdict(dict)
    for d in diagnostics:
        value = d['inference'].get('relative_field_error')
        if value is not None:
            projected[(d['split'],d['channels'],d['window'])][d['history_id']] = float(value)
    projection = [{'split':k[0],'channels':k[1],'window':k[2],
                   'relative_field_error':bootstrap(list(v.values()),('projection',k),resamples),
                   'history_values':v} for k,v in sorted(projected.items())]
    observation_hashes, command_hashes, state_hashes = defaultdict(set), defaultdict(set), defaultdict(set)
    for row in rows:
        if row.get('observation_sha256'):
            observation_hashes[(row['split'],row['history_id'],row['noise_repeat'])].add(row['observation_sha256'])
        if row.get('command_sha256'):
            command_hashes[(row['split'],row['history_id'],row['noise_repeat'],row['action'])].add(row['command_sha256'])
        if row.get('state_sha256'):
            state_hashes[(row['split'],row['history_id'],row['noise_repeat'],row['method'],
                          row.get('training_seed',-1),row['channels'],row['window'])].add(row['state_sha256'])
    mismatched_observations = [{'split':k[0],'history_id':k[1],'noise_repeat':k[2],
                               'distinct_hashes':len(v)} for k,v in observation_hashes.items() if len(v)>1]
    mismatched_commands = [{'split':k[0],'history_id':k[1],'noise_repeat':k[2],
                            'action':k[3],'distinct_hashes':len(v)} for k,v in command_hashes.items() if len(v)>1]
    mismatched_states = [{'split':k[0],'history_id':k[1],'noise_repeat':k[2],
                          'method':k[3],'training_seed':k[4],'channels':k[5],'window':k[6],
                          'distinct_hashes':len(v)} for k,v in state_hashes.items() if len(v)>1]
    frozen_path = run/'frozen.json'
    frozen = read_json(frozen_path) if frozen_path.exists() else None
    noise_repeats = int(spec['noise_repeats'])
    expected_jobs = {'pilot':{(f'pilot_{seed}',0) for seed in spec['pilot_seeds']},
                     'old':{(f'test_{seed}',repeat) for seed in range(5000,5024) for repeat in range(noise_repeats)},
                     'test':{(f'test_{seed}',repeat) for seed in (frozen['test_seeds'] if frozen else [])
                             for repeat in range(noise_repeats)}}
    job_coverage = {split:{'expected_jobs':len(expected),
                          'observed_unique_jobs':len(observed_jobs[split]),
                          'matched_expected_jobs':len(expected & observed_jobs[split]),
                          'missing_jobs':[{'history_id':h,'noise_repeat':r} for h,r in sorted(expected-observed_jobs[split])],
                          'unexpected_jobs':[{'history_id':h,'noise_repeat':r} for h,r in sorted(observed_jobs[split]-expected)],
                          'pending_test_freeze':split=='test' and frozen is None,
                          'completed':expected <= observed_jobs[split] and not (split=='test' and frozen is None)}
                    for split,expected in expected_jobs.items()}
    summary = {'created_utc':datetime.now(timezone.utc).isoformat(),
        'status':'completed' if all(v['completed'] for v in job_coverage.values()) else 'partial',
        'job_coverage':job_coverage,
        'statistical_unit':'Independent complete loading history; noise repetitions, training seeds, frames and future branches do not increase N.',
        'aggregation':'Noise repetitions are averaged within each training seed and history, then available training seeds are equally averaged. Paired contrasts match history/noise; learned-vs-learned comparisons also match training seeds.',
        'bootstrap':{'method':'percentile bootstrap of whole-history means or paired whole-history gains',
                     'resamples':resamples,'interval':.95,'single_history_interval':None,
                     'interpretation':'Descriptive uncertainty; no equivalence, universal sufficiency or multiplicity-adjusted significance claim.'},
        'counts':{'successful_prediction_rows':len(rows),'recorded_failure_events':len(failures),
                  'histories_by_split':{s:len({r['history_id'] for r in rows if r['split']==s}) for s in splits}},
        'pairing_check':{'shared_observation_hashes_passed':not mismatched_observations,
                         'mismatched_observation_units':mismatched_observations,
                         'shared_future_commands_passed':not mismatched_commands,
                         'mismatched_command_units':mismatched_commands,
                         'cross_action_state_reuse_passed':not mismatched_states,
                         'mismatched_state_units':mismatched_states,
                         'rows_missing_command_hash':sum(not r.get('command_sha256') for r in rows),
                         'rows_missing_state_hash':sum(not r.get('state_sha256') for r in rows)},
        'groups':groups,'paired_contrasts':contrasts,'difference_of_information_gains':differences,
        'weak_information':{'group_indices':[i for i,g in enumerate(groups)
             if g['method'] in ('RME-inspired VI','Fixed prior VI') and g['window']=='0.2' and g['action']=='Slow press'],
             'metrics':['posterior_prior_logit_sd_ratio','posterior_prior_mean_shift','final_kl',
                        'force_coverage90','force_interval_width_mN','force_crps_mN'],
             'common_anchor_control':'controls/common_anchor.json' if (run/'controls/common_anchor.json').exists() else None,
             'interpretation':'Scale ratio and logit-mean shift compare the final q with its supplied prior. Waiting-force sensitivity is diagnosed separately; zero force is not assumed globally uninformative.'},
        'projection_diagnostics':projection,'failures':failures,
        'reference_information':{'True state':'Privileged full internal fields at the prediction cut.',
             'Projected truth':'Privileged bounded projection of the full field at the window start; shared FEM replay follows. Field error is not a predictive-performance lower bound.',
             'Full replay':'Privileged known relaxed initial state and complete motion history.',
             'learned_models':'Privileged projected fields supervise offline training; deployment sees only declared sensor channels.',
             'RME-inspired VI':'Data-conditioned prior and same-window likelihood; no automatic Bayesian calibration guarantee.'},
        'probability_scope':'Coverage is the fraction of future time points covered in each trajectory, then averaged by history. It does not treat frames as independent samples. Measurement-noise intervals are separate from latent-state uncertainty.',
        'timing_scope':'Evaluation total_seconds is phase compute under parallel work. Use the dedicated serial latency.csv for online-compute timing.',
        'sources':files,'code_sha256':source_hashes()}
    write_json(run/'summary.json',summary)
    return summary


def latency_suite(run, repeats=3):
    """Serial batch-one compute timing; caller must stop competing batch work."""
    from threadpoolctl import threadpool_info, threadpool_limits
    import torch
    from . import inference_study as study
    if repeats < 1:
        raise ValueError('At least one warm timing is required')
    run = Path(run)
    spec = read_json(run/'study.json')
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    torch.set_num_threads(1)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = run/'latency_records'/stamp
    output.mkdir(parents=True,exist_ok=True)
    methods = ['TRF published','TRF calibrated','GRU','RME prior','RME-inspired VI','Fixed prior VI']
    rows, failures = [], []
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        pools = threadpool_info()
        for channels,window in [('lateral','0.0'),('lateral','0.2'),('lateral','0.8'),('basic','0.2')]:
            selected = methods + (['Snapshot MLP'] if window=='0.0' else [])
            for method in selected:
                if method=='Fixed prior VI' and [channels,window] not in spec['probability_conditions']:
                    continue
                study._MODELS.clear()
                condition = f"{channels}_{window}_{method.replace(' ','_')}"
                for trial in range(repeats+1):
                    temperature = 'cold' if trial==0 else 'warm'
                    destination = output/condition/f'{temperature}_{trial}'
                    options = {'output':str(destination),'windows':[window],'channels':[channels],
                               'actions':['Slow press'],'training_seeds':[11],'methods':[method],
                               'force':True,'remeasure_published':True,'cache_forecasts':False}
                    t = time.perf_counter()
                    try:
                        record = study.evaluate_job((str(run),'pilot','pilot_14000',0,options))
                        wall = time.perf_counter()-t
                        for row in record['rows']:
                            rows.append({**row,'temperature':temperature,'warm_repeat':trial if trial else 0,
                                'dedicated_wall_seconds':wall,
                                'diagnostics_setup_seconds':wall-float(row['total_seconds']),
                                'batch_size':1,'torch_cpu_threads':torch.get_num_threads(),
                                'condition':condition,
                                'timing_scope':'Serial phase compute, including requested posterior replay/prediction; excludes experiment truth generation, privileged projection, setup and output I/O.',
                                'cold_definition':'First condition invocation after model-cache clear within this process; includes learned model loading, not OS/Python startup.'})
                        failures.extend({**f,'temperature':temperature,'warm_repeat':trial} for f in record['failures'])
                    except Exception as exc:
                        wall = time.perf_counter()-t
                        failures.append({'method':method,'channels':channels,'window':window,
                            'temperature':temperature,'warm_repeat':trial,'error':str(exc),
                            'dedicated_wall_seconds':wall,'traceback':traceback.format_exc()})
                    print('latency',condition,temperature,trial,f'{wall:.3f}s',flush=True)
    study.write_csv(run/'latency.csv',rows)
    study.write_csv(output/'latency.csv',rows)
    cpu_model = platform.processor()
    cpu_info = Path('/proc/cpuinfo')
    if cpu_info.exists():
        cpu_model = next((line.partition(':')[2].strip() for line in cpu_info.read_text().splitlines()
                          if line.startswith('model name')), cpu_model)
    write_json(run/'latency.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'record_directory':str(output.relative_to(run)).replace('\\','/'),
        'completed_rows':len(rows),'failures':failures,'warm_repeats':repeats,
        'wall_seconds':time.perf_counter()-started,'cpu':cpu_model,
        'torch_version':torch.__version__,'torch_cpu_threads':torch.get_num_threads(),'threadpools':pools,
        'conditions':'pilot_14000, noise repeat 0, training seed 11; Slow press; CPU, serial, batch 1',
        'online_compute':'Use total_seconds and phase fields; posterior replay and prediction are included where requested. dedicated_wall_seconds also includes experimental setup, true trajectory/projection diagnostics and I/O, so is not deployment latency.',
        'cold_definition':'First selected-condition invocation with cleared learned model cache in an existing Python process. Packages and earlier FEM routines can be warm; not a fresh OS-process measurement.',
        'forecast_cache':'Disabled for every dedicated timing call. Batch evaluation may reuse exact state/action forecasts; likelihood and Jacobian calls are not replaced by that cache.',
        'exclusivity':'Caller is responsible for stopping competing training/evaluation workers before invoking this suite.',
        'code_sha256':source_hashes()})
    return rows


def main():
    from .inference_study import load_run
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['summarize','latency'])
    parser.add_argument('--run',type=Path)
    parser.add_argument('--repeats',type=int,default=3)
    args = parser.parse_args()
    run,_ = load_run(args.run)
    result = summarize(run) if args.stage=='summarize' else latency_suite(run,args.repeats)
    print(json.dumps({'stage':args.stage,'output':str(run/('summary.json' if args.stage=='summarize' else 'latency.csv')),
                      'groups':len(result.get('groups',[])) if isinstance(result,dict) else len(result)},indent=2))


if __name__=='__main__':
    main()
