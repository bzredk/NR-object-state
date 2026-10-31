"""Common-anchor prior, propagation and observation-update controls on pilot data.

These controls keep the six-coefficient anchor at 0.60 s and the complete measured
motion span through 1.40 s. They are separate from the main window-start methods.
No final-test history is read or used for fitting or design choices.
"""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import time
import traceback

import numpy as np

from .physics import Config, SpatialModel
from .estimators import Observer
from .protocols import future_commands
from .inference_physics import WindowProblem
from .inference_study import (load_run, training_prior, fit_trf,
                              trace_and_observations, state_hash)
from .experiments import read_json, write_json, source_hashes


def digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


class CommonAnchorProblem:
    """Fit anchor coefficients using only the declared late observation rows."""

    def __init__(self, model, trace, observed, lateral, update, basis=None):
        self.update = update
        last = len(trace.time) - 1
        start = last if update == 'current' else last - round(.2/model.c.dt)
        if update not in ('current', 'recent_0.2'):
            raise ValueError('Unknown observation-update condition')
        permitted = np.asarray(observed).copy()
        # Unused targets are removed before residual construction; measured
        # position/velocity and command history remain available for propagation.
        permitted[:start, 2:] = 0.
        self.base = WindowProblem(model, trace, permitted, .8, lateral, basis)
        self.selection = np.arange(start, last + 1) - self.base.begin
        self.targets = permitted[start:last + 1, self.base.channels]
        self.sigma = self.base.sigma
        self.start = start

    def prediction_and_jacobian(self, coefficients):
        values, jac = self.base.observation_and_jacobian(coefficients)
        return values[self.selection], jac[self.selection]

    def residual(self, coefficients, jacobian=False, normalize=False):
        if jacobian:
            values, jac = self.prediction_and_jacobian(coefficients)
        else:
            values = self.base.replay(coefficients).observation[
                self.selection][:, self.base.channels]
        residual = (values-self.targets)/self.sigma
        divisor = np.sqrt(len(residual)) if normalize else 1.
        if jacobian:
            return residual.ravel()/divisor, (jac/self.sigma[None, :, None]).reshape(-1, 6)/divisor
        return residual.ravel()/divisor

    def current(self, coefficients):
        return self.base.current(coefficients)

    def record(self):
        trace = self.base.trace
        return {'anchor_time_s': float(trace.time[self.base.begin]),
                'observation_update_interval_s': [float(trace.time[self.start]), float(trace.time[-1])],
                'observation_update_samples': len(self.selection),
                'observation_channels': self.base.channels,
                'motion_interval_s': [float(trace.time[self.base.begin]), float(trace.time[-1])],
                'total_motion_span_s': float(trace.time[-1]-trace.time[self.base.begin]),
                'motion_channels': ['actual position', 'actual velocity at initialization/cut', 'command'],
                'older_force_and_shape': 'excluded before residual construction',
                'observation_selection': 'fixed interval; no contact detector or earlier force scan'}


def weak_diagnostics(problem):
    """Finite probes, not a global proof that a zero-force interval has no information."""
    projected, projection = problem.base.projection()
    direction = np.random.default_rng(6401).normal(size=6)
    direction /= np.linalg.norm(direction)
    candidates = [('projected truth', projected)]
    for magnitude in (.03, .12, .6, 1.5):
        candidates.append((f'uniform {magnitude}', np.full(6, magnitude)))
        candidates.append((f'projected perturbation {magnitude}',
                           np.clip(projected + magnitude*direction, 0., 2.5)))
    rows = []
    predicted = []
    for label, coefficients in candidates:
        try:
            values, jac = problem.prediction_and_jacobian(coefficients)
            whitened = (jac/problem.sigma[None, :, None]).reshape(-1, 6)
            residual = ((values-problem.targets)/problem.sigma).ravel()
            singular = np.linalg.svd(whitened, compute_uv=False)
            predicted.append(values)
            rows.append({'candidate': label, 'coefficients': coefficients.tolist(),
                         'max_predicted_force_N': float(values[:, 0].max()),
                         'any_force_above_1e-9_N': bool(np.any(values[:, 0] > 1e-9)),
                         'any_force_above_observed_contact_threshold': bool(np.any(values[:, 0] > .0045)),
                         'noise_std': problem.sigma.tolist(),
                         'whitened_jacobian_singular_values': singular.tolist(),
                         'numerical_rank': int(np.linalg.matrix_rank(whitened)),
                         'nll_gradient_norm': float(np.linalg.norm(whitened.T@residual)),
                         'gaussian_nll': float(.5*residual@residual + len(values)*np.log(problem.sigma*np.sqrt(2*np.pi)).sum())})
        except Exception as exc:
            rows.append({'candidate': label, 'error': str(exc), 'traceback': traceback.format_exc()})
    stack = np.asarray(predicted)
    spread = np.ptp(stack, axis=0).max(axis=0) if len(stack) else np.full(len(problem.sigma), np.nan)
    return {'update': problem.record(), 'projection_reference': projection, 'candidates': rows,
            'max_candidate_response_range': spread.tolist(),
            'max_candidate_response_range_in_noise_std': (spread/problem.sigma).tolist(),
            'scope': 'Local derivatives and finite candidate probes within the declared positive six-coefficient basis; not a universal observability claim.',
            'zero_force_can_exclude_states': 'Candidate contact and likelihood are evaluated, not assumed absent.'}


def control_job(job):
    run, parent, repeat, settings, draft = job
    run = Path(run)
    directory = run/'controls'/('common_anchor_draft' if draft else 'common_anchor')/parent/f'repeat_{repeat}'
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    trace, observed, noise_seed = trace_and_observations(run, 'pilot', parent, repeat)
    model = SpatialModel(Config(**trace.config)); basis = Observer(model).basis
    commands, phases = future_commands(trace.checkpoint(), model.c, 'Slow press')
    truth = model.simulate(commands, phases, state=trace.checkpoint())
    record = {'history_id': parent, 'split': 'pilot', 'noise_repeat': repeat,
              'noise_seed': noise_seed, 'settings': settings, 'draft_settings': draft,
              'observation_sha256': digest(observed), 'future_command_sha256': digest(commands),
              'future_command_source': 'generated once from the true checkpoint; identical for every method',
              'rows': [], 'failures': [], 'weak_likelihood': [], 'checks': []}
    for channels in ('basic', 'lateral'):
        prior = training_prior(run, '0.8', channels)
        base = WindowProblem(model, trace, observed, .8, channels == 'lateral', basis)
        anchor = base.begin
        np.savez_compressed(directory/f'{channels}_motion.npz',
                            time=trace.time[anchor:], actual_motion=observed[anchor:, :2],
                            command=trace.command[anchor:], observation=observed[anchor:])
        prior_states, prior_replay_seconds = [], []
        for _ in ('0.0', '0.2', '0.8'):
            t = time.perf_counter()
            prior_states.append(base.current(prior['mean']))
            prior_replay_seconds.append(time.perf_counter()-t)
        hashes = [state_hash(state) for state in prior_states]
        assert len(set(hashes)) == 1, 'A no-update condition changed with window label'
        record['checks'].append({'check': 'no_update_common_anchor_invariance', 'channels': channels,
                                 'passed': True, 'nominal_windows': ['0.0', '0.2', '0.8'],
                                 'state_sha256': hashes, 'replay_seconds': prior_replay_seconds})
        no_update_error = None
        for update in ('none', 'current', 'recent_0.2'):
            try:
                if update == 'none':
                    coefficients = prior['mean']
                    state = prior_states[0]
                    diag = {'fit_seconds': 0., 'optimizer_status': 'not_used'}
                    inputs = {'anchor_time_s': float(trace.time[anchor]), 'observation_update_interval_s': None,
                              'motion_interval_s': [float(trace.time[anchor]), float(trace.time[-1])],
                              'total_motion_span_s': float(trace.time[-1]-trace.time[anchor]),
                              'observation_channels': [], 'observation_update_samples': 0}
                    replay_seconds = prior_replay_seconds[0]
                else:
                    problem = CommonAnchorProblem(model, trace, observed, channels == 'lateral', update, basis)
                    coefficients, diag = fit_trf(problem, settings, prior)
                    t = time.perf_counter(); state = problem.current(coefficients); replay_seconds = time.perf_counter()-t
                    inputs = problem.record()
                    if repeat == 0:
                        changed = observed.copy(); changed[:problem.start, 2:] = np.nan
                        if channels == 'basic': changed[:, 3:] = np.nan
                        isolated = CommonAnchorProblem(model, trace, changed, channels == 'lateral', update, basis)
                        np.testing.assert_array_equal(problem.residual(prior['mean']), isolated.residual(prior['mean']))
                        record['checks'].append({'check': 'unused_observation_isolation', 'channels': channels, 'update': update, 'passed': True})
                t = time.perf_counter(); future = model.simulate(commands, phases, state=state); prediction_seconds = time.perf_counter()-t
                error = float(np.sqrt(np.mean((future.force-truth.force)**2))*1000)
                if update == 'none': no_update_error = error
                curve = directory/f'{channels}_{update}.npz'
                np.savez_compressed(curve, time=future.time, true_force=truth.force,
                                    predicted_force=future.force, true_markers=truth.markers,
                                    predicted_markers=future.markers)
                record['rows'].append({'channels': channels, 'update': update, 'force_rmse_mN': error,
                    'improvement_vs_no_update_mN': no_update_error-error,
                    'marker_rmse_um': float(np.sqrt(np.mean((future.markers-truth.markers)**2))*1e6),
                    'coefficients_at_common_anchor': coefficients.tolist(), 'prior_at_common_anchor': prior['mean'].tolist(),
                    'prior_source': 'training projected coefficients at 0.8-second window start',
                    'inputs': inputs, 'motion_record': str((directory/f'{channels}_motion.npz').relative_to(run)),
                    'fit': diag, 'replay_seconds': replay_seconds, 'prediction_seconds': prediction_seconds,
                    'state_sha256': state_hash(state), 'curve_path': str(curve.relative_to(run))})
            except Exception as exc:
                record['failures'].append({'channels': channels, 'update': update, 'error': str(exc), 'traceback': traceback.format_exc()})
        try:
            record['weak_likelihood'].append({'channels': channels,
                **weak_diagnostics(CommonAnchorProblem(model, trace, observed, channels == 'lateral', 'recent_0.2', basis))})
        except Exception as exc:
            record['failures'].append({'channels': channels, 'stage': 'weak_likelihood',
                                      'error': str(exc), 'traceback': traceback.format_exc()})
    record['wall_seconds'] = time.perf_counter()-started
    write_json(directory/'result.json', record)
    print(parent, repeat, f"{record['wall_seconds']:.2f}s", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--histories', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--draft', action='store_true', help='Explicit draft30 settings; save separately from selected calibration')
    args = parser.parse_args()
    run, spec = load_run(args.run)
    if args.draft:
        settings = {'name':'common_anchor_draft30','initial':'train_mean','regularization':0.,'max_nfev':30}
    else:
        settings = read_json(run/'calibration/trf.json')['selected']
    parents = [f'pilot_{seed}' for seed in spec['pilot_seeds'][:args.histories]]
    if not parents or args.repeats < 1 or args.repeats > spec['noise_repeats']:
        raise ValueError('Invalid requested pilot histories or noise repeats')
    jobs = [(str(run), parent, repeat, settings, args.draft) for parent in parents for repeat in range(args.repeats)]
    if args.workers == 1:
        records = [control_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            records = list(pool.map(control_job, jobs))
    summary = []
    for channels in ('basic', 'lateral'):
        for update in ('none', 'current', 'recent_0.2'):
            values = []
            gains = []
            for parent in parents:
                rows = [r for record in records if record['history_id'] == parent for r in record['rows']
                        if r['channels'] == channels and r['update'] == update]
                if rows:
                    values.append(float(np.mean([r['force_rmse_mN'] for r in rows])))
                    gains.append(float(np.mean([r['improvement_vs_no_update_mN'] for r in rows])))
            summary.append({'channels': channels, 'update': update,
                            'history_mean_force_rmse_mN': float(np.mean(values)),
                            'history_mean_improvement_mN': float(np.mean(gains)), 'independent_histories': len(values)})
    output = run/'controls'/('common_anchor_draft.json' if args.draft else 'common_anchor.json')
    write_json(output, {'status': 'completed', 'scope':'pilot diagnostic; common anchor and .8s motion for every update condition',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'independent_histories': len(parents),
        'noise_repeats': args.repeats, 'workers':args.workers, 'settings':settings, 'draft_settings':args.draft,
        'summary':summary, 'records':records, 'code_sha256':source_hashes(),
        'limitations':'This control deliberately grants the same .8s motion to all variants; it is not the original zero/short-history comparison. Candidate sensitivity probes are finite diagnostics, not universal observability proofs.'})
    print(json.dumps({'output':str(output),'summary':summary}, indent=2), flush=True)


if __name__ == '__main__':
    main()
