"""Finite-ensemble Gaussian reference for the frozen uncertainty metrics.

Analysis only: this script does not change estimator outputs, primary scores or
interval construction. Run after dedicated latency measurements have finished.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import platform
import time

import numpy as np
from scipy.special import ndtr, ndtri


def accumulate(record, values):
    values = np.asarray(values, dtype=float)
    record['n'] += values.size
    record['sum'] += float(values.sum())
    record['sum_squares'] += float(values @ values)


def moments(record):
    n = record['n']
    mean = record['sum']/n
    variance = max(0., (record['sum_squares']-n*mean**2)/(n-1))
    se = np.sqrt(variance/n)
    return {'mean':mean,'mc_standard_error':float(se),
            'mc_95_interval':[float(mean-1.96*se),float(mean+1.96*se)],
            'independent_reference_draws':n}


def ensemble_crps(samples, target):
    """Same empirical CRPS as the study, with an efficient sorted pair term."""
    members = samples.shape[1]
    ordered = np.sort(samples, axis=1)
    weights = 2*np.arange(1,members+1)-members-1
    half_pair_sum = ordered @ weights
    absolute = np.abs(samples-target[:,None]).mean(1)
    # half_pair_sum equals .5 * sum_ij |x_i-x_j|.
    empirical = absolute-half_pair_sum/members**2
    fair = absolute-half_pair_sum/(members*(members-1))
    return empirical,fair


def check(draws=200000, members=16, seed=20261030, batch_size=20000):
    if draws < 2 or members < 2 or batch_size < 1:
        raise ValueError('Require at least two independent draws and ensemble members')
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    names = ('coverage','width','empirical_crps','fair_crps_diagnostic',
             'analytic_normal_crps','empirical_minus_analytic','fair_minus_analytic',
             'known_gaussian_interval_coverage')
    totals = {name:{'n':0,'sum':0.,'sum_squares':0.} for name in names}
    for begin in range(0,draws,batch_size):
        n = min(batch_size,draws-begin)
        samples = rng.normal(size=(n,members))
        target = rng.normal(size=n)
        lower,upper = np.quantile(samples,[.05,.95],axis=1,method='linear')
        empirical,fair = ensemble_crps(samples,target)
        analytic = target*(2*ndtr(target)-1)+2*np.exp(-target**2/2)/np.sqrt(2*np.pi)-1/np.sqrt(np.pi)
        values = {'coverage':((target>=lower)&(target<=upper)).astype(float),
                  'width':upper-lower,'empirical_crps':empirical,
                  'fair_crps_diagnostic':fair,'analytic_normal_crps':analytic,
                  'empirical_minus_analytic':empirical-analytic,
                  'fair_minus_analytic':fair-analytic,
                  'known_gaussian_interval_coverage':(np.abs(target)<=ndtri(.95)).astype(float)}
        for name,value in values.items():
            accumulate(totals[name],value)
    results = {name:moments(value) for name,value in totals.items()}
    results['coverage']['difference_from_nominal_0.9'] = results['coverage']['mean']-.9
    analytic_expected = 1/np.sqrt(np.pi)
    return {'created_utc':datetime.now(timezone.utc).isoformat(),
        'status':'completed','scope':'Analysis-only independent N(0,1) forecast/target reference; no robot histories, inference parameters, frozen intervals or primary scores are changed.',
        'design':{'independent_reference_draws':draws,'members':members,'seed':seed,
                  'batch_size':batch_size,'forecast_distribution':'Independent standard normal ensemble members',
                  'target_distribution':'Independent standard normal target for each ensemble',
                  'quantiles':[.05,.95],'quantile_method':'numpy linear, matching the frozen study'},
        'results':results,
        'known_expectations':{'analytic_predictive_crps':float(analytic_expected),
            'finite_empirical_crps':float((1+1/members)*analytic_expected),
            'fair_ensemble_crps_diagnostic':float(analytic_expected),
            'empirical_crps_excess':float(analytic_expected/members),
            'known_gaussian_interval_coverage':.9,
            'known_gaussian_interval_width':float(2*ndtri(.95)),
            'derivation':'E|X-Y|=2/sqrt(pi); the empirical pair correction is (M-1)/(M*sqrt(pi)), giving E[CRPS_M]=(1+1/M)/sqrt(pi). The fair pair correction excludes diagonal pairs and uses M*(M-1).'},
        'interpretation':['Finite-M linear empirical quantiles need not achieve their nominal predictive coverage even for an exactly correct Gaussian distribution.',
            'This Gaussian reference is not a distribution-free correction for the nonlinear FEM force distribution, especially its zero-force atom; it cannot establish that all observed undercoverage is sampling bias.',
            'The frozen empirical CRPS is a proper score of the reported empirical distribution. Its finite-ensemble expectation differs from the score of the underlying Gaussian; the fair-ensemble diagnostic is not substituted for the frozen primary score.',
            'Coefficient coverage uses analytic transformed-Gaussian quantiles, so this empirical-quantile Monte Carlo coverage effect does not apply to it.',
            'Monte Carlo standard errors here quantify independent synthetic reference draws; they do not increase the physical-history sample size of the research experiment.'],
        'runtime_seconds':time.perf_counter()-started,
        'environment':{'python':platform.python_version(),'numpy':np.__version__},
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path(__file__).resolve().parents[1]/'runs/inference/methods_20261030')
    parser.add_argument('--draws',type=int,default=200000)
    parser.add_argument('--members',type=int,default=16)
    parser.add_argument('--seed',type=int,default=20261030)
    parser.add_argument('--batch-size',type=int,default=20000)
    args = parser.parse_args()
    result = check(args.draws,args.members,args.seed,args.batch_size)
    output = args.run/'verification/finite_mc_reference.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'output':str(output),'coverage':result['results']['coverage'],
        'empirical_crps':result['results']['empirical_crps'],
        'known_expectations':result['known_expectations']},indent=2))


if __name__=='__main__':
    main()
