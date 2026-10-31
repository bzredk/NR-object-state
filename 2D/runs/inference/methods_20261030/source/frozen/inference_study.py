"""Controlled estimator comparison with the existing fixed spatial FEM.

Run stages explicitly; final-test histories are generated only after freeze.
The old 24-history batch is seen reanalysis, never a fresh confirmation set.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.metadata
import json
import shutil
import time
import traceback

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit, logit

from .physics import Config, SpatialModel, State, Trace
from .estimators import Observer, ObserverConfig, NOISE_STD
from .protocols import future_commands
from .experiments import ROOT, read_json, write_json, environment, source_hashes
from .information import audit_trace, infer_condition
from .inference_physics import WindowProblem
from .inference_data import prepare_dataset, load_dataset, WINDOWS, CHANNELS, PHYSICS_LIMITS


CONFIG = ROOT / 'configs/inference_study.json'
_MODELS = {}
_CODE_AT_IMPORT = source_hashes()


def load_run(run=None):
    config = read_json(CONFIG)
    run = ROOT / config['run'] if run is None else Path(run)
    return run.resolve(), read_json(run / 'study.json')


def state_hash(state):
    return hashlib.sha256(state.z.tobytes() + state.u.tobytes() +
                          np.array([state.d, state.v, state.t]).tobytes()).hexdigest()


def model_for(run, kind, seed, size):
    key = (str(run), kind, seed, size)
    if key not in _MODELS:
        from .learning_models import load_model
        _MODELS[key] = load_model(run / 'models' / f'{kind}_n{size}_s{seed}.pt')
    return _MODELS[key]


def training_prior(run, window, channels):
    with np.load(run / 'datasets/train.npz', allow_pickle=False) as data:
        y = data['y'][(data['windows'] == str(window)) & (data['channels'] == channels)].astype(float)
    if not len(y):
        raise ValueError('No matching training-label prior')
    z = logit(np.clip(y / 2.5, 1e-4, 1-1e-4))
    return {'mean': y.mean(0), 'sd': np.maximum(y.std(0), .02),
            'latent_mean': z.mean(0), 'latent_sd': np.maximum(z.std(0), .15),
            'fixed_center': 2.5 * expit(z.mean(0))}


def fit_trf(problem, settings, prior, initial=None):
    """Analytic Jacobian of the original FEM, with convergence diagnostics."""
    started = time.perf_counter()
    initial = (prior['mean'] if settings['initial'] == 'train_mean' else np.full(6, .12)) if initial is None else initial
    last, cached = None, None
    penalty = float(settings.get('regularization', 0.))
    def evaluate(c):
        nonlocal last, cached
        if last is None or not np.array_equal(c, last):
            residual, jac = problem.residual(c, jacobian=True, normalize=True)
            if penalty:
                residual = np.r_[residual, penalty * (c-prior['mean']) / prior['sd']]
                jac = np.vstack((jac, np.diag(penalty / prior['sd'])))
            last, cached = c.copy(), (residual, jac)
        return cached
    result = least_squares(lambda c:evaluate(c)[0], np.clip(initial, 1e-7, 2.5-1e-7),
                           jac=lambda c:evaluate(c)[1], bounds=(0., 2.5),
                           max_nfev=int(settings['max_nfev']), ftol=1e-7, xtol=1e-7, gtol=1e-7)
    return result.x, {'encode_seconds':0., 'vi_seconds':0.,
                      'fit_seconds':time.perf_counter()-started,
                      'optimizer_status':str(result.status), 'optimizer_success':bool(result.success),
                      'nfev':int(result.nfev), 'njev':int(result.njev),
                      'optimality':float(result.optimality), 'cost':float(result.cost),
                      'coefficient_start':np.asarray(initial).tolist(), 'coefficients':result.x.tolist(),
                      'settings':settings, 'gradient':'implicit exact-FEM Jacobian'}


def trace_and_observations(run, split, parent, repeat):
    if split == 'old':
        old = ROOT / read_json(run/'study.json')['old_run']
        trace = Trace.load(old/'histories'/f'{parent}.npz')
        with np.load(old/'test'/parent/f'repeat_{repeat}'/'curves.npz') as data:
            observed = data['observed'].copy()
        noise_seed = read_json(old/'test'/parent/f'repeat_{repeat}'/'result.json')['noise_seed']
    else:
        trace = Trace.load(run/'histories'/f'{parent}.npz')
        with np.load(run/'observations'/parent/f'repeat_{repeat}.npz') as data:
            observed = data['observed'].copy()
            noise_seed = int(data['noise_seed'])
    return trace, observed, noise_seed


def _forecast(model, state, action):
    command, phase = model._shared_future_commands[action]
    key=(state_hash(state),action)
    if model._cache_forecasts and key in model._forecast_cache:
        return model._forecast_cache[key]
    future=model.simulate(command, phase, state=state)
    if model._cache_forecasts:model._forecast_cache[key]=future
    return future


def set_shared_commands(model, trace):
    # Experimental commands are generated once, never from a method's estimate.
    model._shared_future_commands = {action:future_commands(trace.checkpoint(), model.c, action)
                                     for action in ('Slow press', 'Fast press')}
    model._cache_forecasts=True
    model._forecast_cache={}


def _calibration_job(job):
    run, parent, candidate = Path(job[0]), job[1], job[2]
    out = run/'calibration/trf'/candidate['name']/f'{parent}.json'
    if out.exists() and read_json(out).get('shared_future_commands'):
        return read_json(out)
    previous=read_json(out) if out.exists() else None
    if previous:shutil.copy2(out,out.with_suffix('.before_shared_commands.json'))
    trace, observed, _ = trace_and_observations(run, 'calibration', parent, 0)
    model = SpatialModel(Config(**trace.config)); basis=Observer(model).basis
    set_shared_commands(model,trace)
    rows=[]
    for window in WINDOWS:
        for channels in CHANNELS:
            p=WindowProblem(model,trace,observed,window,channels=='lateral',basis)
            prior=training_prior(run,window,channels)
            try:
                cached=next((r for r in previous['rows'] if r['window']==window and r['channels']==channels and 'coefficients' in r),None) if previous else None
                if cached:
                    # Fitting never used future actions; only rescore its saved
                    # coefficients under the corrected shared command schedule.
                    c=np.asarray(cached['coefficients']);diag={**cached,'rescored_shared_commands':True}
                else:c,diag=fit_trf(p,candidate,prior)
                pred=_forecast(model,p.current(c),'Slow press')
                truth=_forecast(model,trace.checkpoint(),'Slow press')
                rows.append({'window':window,'channels':channels,'force_rmse_mN':float(np.sqrt(np.mean((pred.force-truth.force)**2))*1000),**diag})
            except Exception as exc:
                rows.append({'window':window,'channels':channels,'error':str(exc),'traceback':traceback.format_exc()})
    record={'parent':parent,'candidate':candidate,'rows':rows,'shared_future_commands':True}
    write_json(out,record)
    return record


def calibrate_trf(run, spec, workers):
    jobs=[(str(run),f'calibration_{seed}',candidate) for candidate in spec['trf_candidates'] for seed in spec['calibration_seeds'][:3]]
    records=parallel(_calibration_job,jobs,workers)
    scores=[]
    for candidate in spec['trf_candidates']:
        rows=[r for record in records if record['candidate']['name']==candidate['name'] for r in record['rows']]
        valid=[r for r in rows if 'force_rmse_mN' in r]
        scores.append({'candidate':candidate,'force_rmse_mN':float(np.mean([r['force_rmse_mN'] for r in valid])),
                       'failed':len(rows)-len(valid),'budget_limited':sum(not r['optimizer_success'] for r in valid),
                       'median_fit_seconds':float(np.median([r['fit_seconds'] for r in valid]))})
    selected=min((s for s in scores if not s['failed']),key=lambda s:s['force_rmse_mN'])['candidate']
    write_json(run/'calibration/trf.json',{'selected':selected,'scores':scores,'source':'3 separate calibration histories; Slow press only','frozen_utc':datetime.now(timezone.utc).isoformat()})
    print(json.dumps(scores,indent=2),flush=True)


def _vi_calibration_job(job):
    from .learning_models import predict_coefficients
    from .variational import infer_vi
    run,parent,candidate=Path(job[0]),job[1],job[2]
    key=f"steps{candidate['steps']}_scale{candidate['scale_multiplier']}"
    out=run/'calibration/vi'/key/f'{parent}.json'
    if out.exists():return read_json(out)
    trace,obs,_=trace_and_observations(run,'calibration',parent,0)
    model=SpatialModel(Config(**trace.config));basis=Observer(model).basis
    set_shared_commands(model,trace)
    neural=model_for(run,'rme',11,128)
    rows=[]
    for channels,window in [('basic','0.2'),('lateral','0.2'),('lateral','0.8')]:
        p=WindowProblem(model,trace,obs,window,channels=='lateral',basis)
        x,length=p.inputs(); center=predict_coefficients(neural,x[None],np.array([length]))[0]
        scale=np.asarray(neural.report['prior_logit_scale'])*candidate['scale_multiplier']
        try:
            result=infer_vi(p,center,scale,seed=12345,steps=candidate['steps'],lr=candidate['lr'],posterior_samples=8)
            pred=_forecast(model,p.current(result['coefficients']),'Slow press')
            truth=_forecast(model,trace.checkpoint(),'Slow press')
            rows.append({'channels':channels,'window':window,'force_rmse_mN':float(np.sqrt(np.mean((pred.force-truth.force)**2))*1000),
                         'seconds':result['seconds'],'last_loss':result['trace'][-1]['loss'],'trace':result['trace']})
        except Exception as exc:rows.append({'channels':channels,'window':window,'error':str(exc)})
    record={'candidate':candidate,'parent':parent,'rows':rows};write_json(out,record);return record


def calibrate_vi(run,spec,workers):
    records=parallel(_vi_calibration_job,[(str(run),f'calibration_{s}',c) for c in spec['vi_candidates'] for s in spec['calibration_seeds'][3:6]],workers)
    scores=[]
    for candidate in spec['vi_candidates']:
        rows=[r for record in records if record['candidate']==candidate for r in record['rows']]
        valid=[r for r in rows if 'force_rmse_mN' in r]
        scores.append({'candidate':candidate,'force_rmse_mN':float(np.mean([r['force_rmse_mN'] for r in valid])),
                       'failed':len(rows)-len(valid),'median_vi_seconds':float(np.median([r['seconds'] for r in valid]))})
    selected=min((s for s in scores if not s['failed']),key=lambda s:s['force_rmse_mN'])['candidate']
    write_json(run/'calibration/vi.json',{'selected':selected,'scores':scores,'source':'3 separate calibration histories;3 prespecified conditions; Slow press only','frozen_utc':datetime.now(timezone.utc).isoformat()})
    print(json.dumps(scores,indent=2),flush=True)


def posterior_curves(problem,samples,action):
    history=prediction=0.; forces=[];markers=[];audits=[]
    for c in samples:
        start=time.perf_counter();state=problem.current(c);history+=time.perf_counter()-start
        start=time.perf_counter();future=_forecast(problem.model,state,action);prediction+=time.perf_counter()-start
        forces.append(future.force);markers.append(future.markers)
        audits.append(audit_trace(future,PHYSICS_LIMITS))
    return np.asarray(forces),np.asarray(markers),history,prediction,audits


def crps(samples,truth):
    return float((np.mean(np.abs(samples-truth),axis=0)-.5*np.mean(np.abs(samples[:,None]-samples[None,:]),axis=(0,1))).mean())


def evaluate_job(job):
    from .learning_models import predict_coefficients
    from .variational import infer_vi, bounded_gaussian_mean
    run,split,parent,repeat=Path(job[0]),job[1],job[2],int(job[3])
    options=job[4] if len(job)>4 else {}
    out=Path(options.get('output',run/'evaluation'/split/parent/f'repeat_{repeat}'))
    if (out/'result.json').exists() and not options.get('force',False):return read_json(out/'result.json')
    out.mkdir(parents=True,exist_ok=True)
    spec=read_json(run/'study.json');frozen=read_json(run/'frozen.json') if (run/'frozen.json').exists() else spec
    trf=read_json(run/'calibration/trf.json')['selected']
    vi_settings=read_json(run/'calibration/vi.json')['selected']
    size=int(frozen.get('selected_training_size',128))
    seeds=options.get('training_seeds',spec['training_seeds'])
    windows=options.get('windows',WINDOWS);channels_list=options.get('channels',CHANNELS)
    actions=options.get('actions',spec['actions']);method_subset=options.get('methods')
    trace,observed,noise_seed=trace_and_observations(run,split,parent,repeat)
    model=SpatialModel(Config(**trace.config));observer=Observer(model);basis=observer.basis
    set_shared_commands(model,trace)
    model._cache_forecasts=options.get('cache_forecasts',True)
    truth={action:_forecast(model,trace.checkpoint(),action) for action in actions}
    # A common future measurement draw is part of the test case, not the method.
    measurement_rng=np.random.default_rng(int(noise_seed)+7_000_019)
    noisy_truth={action:target.force+measurement_rng.normal(0,NOISE_STD[2],target.force.shape)
                 for action,target in truth.items()}
    rows=[];failures=[];diagnostics=[]
    started=time.perf_counter()
    for window in windows:
        for channels in channels_list:
            p=WindowProblem(model,trace,observed,window,channels=='lateral',basis)
            prior=training_prior(run,window,channels)
            projected,projection=p.projection();context={'history_id':parent,'split':split,'noise_repeat':repeat,
                'noise_seed':noise_seed,'channels':channels,'window':str(window), 'training_size':0,
                'observation_sha256':hashlib.sha256(observed.tobytes()).hexdigest(),**p.record()}
            estimates=[]
            def want(name):return method_subset is None or name in method_subset
            def add_fit(name,coefficients,diag,seed=-1,samples=None):
                t=time.perf_counter();state=p.current(coefficients);history=time.perf_counter()-t
                estimates.append((name,seed,state,diag,history,samples,coefficients))
            if want('TRF calibrated'):
                try:
                    c,d=fit_trf(p,trf,prior);add_fit('TRF calibrated',c,d)
                except Exception as exc:failures.append({**context,'method':'TRF calibrated','error':str(exc)})
            if want('TRF published'):
                try:
                    t=time.perf_counter()
                    legacy=Observer(model,replace(ObserverConfig(),use_lateral=channels=='lateral'))
                    if split=='old' and not options.get('remeasure_published',False):
                        old=ROOT/spec['old_run'];key=channels+'_'+str(window)
                        with np.load(old/'test'/parent/f'repeat_{repeat}'/'estimates.npz') as data:
                            tool=data[key+'_tool'];state=State(data[key+'_u'],data[key+'_z'],*tool)
                        source=read_json(old/'test'/parent/f'repeat_{repeat}'/'result.json')
                        diag=next(r for r in source['rows'] if r['method']==key)['inference']
                        seconds=next(r for r in source['rows'] if r['method']==key)['inference_seconds']
                        diag={**diag,'fit_seconds':seconds,'timing_source':'archived parallel batch; remeasured in dedicated latency suite'}
                    else:
                        old_estimate=infer_condition(legacy,trace,observed,float(window) if window!='contact' else 'contact',spec)
                        state=old_estimate.state;diag={**old_estimate.diagnostics,'fit_seconds':time.perf_counter()-t}
                    estimates.append(('TRF published',-1,state,diag,0.,None,None))
                except Exception as exc:failures.append({**context,'method':'TRF published','error':str(exc)})
            if want('Projected truth'):add_fit('Projected truth',projected,{'fit_seconds':0.,'privileged':'full internal field projection at same window start',**projection})
            if want('True state'):estimates.append(('True state',-1,trace.checkpoint(),{'fit_seconds':0.,'privileged':'full internal fields at cut'},0.,None,None))
            if want('Full replay'):
                t=time.perf_counter();full=observer.infer(trace,'full',observed)
                estimates.append(('Full replay',-1,full.state,{'fit_seconds':0.,'privileged':'known relaxed initial state'},time.perf_counter()-t,None,None))
            condition_seeds=seeds[:1] if str(window)=='contact' else seeds
            for seed in condition_seeds:
                predicted={}
                for kind,label in [('gru','GRU'),('rme','RME prior'),('mlp','Snapshot MLP')]:
                    if kind=='mlp' and str(window)!='0.0':continue
                    needed=want(label) or (kind=='rme' and any(want(n) for n in ['RME prior mean','RME-inspired VI','Neural-init TRF']))
                    if not needed:continue
                    t=time.perf_counter();neural=model_for(run,kind,seed,size);x,length=p.inputs()
                    c=predict_coefficients(neural,x[None],np.array([length]))[0].astype(float);encode=time.perf_counter()-t
                    predicted[kind]=(c,neural,encode)
                    if want(label):add_fit(label,c,{'encode_seconds':encode,'fit_seconds':0.,'model':str(run/'models'/f'{kind}_n{size}_s{seed}.pt')},seed)
                if 'rme' in predicted:
                    c,neural,encode=predicted['rme']
                    scale=np.asarray(neural.report['prior_logit_scale'])*vi_settings['scale_multiplier']
                    if want('RME prior mean'):
                        t=time.perf_counter()
                        mean=bounded_gaussian_mean(logit(np.clip(c/2.5,1e-6,1-1e-6)),scale)
                        mean_seconds=time.perf_counter()-t
                        add_fit('RME prior mean',mean,{'encode_seconds':encode+mean_seconds,'fit_seconds':0.,'prior_scale':scale.tolist(),'point_summary_control':True},seed)
                    if want('RME-inspired VI'):
                        try:
                            sampling_seed=int(np.random.SeedSequence([20261030,int(parent.split('_')[-1]),repeat,seed,WINDOWS.index(str(window)),int(channels=='lateral')]).generate_state(1)[0])
                            result=infer_vi(p,c,scale,sampling_seed,steps=vi_settings['steps'],lr=vi_settings['lr'],posterior_samples=spec['posterior_samples'])
                            diag={**result,'encode_seconds':encode,'vi_seconds':result['seconds'],'fit_seconds':0.,'prior_kind':'neural data-conditioned; no calibration guarantee'}
                            samples=result['coefficient_samples'] if (split in ('test','pilot') and [channels,str(window)] in spec['probability_conditions']) else None
                            add_fit('RME-inspired VI',result['coefficients'],diag,seed,samples)
                        except Exception as exc:failures.append({**context,'method':'RME-inspired VI','training_seed':seed,'error':str(exc),'traceback':traceback.format_exc()})
                    representative=(str(window)=='contact' or (channels=='basic' and str(window)=='0.2'))
                    if want('Neural-init TRF') and representative:
                        try:
                            fitted,diag=fit_trf(p,trf,prior,initial=c);diag['encode_seconds']=encode
                            add_fit('Neural-init TRF',fitted,diag,seed)
                        except Exception as exc:failures.append({**context,'method':'Neural-init TRF','training_seed':seed,'error':str(exc)})
                if want('Fixed prior VI') and seed==condition_seeds[0] and [channels,str(window)] in spec['probability_conditions']:
                    try:
                        result=infer_vi(p,prior['fixed_center'],prior['latent_sd']*.3,int(noise_seed)+int(seed),steps=vi_settings['steps'],lr=vi_settings['lr'],posterior_samples=spec['posterior_samples'])
                        diag={**result,'encode_seconds':0.,'vi_seconds':result['seconds'],'fit_seconds':0.,'prior_kind':'fixed training-label distribution'}
                        samples=result['coefficient_samples'] if split in ('test','pilot') else None
                        add_fit('Fixed prior VI',result['coefficients'],diag,-1,samples)
                    except Exception as exc:failures.append({**context,'method':'Fixed prior VI','training_seed':seed,'error':str(exc)})
            for name,seed,state,diag,history,samples,c in estimates:
                key=f"{channels}_{window}_{name.replace(' ','_')}_{seed}"
                serial_diag={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in diag.items()}
                diagnostics.append({**context,'method':name,'training_seed':seed,'inference':serial_diag,'state_sha256':state_hash(state)})
                for action in actions:
                    try:
                        t=time.perf_counter();future=_forecast(model,state,action);predict_seconds=time.perf_counter()-t
                        target=truth[action];prob={};extra={};mc_history=mc_prediction=0.
                        if samples is not None and action=='Slow press':
                            force_samples,marker_samples,mc_history,mc_prediction,mc_audits=posterior_curves(p,samples,action)
                            lower,upper=np.quantile(force_samples,[.05,.95],axis=0)
                            rng=np.random.default_rng(int(noise_seed)+int(seed)+100003+(action=='Fast press'))
                            noisy=force_samples+rng.normal(0,NOISE_STD[2],force_samples.shape)
                            nl,nu=np.quantile(noisy,[.05,.95],axis=0)
                            noisy_target=noisy_truth[action]
                            contact_target=target.force>spec['contact_threshold_N']
                            cl=2.5*expit(np.asarray(diag['q_mu'])-1.644853626951*np.exp(diag['q_log_sigma']))
                            cu=2.5*expit(np.asarray(diag['q_mu'])+1.644853626951*np.exp(diag['q_log_sigma']))
                            prob={'force_coverage90':float(np.mean((target.force>=lower)&(target.force<=upper))),
                                  'force_interval_width_mN':float(np.mean(upper-lower)*1000),
                                  'force_crps_mN':crps(force_samples,target.force)*1000,
                                  'force_contact_coverage90':float(np.mean(((target.force>=lower)&(target.force<=upper))[contact_target])) if contact_target.any() else None,
                                  'force_contact_interval_width_mN':float(np.mean((upper-lower)[contact_target])*1000) if contact_target.any() else None,
                                  'noisy_force_coverage90':float(np.mean((noisy_target>=nl)&(noisy_target<=nu))),
                                  'noisy_force_interval_width_mN':float(np.mean(nu-nl)*1000),
                                  'noisy_force_crps_mN':crps(noisy,noisy_target)*1000,
                                  'coefficient_coverage90':float(np.mean((projected>=cl)&(projected<=cu))),
                                  'posterior_prior_logit_sd_ratio':float(np.mean(np.exp(diag['q_log_sigma'])/diag['prior_scale'])),
                                  'posterior_prior_mean_shift':float(np.linalg.norm(np.asarray(diag['q_mu'])-diag['prior_mu'])),
                                  'final_kl':float(.5*np.sum((np.exp(2*np.asarray(diag['q_log_sigma']))+
                                      (np.asarray(diag['q_mu'])-diag['prior_mu'])**2)/np.asarray(diag['prior_scale'])**2-1+
                                      2*(np.log(diag['prior_scale'])-diag['q_log_sigma'])))}
                            prob['posterior_physics_violation_rate']=float(np.mean([not a['passed'] for a in mc_audits]))
                            extra={'force_lower':lower,'force_upper':upper,'noisy_force_lower':nl,'noisy_force_upper':nu,'coefficient_samples':samples,
                                   'force_samples':force_samples,'noisy_force_samples':noisy,'noisy_force_target':noisy_target,
                                   'posterior_physics_passed':np.array([a['passed'] for a in mc_audits])}
                        relative=Path('curves')/f"{key}_{action.replace(' ','_')}.npz"
                        (out/'curves').mkdir(exist_ok=True)
                        np.savez_compressed(out/relative,time=future.time,command=future.command,true_force=target.force,predicted_force=future.force,
                                            true_markers=target.markers,predicted_markers=future.markers,**extra)
                        fit=float(diag.get('fit_seconds',0.));encode=float(diag.get('encode_seconds',0.));vi=float(diag.get('vi_seconds',0.))
                        row={**context,'method':name,'training_seed':seed,'training_size':size if seed!=-1 else 0,'action':action,
                             'force_rmse_mN':float(np.sqrt(np.mean((future.force-target.force)**2))*1000),
                             'marker_rmse_um':float(np.sqrt(np.mean((future.markers-target.markers)**2))*1e6),
                             'encode_seconds':encode,'history_seconds':fit+history+mc_history,'vi_seconds':vi,
                             'predict_seconds':predict_seconds+mc_prediction,'total_seconds':encode+fit+history+mc_history+vi+predict_seconds+mc_prediction,
                             'curve_path':str((out/relative).relative_to(run)).replace('\\','/'),'success':True,
                             'optimizer_status':diag.get('optimizer_status','not applicable'),
                             'optimizer_success':diag.get('optimizer_success','not claimed for fixed-budget VI'),
                             'state_sha256':state_hash(state),'posterior_samples':0 if not prob else len(samples),
                             'command_sha256':hashlib.sha256(future.command.tobytes()).hexdigest(),
                             'variational_seed':diag.get('seed',''),
                             'physics_audit_scope':'Command-driven future only; generated loading histories audited separately; inferred replay paths not certified',
                             'timing_scope':'parallel evaluation wall time; dedicated batch1 latency separately measured',
                             'physics':audit_trace(future,PHYSICS_LIMITS),**prob}
                        rows.append(row)
                    except Exception as exc:failures.append({**context,'method':name,'training_seed':seed,'action':action,'error':str(exc),'traceback':traceback.format_exc()})
    record={'history_id':parent,'split':split,'repeat':repeat,'rows':rows,'failures':failures,
            'source_sha256':_CODE_AT_IMPORT,'wall_seconds':time.perf_counter()-started}
    write_json(out/'inference.json',diagnostics)
    write_json(out/'result.pending.json',record)
    (out/'result.pending.json').replace(out/'result.json')
    return record


def parallel(function,jobs,workers):
    records=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(function,job):job for job in jobs}
        for future in as_completed(futures):
            record=future.result();records.append(record)
            print(f'{len(records)}/{len(jobs)} completed',flush=True)
    return records


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    names=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=names);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def collect(run):
    rows=[];failures=[]
    for path in sorted((run/'evaluation').rglob('result.json')):
        record=read_json(path);rows.extend(record['rows']);failures.extend(record['failures'])
    failed_rows=[{**f,'success':False,'action':f.get('action','inference failure'),
                  'training_seed':f.get('training_seed',-1),'training_size':0,
                  'force_rmse_mN':None,'marker_rmse_um':None,'total_seconds':None,'curve_path':''}
                 for f in failures]
    write_csv(run/'metrics.csv',rows+failed_rows);write_json(run/'failures.json',failures)
    return rows


def freeze(run,spec,count=None):
    if (run/'frozen.json').exists():raise ValueError('Frozen test already exists; do not change after inspection')
    if read_json(run/'training.json').get('status')!='completed':raise ValueError('Finish training before freezing the independent test')
    from .inference_analysis import paired_values
    pilot=[read_json(path) for path in sorted((run/'evaluation/pilot').rglob('result.json'))]
    rows=[r for record in pilot for r in record['rows'] if r['action']=='Slow press']
    if len(pilot)<4:raise ValueError('At least four independent pilot histories must finish before freeze')
    def selected(method,channels,window):
        return [r for r in rows if r['method']==method and r['channels']==channels and r['window']==window]
    main_methods=('TRF calibrated','GRU','RME-inspired VI')
    definitions=[(method,'basic','0.0',method,'lateral','0.0') for method in main_methods]
    definitions += [(method,'lateral','0.0',method,'lateral',window)
                    for method in main_methods for window in ('0.2','0.8')]
    definitions += [('RME prior mean','basic','0.2','RME-inspired VI','basic','0.2')]
    precision=[];pilot_gains={}
    for a,ca,wa,b,cb,wb in definitions:
        pairs,_=paired_values(selected(a,ca,wa),selected(b,cb,wb))
        gains=np.array([p['gain_mN'] for p in pairs])
        pilot_gains[(a,ca,wa,b,cb,wb)]={p['history_id']:p['gain_mN'] for p in pairs}
        if len(gains)>1:
            sd=float(gains.std(ddof=1))
            precision.append({'left':[a,ca,wa],'right':[b,cb,wb],
                              'pilot_independent_histories':len(gains),'paired_sd_mN':sd,
                              'required_n_for_target':max(2,int(np.ceil((1.96*sd/.5)**2)))})
    for method in ('GRU','RME-inspired VI'):
        for ca,wa,cb,wb in [('basic','0.0','lateral','0.0'),('lateral','0.0','lateral','0.2'),('lateral','0.0','lateral','0.8')]:
            learned=pilot_gains[(method,ca,wa,method,cb,wb)]
            classical=pilot_gains[('TRF calibrated',ca,wa,'TRF calibrated',cb,wb)]
            gains=np.array([learned[h]-classical[h] for h in sorted(learned.keys()&classical.keys())])
            if len(gains)>1:
                sd=float(gains.std(ddof=1))
                precision.append({'contrast':'Difference of information gains','method':method,'reference':'TRF calibrated',
                                  'left':[ca,wa],'right':[cb,wb],'pilot_independent_histories':len(gains),
                                  'paired_sd_mN':sd,'required_n_for_target':max(2,int(np.ceil((1.96*sd/.5)**2)))})
    if not precision:raise ValueError('Pilot paired comparisons are incomplete')
    proposed=max(p['required_n_for_target'] for p in precision)
    automatic=min(24,max(12,proposed))
    count=automatic if count is None else int(count)
    for item in precision:
        item['planned_95_halfwidth_mN']=1.96*item['paired_sd_mN']/np.sqrt(count)
    job_wall=float(np.median([p['wall_seconds'] for p in pilot]))
    record={**spec,'test_seeds':list(range(20000,20000+count)),'selected_training_size':128,
            'frozen_utc':datetime.now(timezone.utc).isoformat(),'code':environment(),
            'learning_packages':{n:importlib.metadata.version(n) for n in ('torch','threadpoolctl')},
            'pilot_independent_histories':len(pilot),'paired_precision_planning':precision,
            'target_halfwidth_mN':.5,'minimum_n':12,'maximum_small_batch_n':24,'uncapped_precision_n':proposed,
            'precision_target_met_in_planning':all(p['planned_95_halfwidth_mN']<=.5 for p in precision),
            'pilot_median_job_seconds':job_wall,'conservative_test_core_seconds':job_wall*count*9,
            'sample_size_reason':'Independent pilot paired SD, descriptive 0.5 mN interval target (one third of force-noise SD), bounded to12--24 complete histories using measured cost; not a task tolerance, equivalence margin or power certification',
            'analysis_rule':'Pointwise whole-history percentile bootstrap; primary contrasts from study.json; no equivalence or VI-over-MAP claim; no post-test tuning',
            'trf':read_json(run/'calibration/trf.json'),'vi':read_json(run/'calibration/vi.json')}
    artifacts=[run/'training.json',run/'physics.json',run/'study.json']
    artifacts+=list((run/'models').glob('*.pt'))+list((run/'datasets').glob('*.npz'))
    record['artifacts_sha256']={str(p.relative_to(run)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(artifacts)}
    write_json(run/'frozen.json',record)
    for source in (ROOT/'src/nr2d').glob('*.py'):
        destination=run/'source/frozen'/source.name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
    return record


def verify(run):
    """Check only scientific invariants that could invalidate the comparison."""
    sets={}
    for name in ('train','validation','calibration','pilot','test'):
        path=run/'datasets'/f'{name}.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as data:
                sets[name]={int(parent.split('_')[-1]) for parent in data['ids'].tolist()}
    overlap=[(a,b) for a in sets for b in sets if a<b and sets[a]&sets[b]]
    trace,observed,_=trace_and_observations(run,'pilot','pilot_14000',0)
    model=SpatialModel(Config(**trace.config));basis=Observer(model).basis
    altered=observed.copy();altered[:,3:]=np.nan
    a=WindowProblem(model,trace,observed,.2,False,basis)
    b=WindowProblem(model,trace,altered,.2,False,basis)
    c=training_prior(run,'0.2','basic')['mean']
    ra,ja=a.residual(c,jacobian=True);rb,jb=b.residual(c,jacobian=True)
    closed=np.array_equal(a.inputs()[0],b.inputs()[0]) and np.array_equal(ra,rb) and np.array_equal(ja,jb)
    # Across methods a command is fixed; across futures a fitted state is fixed.
    command_groups={};state_groups={};noise_groups={};noise_seeds={};source_changes=[]
    frozen=read_json(run/'frozen.json') if (run/'frozen.json').exists() else None
    numeric=('physics.py','protocols.py','estimators.py','inference_physics.py','inference_study.py','inference_data.py','learning_models.py','variational.py')
    for path in (run/'evaluation').rglob('result.json'):
        result=read_json(path)
        if frozen and result['split']=='test':
            source_changes.extend([str(path)+':'+name for name in numeric
                                   if result.get('source_sha256',{}).get(name)!=frozen['code']['source_sha256'][name]])
        for row in result['rows']:
            common=(row['split'],row['history_id'],row['noise_repeat'])
            noise_seeds[common]=row['noise_seed']
            command_groups.setdefault(common+(row['action'],),set()).add(row['command_sha256'])
            state_groups.setdefault(common+(row['channels'],row['window'],row['method'],row['training_seed']),set()).add(row['state_sha256'])
            noise_groups.setdefault(common,set()).add(row['observation_sha256'])
    record={'disjoint_history_splits':not overlap,'overlap':overlap,'split_counts':{k:len(v) for k,v in sets.items()},
            'closed_lateral_inputs_residual_and_jacobian_invariant':closed,
            'shared_future_commands':all(len(v)==1 for v in command_groups.values()),
            'same_estimate_across_future_actions':all(len(v)==1 for v in state_groups.values()),
            'same_noisy_observations_across_methods':all(len(v)==1 for v in noise_groups.values()),
            'independent_noise_seed_identifiers':len(set(noise_seeds.values()))==len(noise_seeds),
            'frozen_test_numerics_unchanged':not source_changes,'changed_source_records':source_changes,
            'evaluated_command_groups':len(command_groups),'evaluated_state_groups':len(state_groups),
            'source':environment()}
    record['passed']=all(record[k] for k in ('disjoint_history_splits','closed_lateral_inputs_residual_and_jacobian_invariant',
                                           'shared_future_commands','same_estimate_across_future_actions','same_noisy_observations_across_methods',
                                           'independent_noise_seed_identifiers','frozen_test_numerics_unchanged'))
    write_json(run/'verification/comparison.json',record)
    if not record['passed']:raise AssertionError(record)
    print(json.dumps({k:v for k,v in record.items() if k!='source'},indent=2))
    return record


def recompute_comparison(run,history_id,noise_repeat,training_seed,channels,window,action,methods):
    run=Path(run)
    split='old' if history_id.startswith('test_5') else history_id.split('_')[0]
    out=run/'interactive'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    record=evaluate_job((str(run),split,history_id,int(noise_repeat),
                         {'output':str(out),'windows':[str(window)],'channels':[channels],
                          'actions':[action],'training_seeds':[int(training_seed)],'methods':list(methods),'force':True}))
    # Rebase paths to the standalone interactive output for the Notebook reader.
    for row in record['rows']:row['curve_path']=str((run/row['curve_path']).relative_to(out)).replace('\\','/')
    write_csv(out/'metrics.csv',record['rows'])
    return {'run':out,'rows':record['rows']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['calibrate-trf','calibrate-vi','evaluate','collect','freeze','data','train','summarize','latency','verify','figures'])
    parser.add_argument('--run',type=Path);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--split',default='pilot',choices=['pilot','old','test','train','validation','calibration'])
    parser.add_argument('--count',type=int);parser.add_argument('--limit',type=int)
    args=parser.parse_args();run,spec=load_run(args.run)
    stage_started=time.perf_counter();stage_utc=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    if args.stage=='calibrate-trf':calibrate_trf(run,spec,args.workers)
    elif args.stage=='calibrate-vi':calibrate_vi(run,spec,args.workers)
    elif args.stage=='freeze':freeze(run,spec,args.count)
    elif args.stage=='data':
        seeds=read_json(run/'frozen.json')['test_seeds'] if args.split=='test' else spec[args.split+'_seeds']
        prepare_dataset(run,args.split,seeds,spec['noise_repeats'],args.workers)
    elif args.stage=='evaluate':
        if args.split=='old':parents=[f'test_{s}' for s in range(5000,5024)]
        elif args.split=='test':parents=[f'test_{s}' for s in read_json(run/'frozen.json')['test_seeds']]
        else:parents=[f'{args.split}_{s}' for s in spec[args.split+'_seeds']]
        if args.limit:parents=parents[:args.limit]
        options={'training_seeds':[11]} if args.split=='pilot' else {}
        jobs=[(str(run),args.split,parent,repeat,options) for parent in parents for repeat in range(spec['noise_repeats'] if args.split!='pilot' else 1)]
        parallel(evaluate_job,jobs,args.workers);collect(run)
    elif args.stage=='collect':collect(run)
    elif args.stage=='train':
        from .learning_models import train_study
        train_study(run,workers=args.workers)
    elif args.stage in ('summarize','latency'):
        from .inference_analysis import summarize,latency_suite
        (summarize if args.stage=='summarize' else latency_suite)(run)
    elif args.stage=='verify':verify(run)
    elif args.stage=='figures':
        from .inference_display import export_inference_figures
        export_inference_figures(run)
    write_json(run/'execution'/f'{stage_utc}_{args.stage}_{args.split}.json',
               {'stage':args.stage,'split':args.split,'workers':args.workers,'status':'completed',
                'wall_seconds':time.perf_counter()-stage_started,'source_sha256':_CODE_AT_IMPORT})


if __name__=='__main__':main()
