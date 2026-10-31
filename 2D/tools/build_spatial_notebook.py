"""Maintain the skill-scaffolded notebook without duplicating research code."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
path = ROOT/"notebooks/spatial_2d_hidden_state.ipynb"
nb = nbf.read(path, as_version=4)
M, C = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
M(r"""# Spatial 2D object hidden state

Explore how loading history changes internal state and the response to a later action.

Open the project with **Dev Containers: Reopen in Container**, select the container's **Python 3.11.9**, and **Run All**.
The default computes a fresh main trajectory and replays saved checks and predictions.
Use **Recompute** for a new calculation; the timeline replays the trajectory.

[Estimator comparison](#6.-Does-estimator-choice-change-the-value-of-information?) | [Sensors and contact](#5.-Sensor-channels-and-contact-history) | [Docker setup](../docs/CONTAINER.md) | [Pilot results](../runs/2d_contact/pilot_c9dd23016896/RESULTS.md)
"""),
C("""from pathlib import Path
import os
import sys
import time

ROOT = next((p for base in [Path.cwd(), *Path.cwd().parents]
             for p in (base, base / '2D') if (p / 'src/nr2d/physics.py').exists()), None)
if ROOT is None:
    raise RuntimeError('Open the repository root in VS Code and run from this project.')
sys.path.insert(0, str(ROOT / 'src'))
get_ipython().run_line_magic('matplotlib', 'widget')

import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Image
from nr2d.physics import Config, SpatialModel, Trace
from nr2d.estimators import ObserverConfig
from nr2d.protocols import protocol
from nr2d.experiments import run_pilot, read_json, write_json, forecasts_for, sweep_for
from nr2d.plotting import archived_history_figure, paired_figure, export_figure
from nr2d.widgets import ReplayDashboard, ForecastDashboard, SweepDashboard, InformationDashboard, checks_table, matching_table, table
import ipywidgets as widgets

NOTEBOOK_START = time.perf_counter()
CONFIG_PATH = ROOT / 'configs/spatial_contact.json'
config = Config(**read_json(CONFIG_PATH))
OBSERVER_CONFIG_PATH = ROOT / 'configs/observer.json'
observer_config = ObserverConfig(**read_json(OBSERVER_CONFIG_PATH))
REPRESENTATIVE_RUN = ROOT / 'runs/2d_contact/pilot_c9dd23016896'
RECOMPUTE_PILOT = False
print('Python:', sys.executable)
print('Project:', ROOT)
"""),
M(r"""## 0. Existing evidence

The archived **history-dependence** study motivates the spatial experiment. Classical observers were competitive;
the shared-feedback study found no useful action benefit. The remaining sections use the 2D solver.
"""),
C("""from io import BytesIO
with plt.ioff():
    archived_fig, archived_source = archived_history_figure()
buffer = BytesIO()
archived_fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
display(Image(data=buffer.getvalue()))
plt.close(archived_fig)
print('Archived source:', archived_source.relative_to(ROOT.parent))
"""),
M(r"""## 1. Physical process: load, hold, release, wait, recontact

A circular indenter presses a 40 x 20 mm plane-strain body with a fixed bottom and free sides.
The FEM body is quasistatic; the tool has inertia and a compliant, damped actuator. Lateral displacement comes from the coupled solve.

The synthetic two-mode material follows

$$
\sigma = C_\infty\epsilon + \sum_{i\in\{f,s\}} C_i(\epsilon-z_i),\qquad
\dot z_i=(\epsilon-z_i)/\tau_i.
$$

The physical state contains internal-strain fields $z_f,z_s$ and tool position/velocity, with fixed material and geometry.
Displacement and contact follow from equilibrium. Command, achieved motion and contact reaction are recorded separately at 100 Hz.
"""),
C("""reuse_histories = (
    not RECOMPUTE_PILOT
    and config.asdict() == read_json(REPRESENTATIVE_RUN / 'config.json')
)
if reuse_histories:
    run = REPRESENTATIVE_RUN
    runtime = {'mode': 'REPLAY: published pilot', 'wall_seconds': 0.0}
else:
    run, runtime = run_pilot(config, force=RECOMPUTE_PILOT, progress=print,
                            observer_config=observer_config)
replay_published = reuse_histories and observer_config.asdict() == ObserverConfig().asdict()
comparison_config = None if replay_published else observer_config
manifest = read_json(run / 'manifest.json')
report = read_json(run / 'checks.json')
from datetime import datetime
commands, phases, _ = protocol(Config(**read_json(run / 'config.json')))
main = SpatialModel(Config(**read_json(run / 'config.json'))).simulate(
    commands, phases, metadata={'source': 'Fresh notebook spatial 2D solve', 'role': 'demonstration'})
live_dir = run / 'notebook_runs' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
main.save(live_dir / 'main.npz')
write_json(live_dir / 'execution.json', {'solver_seconds': main.wall_seconds, 'source': 'fresh main episode',
                                      'config': main.config, 'verification_cache': str(run / 'checks.json')})
print('Mode:', runtime['mode'])
print('This numerical entry call: %.2f s; recorded cold pilot: %.2f s' %
      (runtime['wall_seconds'], manifest['wall_seconds']))
print('Run:', run)
print('Synthetic parameters:', main.config)
print('FRESH main spatial solve: %.2f s; verification table below replays the checked numerical pilot.' % main.wall_seconds)
display(checks_table(report))
"""),
C("""replay = ReplayDashboard(main)
display(replay.widget)
"""),
M(r"""**Reading the animation.** Geometry uses 1x displacement scale. Color shows slow internal strain (simulation truth),
orange points show contact, and teal points show lateral sensors. The three curves share the mesh's time cursor.
After separation, contact force is zero while internal strain and lateral shape continue recovering.

The check table covers relaxation/recovery references, an elastic control, force balance, contact and numerical refinement.
"""),
M(r"""## 2. Why the current observation may be insufficient

Sections 2-4 retain the original eight-history pilot. Section 5 uses new histories and independent noise repeats.

Each history starts relaxed and follows a feasible command sequence. Pair selection uses current tool position, velocity,
force and both lateral markers, taking the first accepted pair in fixed ID order. Future response is excluded from selection.
Material, geometry, tool, current command and elapsed time are shared.

The table reports matching residuals. The 5 um lateral band is a matching tolerance; estimators still receive raw marker values
and can use differences within it. The pair demonstrates similar endpoints, with some instantaneous shape information remaining.
"""),
C("""matching = read_json(run / 'matching.json')
display(matching_table(run))
print('Accepted / attempted:', matching['accepted'], '/', matching['attempted'])
print('Independent test histories:', manifest['independent_test_history_count'])
with plt.ioff():
    pair_fig = paired_figure(run)
buffer = BytesIO()
pair_fig.savefig(buffer, format='png', dpi=130, bbox_inches='tight')
display(Image(data=buffer.getvalue()))
plt.close(pair_fig)
refinement = read_json(run / 'pair_refinement.json')
print('Future force-contrast RMS: %.3f mN (default), %.3f mN (fine mesh)' %
      (refinement['default']['force_contrast_rms_N']*1e3, refinement['fine']['force_contrast_rms_N']*1e3))
print('Contrast change under mesh refinement: %.3f mN RMS' % (refinement['contrast_discretization_rms_N']*1e3))
"""),
M(r"""## 3. One checkpoint, several future actions

Choose History A/B and a future action, then press **Run selected future**.
Each method receives the same future command and known model parameters. Hold/Release at this detached checkpoint give zero contact force.

| Method | Information and approximation |
|---|---|
| Snapshot | Current position, velocity, force and two markers; six-coefficient initial-strain basis, minimum-norm fit |
| Short classical | The last 0.2 s; same sensors, fitting basis and optimizer cap |
| Full classical | Complete measured motion and a known relaxed start; exact-model replay |
| True state | Complete internal fields from simulation; reference for the selected future command |

Predictions use past measurements and future commands. Snapshot/window fits use force and lateral observations;
full replay uses measured motion with its known initial state. Inference time is reported because the algorithms have different costs.
"""),
C("""forecast = ForecastDashboard(run, observer_config=comparison_config)
display(forecast.widget)
"""),
M(r"""## 4. How much history is needed?

Compare 0, 0.04, 0.1, 0.2, 0.4, 0.8 s and full history on validation histories.
Synthetic nominal noise is 0.5 um position, 0.01 mm/s velocity, 1.5 mN force and 1 um per marker,
with the same realization used across methods. Samples are synchronous. Pulse excitation changes the loading commands.

Select excitation and noise, then run the sweep. Inspect the curve and fit status to see when extra history helps.
"""),
C("""sweep = SweepDashboard(run, observer_config=comparison_config)
display(sweep.widget)
"""),
M(r"""**Noise changes the result.** Short windows can perform worse than snapshot under nominal noise.
The plots use two validation histories; `RESULTS.md` reports the batch over all eight test histories.
"""),
C("""noise_preview = run / 'figures/05_noise_and_excitation.png'
if noise_preview.exists():
    display(Image(filename=str(noise_preview)))
else:
    print('Use the noise/excitation dropdowns above to compute and inspect each condition.')
audit_path = run / 'physical_failures.json'
if audit_path.exists():
    audit_failures = read_json(audit_path)
    for item in audit_failures:
        print('RETAINED physical overrun:', item['path'], 'penetration %.4f um > 5 um' % item['max_penetration_um'])
        if 'diagnostic' in item:
            print('Diagnostic doubled penalty: %.4f um; original is preserved, diagnostic excluded from tests.' %
                  item['diagnostic']['max_penetration_um'])
"""),
M(r"""## 5. Sensor channels and contact history

This comparison fixes the material, mesh, contact solver, prediction time and future commands.
Three calibration histories select one shared observer setting before the 24 test histories are evaluated.
Each test history has three independent nominal-noise draws; methods share each draw.

| Available information | Current time | Recent history |
|---|---|---|
| Position, velocity and total force | Snapshot without lateral | 0.2 or 0.8 s without lateral |
| The same channels plus two lateral markers | Snapshot with lateral | 0.2 or 0.8 s with lateral |

Position drives the material replay; force and enabled lateral channels enter the residual.
Measured velocity initializes the prediction. Disabled lateral values are excluded from every estimate.
All finite-window fits use the same six-coefficient basis and optimizer budget. Longer windows cost more solver time.

**Earlier contact** selects the last continuous 0.2 s interval with measured force above 4.5 mN.
After that update, measured position propagates the estimate to the common prediction time.
The selector reads the full prior force trace; this is a longer information span than a recent 0.2 s window.
The interval table below records exactly what was used.

Dropdowns prepare a selection. **Run selection** loads cached predictions; checking **Recompute** repeats inference
and both future rollouts from the saved physical history. Switching actions reuses the same inferred state.
"""),
C("""INFORMATION_RUN = ROOT / 'runs/information/channels_contact_20261026'
information_summary = read_json(INFORMATION_RUN / 'summary.json')
calibration = read_json(INFORMATION_RUN / 'calibration.json')
print('Test histories:', information_summary['completed_histories'])
print('Noise jobs:', information_summary['completed_jobs'], '/', information_summary['expected_jobs'])
print('Shared preset:', calibration['selected'], '| selected on three calibration histories')
print('Optimization nonconvergence:', information_summary['optimizer_nonconvergence'])
print('Physical-limit records:', len(information_summary['physical_failures']))
information = InformationDashboard(INFORMATION_RUN)
display(information.widget)
"""),
M(r"""**Duration and contact content.** The 0.2 s recent window lies in the detached wait.
The timeline separates command phases, force-detected contact, observation updates and propagation.
The lower timeline is the first prespecified test history; error bars summarize all 24 histories.
"""),
C("""display(Image(filename=str(INFORMATION_RUN / 'figures/02_window_and_contact.png')))
display(Image(filename=str(INFORMATION_RUN / 'figures/03_paired_improvements.png')))
display(Markdown('Three noise repeats are averaged within each history. Intervals resample complete histories; '
                 'they are exploratory pointwise 95% intervals. Fast press uses the same estimate as Slow press.'))
"""),
M(r"""## Findings and next experiment

These recorded sensor-ablation results use the published finite-window observer. Section 6 checks how the
information gains depend on the estimator and its prior; these numbers are not estimator-independent limits.

- Current lateral shape reduces mean Slow-press force RMSE from 11.50 to 2.20 mN. It provides substantial information, but history still improves prediction.
- Recent 0.2 s lateral history reduces error to 1.29 mN even though this interval contains no contact. Recovery measurements retain useful information.
- Without lateral sensing, the recent wait provides no contact signal and the coefficients remain at initialization. Earlier 0.2 s contact plus motion propagation gives 0.48 mN; the recent 0.8 s window gives 0.31 mN.
- Using both channels over 0.8 s gives 0.23 mN. The same estimates also improve Fast-press prediction, without another fit.

The earlier eight-history pilot's negative noisy-short-window result is retained; it does not generalize to this independent
24-history, three-noise-draw batch. The new [results](../runs/information/channels_contact_20261026/RESULTS.md) report all paired intervals.
There were 201 optimizer-budget stops among 576 finite-window fits and one retained physical-history tolerance overrun.
Section 6 compares calibrated TRF, learned state estimates and variational refinement on this question.
A continuously updated recursive observer that carries the contact estimate through the wait remains untested.

These remain small-strain, same-material prediction experiments. Reusing a state across two commanded futures does not establish
a control benefit or a universally sufficient state. The 2D common-feedback action comparison remains untested;
first check whether true-state access can improve the selected task.

**Three-minute presentation:** physical recovery -> channel-ablation curves -> window/contact timeline -> paired gains and remaining estimator limits.
"""),
C("""metrics = forecasts_for(run, observer_config=comparison_config)[1]
display(widgets.HTML(table(['Default selected test parent', 'Force RMSE (mN)', 'Lateral RMSE (um)'],
    [[name, '%.4g' % (m['force_rmse_N']*1e3), '%.4g' % (m['marker_rmse_m']*1e6)]
     for name, m in metrics.items()])))
print('Notebook body elapsed: %.2f s (cached pilot and widgets included)' % (time.perf_counter()-NOTEBOOK_START))
print('Figures / GIF:', run / 'figures')
print('Full batch: python -m nr2d.cli batch')
print('New information batch: python -m nr2d.information evaluate --workers 8')
print('Fresh pilot: python -m nr2d.cli demo --recompute')
print('Vector/raster/GIF export: python -m nr2d.cli export')
print('Configuration, raw traces, matching, inference logs and source hashes are stored in the run folder.')
"""),
]
nb.cells.extend([
M(r"""## 6. Does estimator choice change the value of information?

Compare the published and calibrated TRF fits, a small GRU, and an RME-inspired neural prior with variational inference.
All methods infer the same six coefficients at the **start** of the selected observation window, then use the same
measured-motion replay and command-driven FEM prediction. Learned models share simulated state supervision.
The simulator's full state, its six-coefficient projection and relaxed-start replay are privileged references.

The earlier 24 histories are a reanalysis of already inspected data. The new test histories are evaluated after model
selection. Noise repeats and training seeds do not increase the number of independent physical histories.

The default below **replays recorded results**. Dropdowns select a view; **Run selection** applies it.
Checking **Recompute selected comparison** repeats inference and future FEM prediction into a separate interactive
record. The time slider only moves a cursor. Training and batch generation are CLI operations, never implicit widget work.
"""),
C("""from nr2d.inference_display import InferenceDashboard, load_metrics

INFERENCE_RUN = ROOT / 'runs/inference/methods_20261030'
inference_dashboard = None
if (INFERENCE_RUN / 'study.json').exists():
    estimator_spec = read_json(INFERENCE_RUN / 'study.json')
    print('Planned training seeds:', estimator_spec.get('training_seeds', []))
if (INFERENCE_RUN / 'training.json').exists():
    model_training = read_json(INFERENCE_RUN / 'training.json')
    print('Training record status:', model_training.get('status', 'unreported'))
    trained_models = {}
    for item in model_training.get('results', []):
        if item.get('status') == 'completed':
            key = (item['kind'], item['train_history_count'])
            trained_models.setdefault(key, set()).add(item['seed'])
    display(widgets.HTML(table(['Model', 'Training histories', 'Completed training seeds'],
        [[kind, count, ', '.join(map(str, sorted(seeds)))]
         for (kind, count), seeds in sorted(trained_models.items())])))
if (INFERENCE_RUN / 'summary.json').exists():
    estimator_report = read_json(INFERENCE_RUN / 'summary.json')
    print('Analysis status:', estimator_report.get('status', 'unreported'))
    physical_failures = estimator_report.get('physical_failure_count', estimator_report.get('physical_failures'))
    if physical_failures is not None:
        print('Retained physical-limit records:', len(physical_failures) if isinstance(physical_failures, (list, dict)) else physical_failures)
if (INFERENCE_RUN / 'metrics.csv').exists():
    from nr2d.inference_study import recompute_comparison
    estimator_rows = load_metrics(INFERENCE_RUN)
    split_counts = {split: len({r['history_id'] for r in estimator_rows if r.get('split') == split})
                    for split in sorted({r.get('split', 'unspecified') for r in estimator_rows})}
    print('Independent histories by split:', split_counts)
    print('Recorded training sizes:', sorted({r['training_size'] for r in estimator_rows
                                            if str(r.get('training_size', '')) not in ('', '0', '-1')}))
    print('Recorded training seeds:', sorted({r['training_seed'] for r in estimator_rows
                                            if str(r.get('training_seed', '')) not in ('', '0', '-1')}))
    forecast_audits = [r for r in estimator_rows if isinstance(r.get('physics'), dict)]
    forecast_overruns = [r for r in forecast_audits if r['physics'].get('passed') is False]
    print('Retained forecast-limit records:', len(forecast_overruns), '/', len(forecast_audits), 'audited predictions')
    print('Histories with a forecast-limit record:', len({(r['split'], r['history_id']) for r in forecast_overruns}))
    inference_dashboard = InferenceDashboard(INFERENCE_RUN, recompute=recompute_comparison)
    display(inference_dashboard.widget)
else:
    print('No recorded estimator study is present. Generate it with python -m nr2d.inference_study --help.')
"""),
M(r"""**Information, cost and uncertainty.** The figures below report the new test batch. Per-history means first
average noise repeats and training seeds; plotted intervals bootstrap complete histories. Training-seed variation
is recorded separately in the run report. Missing combinations are marked unrun, and failed rows remain in the raw metrics.

For VI, the point curve propagates the posterior mean coefficient state. Its band propagates posterior samples;
it is not a posterior-predictive-mean curve. The 90% interval for the noiseless force response is distinct from
an interval that also includes observation noise. Full-horizon coverage includes zero-force flight; contact-only
coverage is shown separately when recorded. The final panel deliberately shows the lowest recorded coverage
as a post-hoc failure diagnostic, not as the prespecified representative response.

The intervals use 16 Monte Carlo members. An independent Gaussian reference with the same linear quantiles
has 80.66% coverage for a nominal 90% interval, illustrating finite-sample bias. This is not a distribution-free
correction for the FEM results. Analytic coefficient intervals are unaffected; frozen primary metrics are unchanged.

Coefficient coverage is evaluated against the declared projection of the simulator field. It does not establish
coverage of a unique physical internal coordinate. Accuracy is measured on the independent test histories;
the latency plot uses a separate serial warm phase-compute timing suite when available. This excludes context setup
and experiment I/O. Parallel evaluation times are labeled as such. Application cold/resident-warm totals are recorded
in `latency_startup.csv`; offline training costs are separate.
"""),
C("""if (INFERENCE_RUN / 'metrics.csv').exists():
    for figure_name in ['02_information_by_estimator', '03_accuracy_and_latency', '04_probability_checks']:
        figure_path = INFERENCE_RUN / 'figures' / (figure_name + '.png')
        if figure_path.exists():
            display(Image(filename=str(figure_path)))
    display(Markdown('Run report: [results and interpretation](' +
                     '../runs/inference/methods_20261030/RESULTS.md). '
                     'The default replay does not retrain models or regenerate test histories.'))
    print('Updated notebook elapsed: %.2f s' % (time.perf_counter()-NOTEBOOK_START))
"""),
])
nb.metadata.kernelspec = {"display_name": "NR 2D (Docker)", "language": "python", "name": "nr-spatial-state"}
nb.metadata.language_info = {"name": "python", "version": "3.11.9"}
nbf.validate(nb)
nbf.write(nb, path)
print(path)
