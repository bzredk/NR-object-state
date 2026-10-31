# Estimator and information comparison

This study asks whether the value of lateral sensing, recovery history and earlier contact persists when the estimator changes. It compares inference mechanisms with the material, mesh, contact model and future solver held fixed. Adding a GRU or VI is not itself a contribution.

The [completed default run](../runs/inference/methods_20261030/RESULTS.md) is `runs/inference/methods_20261030`. Its manifests, `training.json`, calibration files, `frozen.json`, `summary.json` and complete metric rows contain the measured outcomes. Git includes one complete interactive test history (`test_20000`); the full raw archive remains local. The commands below support regeneration in a separate reproduction run.

## Physical state and shared prediction

The existing FEM represents a bottom-fixed body with free sides and a circular rigid indenter. The material is synthetic: a relaxed elastic branch and fast/slow internal-strain fields, with relaxation times 0.12 and 0.9 s. The mesh is 24 by 12 cells, the body is 40 by 20 mm with 10 mm thickness, and the time step is 0.01 s. [spatial_contact.json](../configs/spatial_contact.json) is authoritative for all parameters. These are assumed known and fixed; they are not estimated material properties.

The simulated state contains the displacement field, both internal-strain fields and the indenter's actual position/velocity. An action is a commanded actuator position. Command, actual motion and contact reaction are separate signals. Before the prediction cut, inference conditions on measured actual motion. After the cut, the shared solver evolves the actuator and material together under the same command array for every method.

Each inferable branch starts from the same representation at the observation-window start `t_b`:

`z_branch(t_b) = sum_j coefficient[branch,j] * basis[j]`, with six coefficients in `[0, 2.5]`.

The basis has three model-generated strain fields per branch. Labels are bounded least-squares projections in the volume-weighted D strain norm, using unit-modulus weighting. This is not projection in total physical free energy across branches. Projection error and basis singular values/rank are recorded. Unique recovery of those coefficients is not assumed.

TRF, GRU and the RME-inspired methods all estimate these **window-start** coefficients. They initialize the equilibrium displacement from measured tool position, then propagate the full internal fields to the common 1.40 s cut using the same measured motion. The propagated field is not projected back into six coefficients. Thus GRU does not receive an easier current-state target while TRF retains a richer propagated field.

For each physical history, Slow/Fast command schedules are generated once from its true checkpoint, independently of an estimate. The same estimated state is used for both future actions. Hash checks verify shared observations/commands and cross-action state reuse. This is open-loop response prediction, not a robot-control comparison.

## Information available to each estimator

| Condition | Observation update | Additional motion inputs |
| --- | --- | --- |
| Current | One frame at 1.40 s | Current measured position/velocity and command |
| Recent 0.2 s | 21 frames, 1.20–1.40 s | Measured motion throughout that interval |
| Recent 0.8 s | 81 frames, 0.60–1.40 s | Measured motion throughout that interval |
| Earlier contact | Last force-detected continuous 0.2 s contact interval | Motion from that interval to 1.40 s, plus the force scan used to select it |

Each condition is run with either position/velocity/force or those signals plus two lateral markers. Disabled lateral channels are zeroed before feature scaling, neural processing and likelihood construction. Isolation checks replace closed-channel values and verify unchanged estimates. The current condition supplies one valid frame; the GRU starts with zero hidden state on every call.

All learning models receive the same valid samples, without downsampling: five sensor channels, command, absolute time and five channel-mask entries. This gives 12 features per frame. Fixed physical scales are used, not statistics fitted to deployment data. Windows are right-padded to 81 frames with explicit valid lengths; padding is masked. Absolute time and the known loading distribution can supply prior information, so an accurate neural prediction alone does not establish recovery from contact evidence.

The current MLP uses the first valid frame of every training window with that window's start-state label. At deployment it is evaluated only in the current-frame condition. This preserves the learning families' label set while testing a one-frame nonlinear regressor.

Contact selection uses observed force above 4.5 mN, not simulator contact labels. Its full force-scan interval, update interval, propagation interval and total motion span are recorded separately. It must not be described as using only 0.2 s of total history.

A separate [common-anchor control](../src/nr2d/inference_controls.py) fixes the state prior at 0.60 s and grants all variants the same 0.8 s measured-motion span. No update, current-only update and recent-0.2 s update then differ only in fitted force/shape rows. Older target rows are removed before fitting. This diagnoses initialization/propagation effects; it is not the main short-window comparison. Finite coefficient probes and likelihood Jacobians check whether zero force excludes candidate states instead of assuming zero force is universally uninformative.

## Estimators and required controls

| Method | Inference and offline information |
| --- | --- |
| TRF published | Original project settings, including seven objective evaluations, original residual scales and initial coefficient 0.12. Archived old estimates remain available. “Published” refers to the recorded project baseline. |
| TRF calibrated | Shared noise-scaled FEM residuals, analytic implicit Jacobian and a training-label mean/scale prior. The bounded calibration grid varies initialization, budget and regularization. The selected setting is read from `calibration/trf.json`. |
| GRU | One 32-unit recurrent layer and bounded six-coefficient head; no retained memory between windows. |
| Snapshot MLP | Two 64-unit hidden layers with a bounded head, evaluated on the current frame. |
| RME prior | Conv1D, learned positions, attention and tied regression block; directly predicts the six-coefficient center. |
| RME prior mean | Same prior distribution and physical-mean convention as VI, without a likelihood update. |
| RME-inspired VI | Data-conditioned neural Gaussian prior in logit coordinates, mean-field VI, exact FEM likelihood and Adam. |
| Fixed prior VI | Training-label logit distribution instead of the neural conditional prior; same VI implementation. |
| Neural-init TRF | RME center initializes calibrated TRF in representative force-only/contact conditions. |

Calibrated TRF minimizes a sample-normalized, sensor-noise-scaled squared residual plus a standardized coefficient penalty. The penalty weight is appended as a residual, so its squared value multiplies the penalty in the objective. The current calibration selects `prior10`: `max_nfev=60`, regularization weight 1 and training-mean initialization. The earlier `ridge60` common-anchor output is retained separately. These settings change more than the iteration cap; improved results cannot be attributed solely to optimization convergence.

GRU, MLP and RME use identical physical parents and privileged projected labels. Training uses coefficient MSE, Adam, 300-epoch maximum and validation patience 40. A two-learning-rate scout (0.003/0.0003) uses 32 training histories and seed 11; each family then uses its selected rate at 32/128 histories and seeds 11/23/37. Training curves, label-mean baselines, hashes, costs and checkpoint reports are saved. No future force response enters training loss.

Original RME estimates window-static payload mass/CoM; this adaptation estimates dynamic internal state at the window start and subsequently evolves it. The verified original architecture, training information, paper/code differences, pinned commit and adaptation table are in [RME_ADAPTATION.md](RME_ADAPTATION.md). This is not a reproduction of the original robot compensation, passivity or runtime claims.

### VI density, point prediction and uncertainty

Both prior and variational distributions are diagonal Gaussians in `z`, transformed by `c = 2.5 sigmoid(z)`. Their common invertible transformation cancels in the KL. Random samples are never clipped. The negative ELBO uses the **sum** of Gaussian sensor NLLs over the selected observations and the exact Gaussian KL; it does not reuse TRF's sample-normalized objective. The observation likelihood conditions on noisy measured motion, without an errors-in-variables model.

The neural center uses the same window as the likelihood. Prior logit scales start from validation residual RMS with a 0.15 floor; VI budget and scale multiplier are chosen on separate calibration histories using Slow press. Fixed-prior scales come from training-label logits, with their declared multiplier. These are data-conditioned inference objectives, without a strict Bayesian-calibration guarantee.

The point output is the physical coefficient mean, integrated by 64-point marginal Gauss-Hermite quadrature, then propagated by the FEM. It is **not** the mean of predicted force trajectories. Because `E[sigmoid(z)]` differs from `sigmoid(E[z])`, compare VI with both the raw neural center and **RME prior mean**. Otherwise a change of point summary could masquerade as likelihood refinement. Fixed-prior initialization and common-anchor controls likewise help identify prior/propagation effects.

Sixteen posterior coefficient samples provide representative Slow-press uncertainty checks for force-only 0.2 s and lateral 0.2/0.8 s conditions. Force-trajectory uncertainty and future measurement-noise uncertainty are separate. Saved samples, interval coverage/width and CRPS are reported; the shared future noisy target does not depend on method or training seed. A prespecified contact-only diagnostic uses a common true-future-force mask above 4.5 mN, exclusively during evaluation, so separated zero-force samples cannot obscure contact-interval coverage. Coverage of projected coefficients concerns the selected projection, not unique recovery of physical internal variables. Posterior/prior scale ratio, logit-mean shift and KL expose prior preservation or overconcentration in weak-information conditions. Posterior physics audits cover future sampled trajectories, not every inferred history-replay path. Fixed-budget completion is not called VI convergence.

A retained numerical floor limits exact-prior-preservation claims. With the actual neural prior scales, a constant-likelihood diagnostic produced a maximum 0.00862 log-scale change (about 0.86%) and `3.22e-6` physical coefficient-mean change after eight Adam steps. Its final distribution matched the real force-only waiting-window run bitwise, whose sampled FEM likelihood gradients were all zero. The toy unit's scale 0.5 did not expose this roundoff amplification. [The full record](../runs/inference/methods_20261030/verification/weak_prior_numerical_drift.json) accompanies the unchanged frozen run; tiny weak-window VI/prior differences must not be called new sensor information.

Force-band coverage also has a finite-Monte-Carlo limitation. With 16 draws and the same linear empirical 5th/95th quantiles, a correctly specified independent Gaussian reference achieved 80.66% coverage (Monte Carlo 95% interval 80.49-80.83%). Thus a reported noisy-force coverage such as 83.8% is not automatically evidence of a miscalibrated posterior. The reference's empirical CRPS was 0.59983, consistent with `(1+1/16)/sqrt(pi) = 0.59945`; the infinite-Gaussian expected score is 0.56419. This is an analysis-only reference, not a distribution-free correction for the FEM force distribution or zero-force mass. Analytic coefficient quantiles do not have this empirical-quantile bias. [Saved diagnostic](../runs/inference/methods_20261030/verification/finite_mc_reference.json); run `python tools/check_monte_carlo_metrics.py` after timing. Frozen scores and intervals remain unchanged.

## Data, calibration and statistical units

| Role | Planned physical parents; see manifests for completion | Use |
| --- | --- | --- |
| Training | 128; nested prefix of 32 | Supervised labels and training-only priors |
| Validation | 16 | Early stopping, learning-rate scout and residual scales |
| Calibration | 8 available | First three select TRF; next three select VI; Slow press only |
| Pilot | Eight available | Cost, diagnostics and precision planning; evaluation uses one noise repeat and training seed 11 |
| Old reanalysis | The previously inspected 24 | Paired reanalysis, not new confirmation |
| New test | Frozen before generation | Independent confirmation after methods/analysis are fixed |

Three independently seeded sensor-noise realizations are prepared per physical parent. All windows and noise repeats from one parent remain in one split. Future branches are generated only during evaluation. Physics-limit episodes remain in the records; they are not silently filtered. The final-test size and precision rationale are stored in `frozen.json`; the test is not regenerated to improve a result.

Primary outcomes are future force RMSE in mN and paired information gains: current lateral benefit, and 0.2/0.8 s history beyond current lateral sensing. Their differences across GRU/RME-inspired VI and TRF address estimator dependence. Earlier-contact comparisons and prior/VI controls are targeted diagnostics. Fast press reuses the same estimate and is excluded from hyperparameter selection.

Noise repeats are averaged within each training seed and physical history, then training seeds are equally averaged. Paired comparisons match history/noise and shared learned seeds where applicable. Bootstrap resampling uses whole histories. Training-seed variability is reported separately; frames, noise draws, trained seeds and future branches do not increase independent N. Old, pilot and new-test results remain separate. An interval containing zero does not establish equivalence or noninferiority.

## Reproduce inside Docker

Use [the existing Dev Container](CONTAINER.md). The project bind mount retains code, data, checkpoints and outputs. Select **NR 2D (Docker)** for [the Notebook](../notebooks/spatial_2d_hidden_state.ipynb). The Windows `.venv` is not used. All commands below run from `/workspaces/NR-object-state/2D`.

For an isolated rerun, initialize a new run name once:

```bash
python -c "from pathlib import Path; from nr2d.experiments import read_json,write_json; p=Path('runs/inference/reproduction'); p.mkdir(parents=True,exist_ok=False); s=read_json('configs/inference_study.json'); s['run']=str(p); write_json(p/'study.json',s)"
```

Set the run once for the sequence below. Use `INFERENCE_RUN=runs/inference/methods_20261030` only when intentionally replaying the supplied default run. Existing completed datasets/checkpoints are reused after their checks; omit `freeze` when that run already has a frozen test.

```bash
INFERENCE_RUN=runs/inference/reproduction
python -m nr2d.inference_study data --run "$INFERENCE_RUN" --split train --workers 4
python -m nr2d.inference_study data --run "$INFERENCE_RUN" --split validation --workers 4
python -m nr2d.inference_study data --run "$INFERENCE_RUN" --split calibration --workers 4
python -m nr2d.inference_study data --run "$INFERENCE_RUN" --split pilot --workers 4
python -m nr2d.inference_study train --run "$INFERENCE_RUN" --workers 3
python -m nr2d.inference_study calibrate-trf --run "$INFERENCE_RUN" --workers 3
python -m nr2d.inference_study calibrate-vi --run "$INFERENCE_RUN" --workers 3
python -m nr2d.inference_study verify --run "$INFERENCE_RUN"
python -m nr2d.inference_study evaluate --run "$INFERENCE_RUN" --split pilot --workers 3
python -m nr2d.inference_controls --run "$INFERENCE_RUN" --histories 4 --repeats 1 --workers 2
python -m nr2d.inference_study freeze --run "$INFERENCE_RUN"
python -m nr2d.inference_study data --run "$INFERENCE_RUN" --split test --workers 4
python -m nr2d.inference_study evaluate --run "$INFERENCE_RUN" --split test --workers 3
python -m nr2d.inference_study evaluate --run "$INFERENCE_RUN" --split old --workers 3
python -m nr2d.inference_study verify --run "$INFERENCE_RUN"
python -m nr2d.inference_study collect --run "$INFERENCE_RUN"
python -m nr2d.inference_study summarize --run "$INFERENCE_RUN"
```

Stop competing training/evaluation workers before timing:

```bash
python -m nr2d.inference_study latency --run "$INFERENCE_RUN"
python -m nr2d.inference_study figures --run "$INFERENCE_RUN"
python tools/build_spatial_notebook.py
python tools/execute_spatial_notebook.py
```

Latency uses CPU, one thread and batch one, with one first-invocation measurement and three warmed calls per selected condition. The first call includes model loading within an existing process; it is not OS/Python startup. `total_seconds` sums encoding, fitting/replay, VI and future prediction, including requested posterior Monte Carlo propagation. Dedicated timing disables the exact-state/action forecast cache used to accelerate duplicate batch forecasts; that cache does not replace likelihood or Jacobian calculations. `dedicated_wall_seconds` additionally includes experimental truth/projection/setup and I/O, so is not deployment latency. Training is timed separately. No GPU/CUDA or original-RME real-time claim follows from this CPU study.

`latency_startup.csv` supplements these phase measurements for TRF, GRU and RME-inspired VI at lateral 0.8 s. One fresh Python child per method performs one complete cold application call and exits; the parent measures launch, imports, context/model initialization, input/preset reads, inference, point/Monte Carlo forecasts, output I/O and exit. A separate child warms its resident context once, then measures three `app_warm_total_seconds` calls starting before window construction, masking, input packing and prior/model lookup. These warm timers include inference/replay/forecasts but exclude oracle generation, projection and output I/O. The input fixture contains only known physical context, causal observations/commands and the already-issued future command, with no evaluation internal fields or future response. Fresh processes do not imply a flushed OS/filesystem cache.

An execution-only amendment bounded that forecast cache to 16 entries per worker after memory growth interrupted the batch. It preceded any completed final-test job or final-metric inspection. The original freeze/source, exact code diff, effective source hashes and bitwise cached/uncached array checks are preserved in `verification/runtime_cache_amendment.json` and `source/cache_amendment/` under the run. Model, likelihood, seeds, data and analysis rules were unchanged. Completed old-reanalysis jobs were retained; interrupted jobs were restarted. Batch wall times can reflect cache reuse and worker contention, so dedicated timing remains the latency source.

`summary.json`, `metrics.csv`, `failures.json`, `latency.csv`, per-history `result.json`/`inference.json` and raw curve files are the reviewable outputs. Notebook Run All computes the lightweight physical demo and replays inference caches; an explicit Recompute button writes a separate interactive record. Its `INFERENCE_RUN` cell defaults to the supplied run; select the new path there when viewing a separate reproduction. A restarted kernel must not rely on prior cell state.

## Interpretation boundary

This is a fixed synthetic material, geometry, loading distribution, sensor model and six-coefficient initialization family. It can test predictive information and computational tradeoffs within that scope. It does not establish minimal state, universal observability, calibrated beliefs under model mismatch, closed-loop action gain, cross-material/mechanism transfer or real-robot readiness. Learned benefits may reflect a training-distribution prior; weak-information and data-matched-prior controls are necessary before attributing them to new observation evidence. The earlier shared-feedback negative result remains part of the project.
