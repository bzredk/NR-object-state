# Model and comparison

[2D overview](../README.md) · [Experiment design](DESIGN.md) · [Method comparison](INFERENCE_STUDY.md) · [Parameters](CONFIGURATION.md)

## Physical model

The body uses small-strain plane strain, a fixed bottom and traction-free sides. Quasistatic equilibrium is solved at every step, with a dynamic normal tool coordinate and frictionless circular contact. The numerical parameters are in [CONFIGURATION.md](CONFIGURATION.md#physical-parameters).

Each element carries two three-component internal-strain tensors. Stress combines equilibrium elasticity and two Maxwell branches; internal strains use backward Euler. The state contains these fields and tool position/velocity, together with fixed material and geometry. Equilibrium determines displacement and contact. The circular gap gradient produces lateral and vertical reactions; the penalty contact permits small penetration.

## Observations and inference

The five observed channels are actual tool position, velocity, normal force and two lateral marker displacements. Samples share a simulated clock. Each command after the first applies over the preceding integration interval. Inference uses measured motion; future prediction uses the command-driven tool model.

In the recorded pilot and sensor/contact study, snapshot and finite-window fits estimate nonnegative coefficients on model-generated initial-strain shapes. Position drives the replay; force and lateral displacement form the residual. The residual is divided by its channel scales and by the square root of the number of samples. A coefficient penalty is appended before least squares. [The published observer preset](../configs/observer.json) exposes those scales, penalty, basis depths, optimizer limits and history windows.

Full-history replay starts from known relaxed internal strain. It does not fit coefficients. True-state prediction receives the complete internal fields as a reference for the same future command. Neither is an absolute upper bound over all possible controllers. Optimization counts and actual inference time are recorded; a shared evaluation cap does not imply equal solver work across window lengths.

The original pilot matches all five current channels before future branching. The limits are proximity bands: estimators retain the raw values. Both markers are symmetric under centered loading and provide one lateral shape mode. In that pilot, noise is shared across methods, with one realization per condition and seed 123 reused across parents. Sensor limits and noise levels are synthetic; [their values are recorded here](CONFIGURATION.md#observations-and-comparison-conditions).

The later sensor/contact study masks force-only versus force-and-lateral residuals and compares current measurements, recent windows and earlier contact updates. Disabled lateral inputs are excluded from initialization and fitting. Noise is shared across methods within each history/repeat, with independent draws across histories and repeats. [Recorded conditions and checks](../runs/information/channels_contact_20261026/RESULTS.md).

The implemented [method comparison](INFERENCE_STUDY.md) keeps that physical system and compares calibrated TRF, GRU, a current-frame MLP and an RME-inspired neural prior with VI. Each estimates the same six coefficients at the observation-window start, then uses the same full-field propagation and future commands. Learning models share privileged projection labels; the MLP receives only each window's first frame. [The RME audit](RME_ADAPTATION.md) separates the original method from this adaptation. Execution status and results belong to the run records, not the protocol.

## Data

### Original pilot

The pilot contains 3 calibration, 2 validation and 8 test histories, plus two validation histories for window sweeps. Each complete history owns its future/noise branches. All 28 test pairs pass the chosen matching bands; the independent test count remains eight.

[The pilot manifest](../runs/2d_contact/pilot_c9dd23016896/manifest.json) records those splits and [the results](../runs/2d_contact/pilot_c9dd23016896/RESULTS.md) report the method comparison. New code and observer settings write separate comparison records. Published source snapshots retain the implementation that produced each earlier result.

### Sensor channels and contact history

Three separate calibration histories select the observer preset. The test batch contains 24 independent loading histories, each with three independent noise draws. The prediction cut is 1.40 s. Each estimated state is reused for Slow press and Fast press; noise repeats and action branches do not increase the independent sample count. [Data, intervals and reproduction commands](../runs/information/channels_contact_20261026/RESULTS.md).

The method comparison uses separate whole-history training, validation, calibration, pilot and frozen-test roles. It keeps the old 24 histories as seen reanalysis. [Split definitions and the freeze procedure](INFERENCE_STUDY.md#data-calibration-and-statistical-units) specify which data may select models or inference settings.

## Results and scope

The earlier evidence covers a small-strain, same-material, same-geometry synthetic prediction study with the published observer. Peak strain is about 4.7%; refinement checks cover total force and the selected pair contrast. Matching residuals leave instantaneous information, especially through lateral shape. The six-coefficient representation is approximate. Those results do not establish that information gains are independent of the estimator.

The [pilot report](../runs/2d_contact/pilot_c9dd23016896/RESULTS.md) includes the failed penetration criterion; the [physical audit](../runs/2d_contact/pilot_c9dd23016896/physical_failures.json) preserves the original and diagnostic runs. The later [sensor/contact report](../runs/information/channels_contact_20261026/RESULTS.md) retains its own physical overruns and optimization limits. The [current comparison](DESIGN.md#next-comparison) tests estimator dependence. Recursive updates, common-feedback action benefit, real-material calibration, sensor delay and broader transfer remain untested here.

The 2D geometry, lateral deformation and force response address Nadia's feedback through the hidden-state question. Solver and estimator choices are implementation decisions made in this project.

## Implementation

| Module | Responsibility |
| --- | --- |
| [physics.py](../src/nr2d/physics.py) | FEM, internal fields, contact, tool dynamics and saved traces |
| [protocols.py](../src/nr2d/protocols.py) | Loading histories and future commands |
| [estimators.py](../src/nr2d/estimators.py) | Observer configuration, inference and shared prediction |
| [experiments.py](../src/nr2d/experiments.py) | Data splits, matching, comparisons and run records |
| [information.py](../src/nr2d/information.py) | Sensor masks, contact-window comparison, calibration and grouped evaluation |
| [inference_physics.py](../src/nr2d/inference_physics.py) | Shared window-start state, projection and exact-FEM observation Jacobians |
| [inference_data.py](../src/nr2d/inference_data.py) | Whole-history splits, shared noisy windows and privileged projection labels |
| [learning_models.py](../src/nr2d/learning_models.py) | GRU, MLP and RME-inspired encoders, training and checkpoint replay |
| [variational.py](../src/nr2d/variational.py) | Bounded-state VI with the FEM likelihood |
| [inference_study.py](../src/nr2d/inference_study.py) | Calibration, freeze and shared method evaluation |
| [inference_controls.py](../src/nr2d/inference_controls.py) / [inference_analysis.py](../src/nr2d/inference_analysis.py) | Common-anchor controls, history-level statistics and serial latency checks |
| [inference_display.py](../src/nr2d/inference_display.py) | Recorded method-comparison figures and Notebook controls |
| [checks.py](../src/nr2d/checks.py) | Physical references and refinement checks |
| [plotting.py](../src/nr2d/plotting.py) / [widgets.py](../src/nr2d/widgets.py) | Figures and native Notebook interaction |
| [cli.py](../src/nr2d/cli.py) | The same implementation from the command line |

[Setup](CONTAINER.md) · [Parameters](CONFIGURATION.md) · [Experiment design](DESIGN.md) · [Evidence index](../runs/README.md)
