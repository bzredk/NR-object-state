# State estimation: literature and next experiment

Updated 2026-11-04. Based on the [completed 2D comparison](../runs/inference/methods_20261030/RESULTS.md), paper methods and appendices, and the code revisions linked below. External methods were inspected, not reproduced. The experiments proposed here have not been run.

Our current problem is nonlinear state estimation with a known physical model. Material parameters stay fixed; internal relaxation fields evolve. Each method estimates six coefficients at the window start, then the same FEM propagates them and predicts the next press. TRF solves the bounded least-squares problem; it is not a separate estimation theory. The GRU also starts fresh for each window, so this study has not yet tested a persistent observer.

## What the results establish

With lateral sensing and 0.8 s history, Slow-press force RMSE is **0.197 mN for calibrated TRF, 0.437 for GRU and 0.605 for VI**. The test uses 24 independent histories, three noise repeats and three training seeds. GRU is faster in the recorded CPU case; VI sometimes improves its prior but does not outperform TRF here.

Current lateral shape helps. History adds information, but the 0.2 s window contains detached waiting while 0.8 s includes contact and recovery. Their difference also changes the state anchor and motion span. In 144 weak-information fits, calibrated TRF keeps the training mean unchanged. Accurate prediction can therefore come from the prior.

Reusing one estimate for Slow and Fast press supports prediction across those two commands. It does not establish action benefit. The [1D shared-feedback comparison](../../1D/RESULTS.md) found no useful improvement in action; that result still constrains the next control experiment.

## Closest work

| Work | Hidden quantity and available information | Mechanism and evidence | Relevance here |
| --- | --- | --- | --- |
| [Schiller et al., MHE, 2023](https://arxiv.org/abs/2202.12744v2) | Dynamic state; known nonlinear model, inputs and noisy outputs | Constrained moving-window optimization with a prior from the previous estimate; stability analysis and two simulated systems | Closest classical formulation. Our independent window fits do not carry an arrival prior. |
| [Becker et al., RKN, 2019](https://proceedings.mlr.press/v97/becker19a.html) | Learned latent state; observations and control inputs | Learned transitions and factorized Kalman updates; real hysteretic robot-joint prediction | A direct precedent for retaining history to predict later responses under control inputs. |
| [Revach et al., KalmanNet, 2022](https://arxiv.org/abs/2107.10043v3) | Dynamic state; approximate transition/observation models and measurements | Learned correction gain within model-based prediction; noise/model-mismatch tests | A possible update mechanism if classical recursive inference leaves a measured gap. |
| [Ai et al., RoboPack, 2024](https://www.roboticsproceedings.org/rss20/p130.pdf) | Object geometry and latent physics; vision, touch and tool motion | Recurrent estimation, graph prediction and MPC; robot pushing and packing | Already covers estimation from interaction history followed by prediction and control. |
| [Jaszczuk and Figueroa, RME, 2025](https://proceedings.mlr.press/v305/jaszczuk25a.html) | Window-static payload mass/CoM; joint motion and torque residuals | Neural prior plus physical-likelihood VI; real-robot compensation | Source of our auxiliary method, but its target is an unknown parameter rather than evolving material state. |

The descriptions below are paper/code facts unless marked **Interpretation**. Unverified details are stated separately; an unreported experiment does not imply inability.

### Moving horizon estimation

[Sections III-A/B, Eqs. 6–8 and Theorem 1](https://arxiv.org/pdf/2202.12744v2) combine recent measurement residuals, disturbance penalties and an arrival term centered on an earlier estimate. The optimized state is propagated to the current time. This retains information after its measurements leave the window. Section V tests a chemical process and a 12-state quadrotor in simulation; Appendices A/B support the stability analysis. There is no learned prior or state-label training.

**Interpretation:** an arrival term is the important addition to our finite-window fit; a warm start alone does not change its information. The theorem requires detectability, horizon and optimization assumptions that our contact FEM has not established. Author code was not verified.

### Recurrent Kalman Networks

[Sections 2.1–2.6, Eqs. 1–9](https://proceedings.mlr.press/v97/becker19a/becker19a.pdf) encode measurements and their uncertainty, propagate a latent mean/covariance, then correct them with a structured Kalman update. Dynamics and representation are learned. The latent state need not match physical coordinates.

The real Hydro-Lek joint experiment uses position and controller current at 100 Hz. Training windows provide 300 observed steps followed by 200 steps with current only; Figure 4 uses a 200-step observation period. [Appendix D.4](https://proceedings.mlr.press/v97/becker19a/becker19a-supp.pdf) reports six sequences, five for training and one for testing, with GRU/LSTM comparisons. The [inspected code](https://github.com/LCAS/RKN/tree/81902ad12f728fc75d7a1bbda824520e7d70eb5d) is an older TensorFlow implementation; a complete robot-data reproduction was not verified.

**Interpretation:** this is strong prior work for memory-dependent response prediction. It does not establish recovery of unique internal variables or a shared-feedback control advantage.

### KalmanNet

[Section III, Eqs. 12–14 and Figs. 2–4](https://arxiv.org/html/2107.10043v3) retain model-based state prediction and learn the correction gain from measurement/prediction differences. State trajectories supervise training. Section IV compares classical filters and neural estimators under selected nonlinearities and mismatch; the real NCLT example concerns odometry, not manipulation.

The [inspected implementation](https://github.com/KalmanNet/KalmanNet_TSP/blob/828a2cf529bc84f43b37d543d916fe5858054457/KNet/KalmanNet_nn.py) retains three GRU memories. Our action-driven FEM would need an explicit input interface. One released Lorenz example reuses a clean trajectory with fresh noise across splits; it is not an independent-history test.

**Interpretation:** learning the update may help when the noise model or correction rule is inadequate. It cannot recover information absent from the observations, and it does not itself provide a calibrated belief.

### RoboPack

[Sections III-A/C/D, Eqs. 1–9 and Fig. 3](https://www.roboticsproceedings.org/rss20/p130.pdf) separate a tactile-informed recurrent estimator from a future predictor that does not require future touch. Training uses particle-trajectory error rather than physical-parameter labels. A Panda, Soft-Bubble sensors and depth cameras support pushing and packing; Tables I–III report prediction and task results.

The [released estimator](https://github.com/BoAi01/robopack/blob/ac02a01006b50dbb7379b676ec5a622921875c5b/dynamics/models/estimator_predictor_dpi_net_obj_latent_lstm.py) retains LSTM hidden/cell state beyond its 16-dimensional physics output. Its model inputs use realized tool displacements. Its physics baseline has different sensory information; it is not a matched-sensor comparison with our observer. Full data-split reconstruction remains unverified.

**Interpretation:** history-based physical estimation for control is established. Our sparse sensing and controlled relaxation mechanism define a different test, but do not by themselves establish novelty.

### Rapid Mismatch Estimation

Sections 4–5, Eqs. 9–18 and Appendices A–H describe a network prior refined by a torque likelihood, followed by payload compensation. The estimator uses a 200 ms window; change detection additionally uses 500 ms of force history and a 230 ms stabilization delay. Appendix D includes PVIK pick-and-place with SoftHand2 and OptiTrack. These are paper results, not only project-page demonstrations.

The [pinned code](https://github.com/mateusz-jaszczuk/RapidMismatchEstimation/tree/a2fe683ccc7a921692949445f867a73938933d89) restarts inference for each call rather than carrying a filtered posterior. The complete training pipeline and parent-group split were not verified. See the [existing adaptation audit](RME_ADAPTATION.md) for architecture and likelihood details.

**Interpretation:** RME supports separating prior quality, likelihood refinement and control benefit. It does not validate our VI uncertainty or justify increasing VI complexity after the current mixed results.

## What remains unresolved

| Possible limitation | What we can say now |
| --- | --- |
| Missing information | Force-only waiting fits can be prior-dominated. A local zero likelihood gradient is not a global non-observability proof. |
| State representation | Projected true coefficients give 0.086 mN versus TRF's 0.197 mN in the main condition. This shows their predictive capacity; it does not separate representation and estimation error or give a strict lower bound. Six start coefficients are not yet a verified recursive state. |
| Prior and optimization | Calibration changed residual weights, prior, initialization, Jacobian and budget together. Its gain cannot be assigned to convergence alone. |
| Noisy motion | Measured motion drives replay as if exact. Input-noise propagation has not been isolated. |
| Model mismatch | The generator and predictor share the same model. Current errors do not demonstrate a need to learn missing physics. |
| Uncertainty | VI gains depend on the condition. The 16-sample interval diagnostic complicates force-coverage interpretation; calibrated belief remains unestablished. |

## Next comparisons

**First: retain information after contact leaves the window.** Borrow the arrival-prior idea from MHE. Reuse the FEM and observation Jacobians; carry the estimated fields, or first validate a reduced recursive state. Compare a reset short-window fit, propagation of the previous estimate without updates, and propagation with new observations. Use the same initial prior, motion stream and permitted sensor channels. A full-prefix estimate is the reference. Overlapping measurements must not be counted twice.

Test at contact end, during waiting and at recontact, with varied waiting times and lateral sensing on/off. Measure future force/shape error, drift and update time. Use the now-seen histories for design and fresh histories for confirmation. Stop expanding this route if retaining older information adds no useful accuracy at the required cost. If projection dominates, repair the representation first.

**Second, only if needed: learn the correction.** Borrow KalmanNet's model-prediction/learned-update split after finding a repeatable weakness in the classical observer. Keep the FEM and compare against a calibrated classical update with matched data and supervision. Stop if the gain disappears on new histories, comes from extra prior information, or adds complexity without a useful accuracy/cost improvement. Our present results do not justify another large encoder or a broader VI sweep.

## Contribution boundary

Finite-window fitting, recurrent filtering, learned physical memory and neural-prior VI are established methods. Applying them to this FEM is baseline work.

The question worth testing is narrower: **when contact information is intermittent and internal state keeps relaxing, what must be retained until the next interaction?** A useful result must separate retained information from current shape, prior knowledge and ordinary feedback. Prediction evidence should precede a shared-controller action test, including a true-state reference under the same action choices. If that reference offers no useful benefit, do not claim an action contribution for that task.

[Earlier literature audit](../../research/hidden_state_reframing/closest_work.md) · [State formulation](../../research/hidden_state_reframing/formalization.md) · [Study protocol](INFERENCE_STUDY.md)
