# Run records

[2D overview](../README.md) · [Batch commands](../tools/README.md)

## Scientific evidence

The latest run is [estimator choice, sensing and history](inference/methods_20261030/RESULTS.md):
calibrated TRF, GRU and RME-inspired VI on 24 fresh test histories, with three noise repeats and three training seeds.
The old 24-history reanalysis is separate. `inference/methods_20261030/` holds datasets, model weights, per-history results,
source/configuration freezes, latency records, figures and verification. Notebook Section 6 compares these auxiliary methods.
Git includes complete metrics and the first prescribed test history (`test_20000`) for interactive replay and recomputation;
full physical trajectories and repeated individual forecasts remain in the local archive.

The earlier comparison is [sensor channels and contact history](information/channels_contact_20261026/RESULTS.md):
24 new loading histories, three independent noise draws each, and the same estimated state used for two future commands.
Its `summary.json`, `metrics.csv`, `calibration.json` and `failures.json` accompany the three figures.
`histories/` stores physical episodes; `test/<history>/repeat_<n>/` stores observations, estimated states and predictions.

The original spatial experiment is [pilot_c9dd23016896](2d_contact/pilot_c9dd23016896/RESULTS.md). Its eight histories and negative noise result are retained.

| Record | Use |
| --- | --- |
| [Results](2d_contact/pilot_c9dd23016896/RESULTS.md) | Eight-history comparison, scope and retained failure |
| [Manifest](2d_contact/pilot_c9dd23016896/manifest.json) | History generation, splits, environment and source versions |
| [Matching](2d_contact/pilot_c9dd23016896/matching.json) | All current channels, residuals, limits and pair-selection rule |
| [Batch](2d_contact/pilot_c9dd23016896/batch.json) | Per-history method results across actions and noise settings |
| [Physical checks](2d_contact/pilot_c9dd23016896/checks.json) / [failures](2d_contact/pilot_c9dd23016896/physical_failures.json) | Numerical checks and over-limit traces |
| [Pair refinement](2d_contact/pilot_c9dd23016896/pair_refinement.json) | The same selected pair under mesh refinement |

Within the pilot, `histories/` and `main.npz` contain trajectories and internal fields. `forecasts/`, `sweeps/` and `figures/` contain the published comparisons. `source/` and `environment/` preserve the implementation used at execution time.

New settings or recomputations use `comparisons/observer_<key>/`, where the key includes observer settings and numerical source hashes. Each forecast records its actual settings and inference diagnostics. Unconfigured calls may replay the original published cache; this is labeled REPLAY. No earlier result is relabeled as a new experiment.

## Development records

These check software and presentation, rather than adding independent scientific samples.

| Record | Check |
| --- | --- |
| [Observer configuration regression](implementation_checks/observer_config/README.md) | Published defaults, cache isolation and Notebook calls after the refactor |
| [Docker validation](container_validation/20261026_docker_layout/validation.json) | Rebuilt environment, physics checks, fresh kernel, callbacks and exports |
| [Native controls](container_validation/20261026_docker_layout/native_controls.json) | Direct VS Code playback, slider, pause and reset observations |

<details>
<summary>Earlier development records</summary>

- [Initial physical checks](2d_contact/development): coarse-mesh failure and subsequent refinement.
- [Windows frontend](2d_contact/pilot_c9dd23016896/frontend/windows): original execution and interaction records.
- [1D workcopy snapshot](../../1D/provenance/workcopies/20261026_cleanup/README.md): recovery of earlier local reproductions.

</details>

Repeated Notebook sessions remain local under `notebook_runs/` or `runs/notebook_validation/`. Only the current container validation is published; the [spatial animation](2d_contact/pilot_c9dd23016896/figures/01_spatial_process.gif) is shared with the scientific pilot. Retained historical records keep their execution-time paths and hashes.
