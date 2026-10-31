"""Rebuild the measured training table and audit the frozen learning inputs."""
import hashlib
import json
from pathlib import Path

import numpy as np

from nr2d.inference_data import load_dataset
from nr2d.learning_models import _data_hash, first_frame_inputs


run = Path(__file__).resolve().parents[1]
read = lambda p: json.loads(p.read_text(encoding="utf-8"))
training = read(run / "training.json")
scouts = read(run / "training_scouts.json")
validation = load_dataset(run / "datasets/validation.npz")
checks = []
table = []

for kind in ("gru", "rme", "mlp"):
    scores = {}
    candidates = [r for r in scouts["results"] if r["kind"] == kind and r["status"] == "completed"]
    chosen_lr = min(candidates, key=lambda r: r["report"]["best_validation_mse"])["learning_rate"]
    assert chosen_lr == training["selected_learning_rates"][kind]
    for size in (32, 128):
        train = load_dataset(run / "datasets/train.npz", max_histories=size)
        tx, tl = train["x"], train["lengths"]
        vx, vl = validation["x"], validation["lengths"]
        if kind == "mlp":
            tx, tl = first_frame_inputs(tx)
            vx, vl = first_frame_inputs(vx)
            assert np.all(tl == 1) and np.all(tx[:, 1:] == 0)
        rows = sorted((r for r in training["results"] if r["kind"] == kind and
                       r["train_history_count"] == size), key=lambda r: r["seed"])
        assert [r["seed"] for r in rows] == [11, 23, 37]
        samples = []
        for row in rows:
            path = run / row["checkpoint"]
            report = read(path.with_suffix(".json"))
            assert report["training_data_sha256"] == _data_hash(tx, tl, train["y"])
            assert report["validation_data_sha256"] == _data_hash(vx, vl, validation["y"])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == report["checkpoint_sha256"]
            np.testing.assert_allclose(report["initialization_mean_coefficients"], train["y"].mean(0))
            np.testing.assert_allclose(report["prior_logit_scale"],
                                       np.maximum(.15, report["validation_logit_residual_rmse"]))
            assert report["train_example_count"] == size * 24
            assert report["validation_example_count"] == 16 * 24
            assert report["learning_rate"] == chosen_lr
            assert report["best_validation_mse"] == min(r["validation_mse"] for r in report["learning_curve"])
            samples.append({"seed": row["seed"], "coefficient_validation_mse": report["best_validation_mse"],
                            "training_wall_seconds": report["training_wall_seconds"],
                            "epochs_completed": report["epochs_completed"], "best_epoch": report["best_epoch"],
                            "checkpoint": str(path.relative_to(run)), "checkpoint_sha256": report["checkpoint_sha256"]})
        scores[str(size)] = float(np.mean([s["coefficient_validation_mse"] for s in samples]))
        table.append({"kind": kind, "training_histories": size, "parameter_count": report["parameter_count"],
                      "learning_rate": chosen_lr, "three_seed_results": samples,
                      "mean_coefficient_validation_mse": scores[str(size)],
                      "mean_training_wall_seconds": float(np.mean([s["training_wall_seconds"] for s in samples]))})
    selected = 128 if scores["128"] <= scores["32"] else 32
    assert selected == training["selected_sizes"][kind]
checks.append("All 18 checkpoints, hashes, full label counts, training-only means, validation scales and selected sizes match the saved inputs")

datasets = []
overruns = []
seen_seeds = set()
for split in ("calibration", "pilot", "train", "validation"):
    manifest = read(run / "datasets" / f"{split}.json")
    data = load_dataset(run / "datasets" / f"{split}.npz")
    assert not (seen_seeds & set(manifest["seeds"]))
    seen_seeds.update(manifest["seeds"])
    assert np.isfinite(data["x"]).all() and np.isfinite(data["y"]).all()
    assert np.all((data["y"] >= 0) & (data["y"] <= 2.5))
    for index, length in enumerate(data["lengths"]):
        assert np.all(data["x"][index, length:] == 0)
    basic = data["channels"] == "basic"
    assert np.all(data["x"][basic, :, 3:5] == 0)
    assert np.all(data["x"][basic, :, 10:12] == 0)
    for history in manifest["histories"]:
        for failure in history["failures"]:
            overruns.append({"history_id": history["id"], **failure})
    datasets.append({"split": split, "independent_histories": manifest["independent_history_count"],
                     "examples": manifest["example_count"], "generation_wall_seconds": manifest["wall_seconds"],
                     "failure_count": len(manifest["failures"]), "dataset_sha256": manifest["dataset_sha256"]})
checks.append("The four physical-history seed sets are disjoint; all labels/inputs are finite, padding and disabled side channels are zero")

summary = {
    "status": "reviewed", "run": str(run), "device": "CPU", "torch": "2.8.0+cpu", "threads_per_fit": 1,
    "metric": "Coefficient validation MSE on 16 held-out physical histories; not future-force prediction error",
    "independent_unit": "Complete physical loading history; windows, noise draws and training seeds are not new physical samples",
    "table": table, "selected_sizes": training["selected_sizes"],
    "total_training_elapsed_wall_seconds": training["wall_seconds"],
    "main_comparison_unique_fit_count": training["main_comparison_training_fit_count"],
    "total_actual_scout_and_main_fit_count": training["actual_training_fit_count"],
    "sum_final_comparison_fit_wall_seconds": training["main_comparison_sum_fit_wall_seconds"],
    "timing_note": "Per-fit wall durations overlap under parallel execution and are not CPU time. The 72.67 s equal-label MLP correction overlaps total elapsed; do not add it again. Overfit diagnostics and data generation are separate.",
    "overfit_before_head_initialization": read(run / "checks/overfit_before_head_init/overfit.json"),
    "overfit_after_head_initialization": read(run / "checks/overfit.json"),
    "overfit_comparison_note": "Before-initialization MLP used 24 snapshot-subset examples; after uses all 96 labels. Its before/after difference therefore does not isolate initialization. GRU and RME use the same 96 examples before/after.",
    "physical_data": datasets, "retained_physical_overruns": overruns,
    "audit_passes": checks,
    "information_contract": {
        "shared_labels": "Every method trains on the same privileged six-coefficient projection labels at the window START",
        "mlp": "First valid frame of every shared window; full label set retained. Deployment comparison is 0 s only",
        "selection": "Learning rate, early stopping and 32/128-history choice use validation only; no final-test evidence is claimed here",
        "prior": "Head biases use training-only pooled label means. TRF's condition-specific training prior is data-matched but is not the same numerical initialization. Neural prior scales use validation logit residual RMS, then calibration selects a multiplier",
        "projection": "Bounded least squares in the volume-weighted D strain norm with unit-modulus weighting. Branch moduli are not included, so this is not a total-free-energy metric"
    },
    "limits": [
        "Validation reused for early stopping, LR/size selection and prior-residual scales; scales are empirical, not an independent calibration guarantee",
        "All data share one synthetic material, geometry and loading family. Absolute time is a deployed causal input; schedule-based priors may help within this scope",
        "Complete dataset cache identity records design/physics, while original source hashes are provenance. Cache replay is not an automatic assertion that current source code is numerically equivalent",
        "Three marginal penetration overruns are retained and must be reported; they do not count as successful passes of the 5 micrometer criterion"
    ]
}
output = run / "checks/training_summary.json"
output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps({"output": str(output), "table": [{k: r[k] for k in ("kind", "training_histories", "mean_coefficient_validation_mse", "mean_training_wall_seconds")} for r in table],
                  "retained_physical_overruns": len(overruns), "audit_pass_count": len(checks)}, indent=2))
