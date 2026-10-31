"""Small supervised window-start state estimators sharing the six FEM coefficients.

Inputs are causal, right-padded observation windows. Scaling is supplied by the
dataset builder; this module never estimates normalization from deployment data.
The RME encoder preserves the released convolution/attention/shared-head pattern,
but adapts its inputs, sequence lengths and outputs to the present experiment.
"""
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


FEATURE_NAMES = ["position", "velocity", "force", "left", "right", "command",
                 "time", "position_mask", "velocity_mask", "force_mask",
                 "left_mask", "right_mask"]


class StateRegressor(nn.Module):
    """Estimate coefficients at the start of the supplied observation window."""

    def __init__(self, kind="gru", hidden=32, input_size=12, output_size=6,
                 max_length=81, coefficient_upper=2.5):
        super().__init__()
        if kind not in {"gru", "mlp", "rme"}:
            raise ValueError(f"Unknown state estimator: {kind}")
        self.config = dict(kind=kind, hidden=hidden, input_size=input_size,
                           output_size=output_size, max_length=max_length,
                           coefficient_upper=coefficient_upper)
        self.kind = kind
        if kind == "gru":
            self.gru = nn.GRU(input_size, hidden, num_layers=1, batch_first=True)
            self.head = nn.Linear(hidden, output_size)
        elif kind == "mlp":
            width = 2 * hidden
            self.mlp = nn.Sequential(nn.Linear(input_size, width), nn.ReLU(),
                                     nn.Linear(width, width), nn.ReLU(),
                                     nn.Linear(width, output_size))
        else:
            # Official RME: k=5/pad=2, no convolution activation or LayerNorm,
            # learned positions, attention output mean, one tied block 3 times.
            self.conv = nn.Conv1d(input_size, 64, kernel_size=5, padding=2)
            self.position = nn.Parameter(torch.randn(1, max_length, 64))
            self.attention = nn.MultiheadAttention(64, 8, batch_first=True)
            self.shared = nn.Sequential(nn.Linear(64, 256), nn.ReLU(),
                                        nn.Linear(256, 64), nn.Dropout(.1))
            self.head = nn.Linear(64, output_size)

    def forward(self, x, lengths):
        valid = torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]
        # torch.where removes even NaN/Inf values on a disabled sensor channel.
        observed = torch.where(x[:, :, 7:12] > .5, x[:, :, :5],
                               torch.zeros_like(x[:, :, :5]))
        x = torch.cat((observed, x[:, :, 5:]), dim=-1)
        x = torch.where(valid[:, :, None], x, torch.zeros_like(x))
        if self.kind == "gru":
            packed = pack_padded_sequence(x, lengths.detach().cpu(),
                                          batch_first=True, enforce_sorted=False)
            _, state = self.gru(packed)  # Every call starts with a zero hidden state.
            logits = self.head(state[-1])
        elif self.kind == "mlp":
            if torch.any(lengths != 1):
                raise ValueError("The current-observation MLP requires one-frame windows")
            logits = self.mlp(x[:, 0])
        else:
            if x.shape[1] > self.config["max_length"]:
                raise ValueError("Window exceeds the learned positional table")
            sequence = self.conv(x.transpose(1, 2)).transpose(1, 2)
            sequence = sequence + self.position[:, :x.shape[1]]
            attended, _ = self.attention(sequence, sequence, sequence,
                                         key_padding_mask=~valid, need_weights=False)
            state = (attended * valid[:, :, None]).sum(1) / lengths[:, None]
            for _ in range(3):
                state = self.shared(state)
            logits = self.head(state)
        return self.config["coefficient_upper"] * torch.sigmoid(logits)


def predict_coefficients(model, x, lengths, batch_size=256):
    """Return bounded point estimates without retained memory between windows."""
    model.eval()
    values = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            values.append(model(torch.as_tensor(x[start:start+batch_size], dtype=torch.float32, device=device),
                                torch.as_tensor(lengths[start:start+batch_size], dtype=torch.long, device=device))
                          .cpu().numpy())
    return np.concatenate(values)


def coefficient_logits(values, upper=2.5, epsilon=1e-4):
    """Finite coordinates for the common sigmoid-supported VI prior.

    Projection labels can lie on a bound. Only this logit diagnostic uses the
    declared epsilon interior approximation; training still uses original labels.
    """
    scaled = np.clip(np.asarray(values) / upper, epsilon, 1-epsilon)
    return np.log(scaled) - np.log1p(-scaled)


def first_frame_inputs(x):
    """Give the MLP one frame while retaining every shared window-start label."""
    result = np.zeros_like(x)
    result[:, 0] = np.asarray(x)[:, 0]
    return result, np.ones(len(result), dtype=np.int64)


def _data_hash(x, lengths, y):
    digest = hashlib.sha256()
    for array in (x, lengths, y):
        a = np.ascontiguousarray(array)
        digest.update(str((a.shape, a.dtype.str)).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def train_model(train_x, train_lengths, train_y, val_x, val_lengths, val_y,
                kind, seed, output_path, epochs=300, hidden=32, patience=40,
                batch_size=64, learning_rate=.003,
                input_transformation="Causal windows supplied by the dataset builder"):
    """Fit the shared privileged projection labels with validation early stopping.

    MLP calls must select one-frame samples. Physical histories, rather than
    windows or noise draws, must be split by the caller before invoking training.
    No future response enters this coefficient-regression objective.
    """
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    torch.use_deterministic_algorithms(True)
    train_x, val_x = np.asarray(train_x, np.float32), np.asarray(val_x, np.float32)
    train_lengths, val_lengths = np.asarray(train_lengths, np.int64), np.asarray(val_lengths, np.int64)
    train_y, val_y = np.asarray(train_y, np.float32), np.asarray(val_y, np.float32)
    if kind == "mlp" and (np.any(train_lengths != 1) or np.any(val_lengths != 1)):
        raise ValueError("Train the current-observation MLP on one-frame samples only")
    model = StateRegressor(kind=kind, hidden=hidden, input_size=train_x.shape[-1],
                           output_size=train_y.shape[-1], max_length=train_x.shape[1])
    # A sigmoid head otherwise starts near 1.25 even when all physical labels
    # are much smaller. The same training-only mean prior initializes every net.
    output_head = model.mlp[-1] if kind == "mlp" else model.head
    label_mean = train_y.mean(axis=0)
    with torch.no_grad():
        nn.init.normal_(output_head.weight, mean=0., std=.01)
        output_head.bias.copy_(torch.as_tensor(coefficient_logits(label_mean), dtype=torch.float32))
    x = torch.from_numpy(train_x)
    lengths = torch.from_numpy(train_lengths)
    labels = torch.from_numpy(train_y)
    vx, vl, vy = torch.from_numpy(val_x), torch.from_numpy(val_lengths), torch.from_numpy(val_y)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(int(seed) + 10000)
    curve = []
    best_loss, best_epoch, stale = float("inf"), -1, 0
    for epoch in range(int(epochs)):
        model.train()
        order = torch.randperm(len(x), generator=generator)
        loss_sum = 0.
        for start in range(0, len(x), batch_size):
            selected = order[start:start+batch_size]
            predicted = model(x[selected], lengths[selected])
            loss = (predicted-labels[selected]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(selected)
        model.eval()
        with torch.no_grad():
            validation = float((model(vx, vl)-vy).square().mean())
        curve.append({"epoch": epoch, "train_mse": loss_sum/len(x), "validation_mse": validation})
        if not np.isfinite(validation):
            raise FloatingPointError("Non-finite neural validation loss")
        if validation < best_loss - 1e-10:
            best_loss, best_epoch, stale = validation, epoch, 0
            best_state = deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    prediction = predict_coefficients(model, val_x, val_lengths)
    logit_error = coefficient_logits(val_y) - coefficient_logits(prediction)
    # RMS includes systematic residual bias; the standard deviation is recorded
    # separately rather than silently dropping that bias from prior uncertainty.
    rms = np.sqrt(np.mean(logit_error**2, axis=0))
    report = {
        "kind": kind, "seed": int(seed), "device": "cpu", "torch_version": str(torch.__version__),
        "torch_num_threads": torch.get_num_threads(),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "architecture": model.config, "feature_names": FEATURE_NAMES,
        "normalization": "Fixed physical scaling supplied by dataset builder; no deployment-data normalization",
        "input_transformation": input_transformation,
        "target": "Six bounded coefficients at observation-window start; shared simulator projection supervision",
        "loss": "Mean squared coefficient error; no future-response labels",
        "head_initialization": "Training-only coefficient mean transformed to logit bias; weights Normal(0, 0.01)",
        "initialization_mean_coefficients": label_mean.tolist(),
        "initialization_logit_epsilon": 1e-4,
        "train_example_count": len(train_x), "validation_example_count": len(val_x),
        "training_data_sha256": _data_hash(train_x, train_lengths, train_y),
        "validation_data_sha256": _data_hash(val_x, val_lengths, val_y),
        "learning_rate": learning_rate, "batch_size": batch_size, "epochs_requested": epochs,
        "epochs_completed": len(curve), "patience": patience, "best_epoch": best_epoch,
        "best_validation_mse": best_loss, "learning_curve": curve,
        "validation_logit_residual_sd": np.std(logit_error, axis=0).tolist(),
        "validation_logit_residual_rmse": rms.tolist(),
        "prior_logit_scale": np.maximum(.15, rms).tolist(),
        "prior_scale_rule": "Validation logit residual RMS with minimum 0.15; not a calibration guarantee",
        "logit_label_epsilon": 1e-4,
        "training_wall_seconds": time.perf_counter()-started,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.config, "state_dict": model.state_dict(), "report": report}, output_path)
    report["checkpoint_sha256"] = hashlib.sha256(output_path.read_bytes()).hexdigest()
    output_path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def load_model(path):
    """Load an experiment-owned checkpoint on CPU, ready for isolated inference."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    model = StateRegressor(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.report = checkpoint["report"]
    return model.eval()


def _fit_study_job(job):
    from .inference_data import load_dataset

    run, kind, size, seed, rate, phase = job
    run = Path(run)
    started = time.perf_counter()
    output = (run / "models/scouts" / f"{kind}_n{size}_s{seed}_lr{rate:g}.pt" if phase == "scout"
              else run / "models" / f"{kind}_n{size}_s{seed}.pt")
    result = {"kind": kind, "train_history_count": size, "seed": seed,
              "learning_rate": rate, "phase": phase, "checkpoint": str(output.relative_to(run))}
    try:
        train = load_dataset(run / "datasets/train.npz", max_histories=size)
        val = load_dataset(run / "datasets/validation.npz")
        transformation = "Causal observation window"
        if kind == "mlp":
            train["x"], train["lengths"] = first_frame_inputs(train["x"])
            val["x"], val["lengths"] = first_frame_inputs(val["x"])
            transformation = "First valid frame of every window; same complete window-start label set as GRU/RME"
        if output.exists() and output.with_suffix(".json").exists():
            load_model(output)
            report = json.loads(output.with_suffix(".json").read_text())
            assert report["learning_rate"] == rate
            assert report["seed"] == seed and report["epochs_requested"] == 300
            assert report["patience"] == 40 and report["batch_size"] == 64
            assert report["head_initialization"].startswith("Training-only")
            assert report["training_data_sha256"] == _data_hash(train["x"], train["lengths"], train["y"])
            assert report["validation_data_sha256"] == _data_hash(val["x"], val["lengths"], val["y"])
            assert hashlib.sha256(output.read_bytes()).hexdigest() == report["checkpoint_sha256"]
            result["mode"] = "replay"
        else:
            report = train_model(train["x"], train["lengths"], train["y"],
                                  val["x"], val["lengths"], val["y"], kind, seed, output,
                                  epochs=300, patience=40, batch_size=64, learning_rate=rate,
                                  input_transformation=transformation)
            result["mode"] = "trained"
        baseline = float(np.mean((val["y"]-train["y"].mean(0))**2))
        result.update(status="completed", report=report,
                      validation_training_mean_baseline_mse=baseline)
    except Exception as exc:
        result.update(status="failed", error=str(exc), traceback=traceback.format_exc())
    result["job_wall_seconds"] = time.perf_counter()-started
    return result


def train_study(run, workers=3):
    """Reproduce the two-LR scout and 32/128-history, three-seed comparison.

    Only the fixed training/validation datasets enter selection. Existing models
    replay only after their data hashes, training settings and file hash agree.
    A complete replay preserves the original measured training duration.
    """
    run = Path(run)
    kinds, sizes, seeds, rates = ("gru", "rme", "mlp"), (32, 128), (11, 23, 37), (.003, .0003)
    started = time.perf_counter()

    def save(path, record):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")

    previous_path = run / "training.json"
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())
        ready = all((run / "models" / f"{kind}_n{size}_s{seed}.{extension}").exists()
                    for kind in kinds for size in sizes for seed in seeds for extension in ("pt", "json"))
        if previous.get("status") == "completed" and ready:
            selected = previous["selected_learning_rates"]
            checks = [_fit_study_job((str(run), kind, size, seed, selected[kind], "main"))
                      for kind in kinds for size in sizes for seed in seeds]
            failures = [check for check in checks if check["status"] != "completed"]
            if failures:
                save(run / "training_replay_failures.json", failures)
                raise RuntimeError("Training checkpoint replay failed data/settings/hash checks")
            previous["last_replay_verification"] = {"status": "passed", "checkpoint_count": len(checks),
                                                     "wall_seconds": time.perf_counter()-started}
            save(previous_path, previous)
            return previous

    def execute(jobs, record, destination):
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            futures = [pool.submit(_fit_study_job, job) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                record["results"].append(result)
                record["wall_seconds"] = time.perf_counter()-started
                save(destination, record)
                message = {key: result[key] for key in ("kind", "train_history_count", "seed", "learning_rate", "phase", "status")}
                if result["status"] == "completed":
                    message.update(validation_mse=result["report"]["best_validation_mse"],
                                   epochs=result["report"]["epochs_completed"],
                                   training_seconds=result["report"]["training_wall_seconds"])
                else:
                    message["error"] = result["error"]
                print(json.dumps(message), flush=True)

    scout = {"status": "running", "workers": workers, "learning_rates": list(rates),
             "selection_split": "validation", "train_history_count": 32, "seed": 11, "results": []}
    execute([(str(run), kind, 32, 11, rate, "scout") for kind in kinds for rate in rates],
            scout, run / "training_scouts.json")
    selected = {}
    for kind in kinds:
        completed = [r for r in scout["results"] if r["kind"] == kind and r["status"] == "completed"]
        if not completed:
            raise RuntimeError(f"All validation-only LR scouts failed for {kind}")
        selected[kind] = min(completed, key=lambda r: r["report"]["best_validation_mse"])["learning_rate"]
    scout.update(status="completed", selected_learning_rates=selected)
    save(run / "training_scouts.json", scout)
    record = {"status": "running", "workers": workers, "train_sizes": list(sizes),
              "training_seeds": list(seeds), "selected_learning_rates": selected,
              "hyperparameters": {"epochs": 300, "patience": 40, "batch_size": 64},
              "results": [], "selected_scout_reuse": [], "final_test_used": False}
    for kind in kinds:
        chosen = next(r for r in scout["results"] if r["kind"] == kind and r["learning_rate"] == selected[kind] and r["status"] == "completed")
        source = run / chosen["checkpoint"]
        destination = run / "models" / f"{kind}_n32_s11.pt"
        shutil.copy2(source, destination)
        shutil.copy2(source.with_suffix(".json"), destination.with_suffix(".json"))
        record["selected_scout_reuse"].append(str(destination.relative_to(run)))
    execute([(str(run), kind, size, seed, selected[kind], "main")
             for kind in kinds for size in sizes for seed in seeds], record, previous_path)
    if any(r["status"] != "completed" for r in record["results"]):
        record["status"] = "failed"
        save(previous_path, record)
        raise RuntimeError("Some training fits failed; inspect training.json")
    record["status"] = "completed"
    record["actual_training_fit_count"] = len(scout["results"]) + len(record["results"]) - len(record["selected_scout_reuse"])
    record["mean_validation_mse"] = {kind: {
        str(size): float(np.mean([r["report"]["best_validation_mse"] for r in record["results"]
                                 if r["kind"] == kind and r["train_history_count"] == size]))
        for size in sizes} for kind in kinds}
    record["selected_sizes"] = {kind: 128 if loss["128"] <= loss["32"] else 32
                                for kind, loss in record["mean_validation_mse"].items()}
    record["wall_seconds"] = time.perf_counter()-started
    save(previous_path, record)
    return record
