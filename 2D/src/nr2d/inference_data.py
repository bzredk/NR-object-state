"""Whole-history data for the shared window-start state inference comparison."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import inspect
from multiprocessing import get_context
from pathlib import Path
import shutil
import time
import traceback

import numpy as np

from .physics import Config, SpatialModel, Trace
from .protocols import protocol
from .estimators import Observer
from .experiments import ROOT, environment, history_parameters, read_json, write_json
from .information import audit_trace, seed_for
from .inference_physics import WindowProblem


WINDOWS = ("0.0", "0.2", "0.8", "contact")
CHANNELS = ("basic", "lateral")
NOISE_ROOT_SEED = 20261030
PHYSICS_LIMITS = {"balance_N": 2e-7, "penetration_m": 5e-6, "strain": .06}


def _digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _history_rows(job):
    run, split, seed, repeats, physics = job
    run = Path(run)
    parent = f"{split}_{seed}"
    started = time.perf_counter()
    entry = {"id": parent, "seed": seed, "split": split, "status": "running",
             "failures": [], "conditions": []}
    rows = []
    try:
        model = SpatialModel(Config(**physics))
        parameters = history_parameters(seed)
        path = run / "histories" / f"{parent}.npz"
        if path.exists():
            trace = Trace.load(path)
            if trace.config != physics:
                raise ValueError("Cached history has different physical settings; use a new run")
            entry["physical_history_mode"] = "replay"
        else:
            commands, phases, _ = protocol(model.c, **parameters)
            trace = model.simulate(commands, phases,
                                   metadata={"id": parent, "seed": seed, "split": split,
                                             "parameters": parameters, "role": "learning data"})
            trace.save(path)
            entry["physical_history_mode"] = "computed"
        if not np.isclose(trace.time[-1], 1.4, atol=1e-10, rtol=0):
            raise ValueError("All data must end at the common 1.40 s prediction cut")
        entry.update(parameters=parameters, physics=audit_trace(trace, PHYSICS_LIMITS),
                     trace_path=str(path.relative_to(run)),
                     trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     prediction_cut=float(trace.time[-1]))
        # Physical over-limit episodes remain in the data; they are not filtered.
        if not entry["physics"]["passed"]:
            entry["failures"].append({"stage": "physics", "kind": "limit_exceeded",
                                      **entry["physics"]})
        observer = Observer(model)
        basis = observer.basis
        projections = {}
        for repeat in range(repeats):
            noise_seed = seed_for({"noise_root_seed": NOISE_ROOT_SEED}, seed, repeat)
            observed = observer.observations(trace, noise=1., seed=noise_seed)
            observation_path = run / "observations" / parent / f"repeat_{repeat}.npz"
            observation_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(observation_path, observed=observed, time=trace.time,
                                command=trace.command, noise_seed=noise_seed)
            observation_sha = _digest(observed)
            for window in WINDOWS:
                for channels in CHANNELS:
                    context = {"id": parent, "repeat": repeat, "window": window,
                               "channels": channels, "noise_seed": noise_seed}
                    try:
                        problem = WindowProblem(model, trace, observed, window,
                                                channels == "lateral", basis=basis)
                        if problem.begin not in projections:
                            projections[problem.begin] = problem.projection()
                        coefficients, projection = projections[problem.begin]
                        x, length = problem.inputs()
                        rows.append({"x": x, "lengths": length, "y": coefficients.astype(np.float32),
                                     "ids": parent, "windows": window, "channels": channels,
                                     "repeats": repeat, "noise_seeds": noise_seed,
                                     "begin": problem.begin, "end": problem.end,
                                     "projection_error": projection["relative_field_error"],
                                     "basis_rank": projection["basis_rank"],
                                     "observation_sha256": observation_sha})
                        entry["conditions"].append({**context, **problem.record(),
                                                     "begin_index": problem.begin,
                                                     "end_index": problem.end,
                                                     "observation_sha256": observation_sha,
                                                     "projection": projection})
                    except Exception as exc:
                        entry["failures"].append({**context, "stage": "window",
                                                  "error": str(exc), "traceback": traceback.format_exc()})
        entry["status"] = "completed" if len(rows) == repeats * len(WINDOWS) * len(CHANNELS) else "failed"
    except Exception as exc:
        entry.update(status="failed")
        entry["failures"].append({"stage": "history", "error": str(exc),
                                  "traceback": traceback.format_exc()})
    entry["wall_seconds"] = time.perf_counter() - started
    entry["row_count"] = len(rows)
    write_json(run / "data_records" / f"{parent}.json", entry)
    return rows, entry


def prepare_dataset(run, split, seeds, noise_repeats=3, workers=4):
    """Generate causal windows grouped by complete physical history.

    Every method shares the exact per-history/repeat observed array. Labels are
    privileged bounded projections of the physical state at the window start.
    The saved history has no future branch and no future signal enters x or y.
    """
    run = Path(run)
    seeds = list(map(int, seeds))
    if len(seeds) != len(set(seeds)):
        raise ValueError("A split cannot contain duplicate physical-history seeds")
    datasets = run / "datasets"
    datasets.mkdir(parents=True, exist_ok=True)
    (run / "histories").mkdir(exist_ok=True)
    physics_path = run / "physics.json"
    if not physics_path.exists():
        write_json(physics_path, read_json(ROOT / "configs/spatial_contact.json"))
    physics = read_json(physics_path)
    for other in datasets.glob("*.json"):
        if other.stem != split:
            overlap = set(seeds) & set(read_json(other).get("seeds", []))
            if overlap:
                raise ValueError(f"Physical history seeds overlap split {other.stem}: {sorted(overlap)}")
    identity = {"split": split, "seeds": seeds, "noise_repeats": noise_repeats,
                "noise_root_seed": NOISE_ROOT_SEED, "physics": physics,
                "windows": list(WINDOWS), "channels": list(CHANNELS)}
    dataset_path = datasets / f"{split}.npz"
    manifest_path = datasets / f"{split}.json"
    if dataset_path.exists() and manifest_path.exists():
        cached = read_json(manifest_path)
        if cached.get("identity") == identity and cached.get("status") == "completed":
            return dataset_path
        raise ValueError("Dataset cache has a different design or failed generation; use a new run or diagnose it")
    source = run / "source" / "data_prepare" / split
    source.mkdir(parents=True, exist_ok=True)
    source_hashes = {}
    for module in ("physics.py", "protocols.py", "estimators.py", "experiments.py",
                   "information.py", "inference_physics.py", "inference_data.py"):
        original = Path(__file__).with_name(module)
        shutil.copy2(original, source / module)
        source_hashes[module] = hashlib.sha256(original.read_bytes()).hexdigest()
    started = time.perf_counter()
    jobs = [(str(run), split, seed, noise_repeats, physics) for seed in seeds]
    results = {}
    if workers == 1:
        for job in jobs:
            results[job[2]] = _history_rows(job)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            futures = {pool.submit(_history_rows, job): job[2] for job in jobs}
            for future in as_completed(futures):
                seed = futures[future]
                results[seed] = future.result()
                print(f"{split}: {len(results)}/{len(seeds)} histories", flush=True)
    rows, histories = [], []
    for seed in seeds:
        samples, record = results[seed]
        rows.extend(samples)
        histories.append(record)
    arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]} if rows else {}
    np.savez_compressed(dataset_path, **arrays)
    manifest = {"identity": identity, "split": split, "seeds": seeds,
                "status": "completed" if all(h["status"] == "completed" for h in histories) else "failed",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "independent_history_count": len(histories), "example_count": len(rows),
                "noise_repeats": noise_repeats, "workers": workers,
                "wall_seconds": time.perf_counter()-started,
                "physics_limits": PHYSICS_LIMITS, "source_sha256": source_hashes,
                "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
                "normalization": "Fixed physical scales; closed channels zeroed before scaling",
                "supervision": "Simulator internal fields projected at window START; all learning models share labels",
                "projection_definition": inspect.getdoc(WindowProblem.projection),
                "environment": environment(), "histories": histories,
                "failures": [failure for history in histories for failure in history["failures"]]}
    write_json(manifest_path, manifest)
    print(f"{split}: {len(histories)} histories, {len(rows)} windows, {manifest['wall_seconds']:.2f} s", flush=True)
    if manifest["status"] != "completed":
        raise RuntimeError(f"Incomplete dataset preserved at {manifest_path}; inspect failure records before training")
    return dataset_path


def load_dataset(path, max_histories=None, snapshot_only=False):
    """Select deterministic whole-history prefixes before selecting snapshots."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    ids = list(dict.fromkeys(arrays["ids"].tolist()))
    selected = ids if max_histories is None else ids[:int(max_histories)]
    mask = np.isin(arrays["ids"], selected)
    if snapshot_only:
        mask &= arrays["lengths"] == 1
    result = {name: values[mask] for name, values in arrays.items()}
    result.update(path=str(path), selected_history_ids=selected,
                  independent_history_count=len(selected))
    return result
