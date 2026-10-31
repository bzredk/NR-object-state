"""Exercise recorded ipympl widgets and one real, separately saved FEM rerun."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import time

import matplotlib
matplotlib.use("module://ipympl.backend_nbagg")
import ipywidgets as W
import numpy as np

from nr2d.inference_display import InferenceDashboard, _filter, _valid
from nr2d.inference_study import recompute_comparison


def verify(run, split="test", recompute=True):
    run = Path(run)
    started = time.perf_counter()
    dashboard = InferenceDashboard(run, recompute=recompute_comparison)
    records = [r for r in dashboard.rows if r.get("split") == split]
    if not records:
        raise ValueError(f"No {split} records exist; another split was not substituted.")
    study = json.loads((run / "study.json").read_text(encoding="utf-8"))
    available_units = {(r["history_id"], int(r["noise_repeat"])) for r in records}
    planned_histories = None
    if split == "test" and (run / "frozen.json").exists():
        planned_histories = len(json.loads((run / "frozen.json").read_text(encoding="utf-8"))["test_seeds"])
    planned_units = planned_histories * study["noise_repeats"] if planned_histories is not None else None
    history = min(r["history_id"] for r in records)
    dashboard.history.options = sorted({r["history_id"] for r in records})
    dashboard.history.value = history
    dashboard.repeat.value = min(int(r["noise_repeat"]) for r in records if r["history_id"] == history)
    dashboard.methods.value = ("TRF calibrated", "GRU")
    checks = []
    for channels in ("basic", "lateral"):
        for window in ("0.0", "0.2", "0.8", "contact"):
            for action in ("Slow press", "Fast press"):
                dashboard.channels.value, dashboard.window.value = channels, window
                dashboard.action.value = action
                dashboard.button.click()
                assert dashboard.last_error is None, dashboard.last_error
                assert "REPLAY" in dashboard.status.value
                selected = _filter(dashboard.rows, history_id=history,
                                   noise_repeat=dashboard.repeat.value,
                                   training_seed=dashboard.seed.value, training_size=dashboard.size.value,
                                   channels=channels, window=window, action=action)
                assert any(r["method"] == "TRF calibrated" and _valid(r) for r in selected)
                assert "TRF calibrated" in dashboard.details.value
                checks.append({"channels": channels, "window": window, "action": action,
                               "training_seed": dashboard.seed.value, "status": "passed"})
    dashboard.window.value, dashboard.channels.value = "0.8", "lateral"
    seed_checks = []
    for seed in list(dashboard.seed.options):
        dashboard.seed.value = seed
        dashboard.button.click()
        assert dashboard.last_error is None, dashboard.last_error
        assert "TRF calibrated" in dashboard.details.value
        selected = _filter(dashboard.rows, history_id=history, noise_repeat=dashboard.repeat.value,
                           training_seed=seed, training_size=dashboard.size.value,
                           channels="lateral", window="0.8", action=dashboard.action.value)
        baseline = [r for r in selected if r["method"] == "TRF calibrated" and _valid(r)]
        assert len(baseline) == 1 and int(baseline[0]["training_seed"]) == -1
        learned = [r for r in selected if r["method"] == "GRU" and _valid(r)]
        if not learned:
            assert "Unrun or failed" in dashboard.details.value
        seed_checks.append({"seed": seed, "shared_baseline_rows": len(baseline),
                            "learned_cache_available": bool(learned), "status": "passed"})
    dashboard.slider.value = dashboard.slider.max // 2
    cursor_x = float(dashboard.timeline[dashboard.slider.value])
    assert all(np.allclose(line.get_xdata(), [cursor_x, cursor_x]) for line in dashboard.cursors)
    assert isinstance(dashboard.fig.canvas, W.Widget)
    rerun = None
    if recompute:
        dashboard.seed.value = min(dashboard.seed.options)
        dashboard.window.value, dashboard.channels.value = "0.0", "basic"
        dashboard.action.value = "Slow press"
        metrics = run / "metrics.csv"
        before = hashlib.sha256(metrics.read_bytes()).hexdigest()
        outputs_before = set((run / "interactive").glob("*"))
        dashboard.recompute.value = True
        dashboard.button.click()
        assert dashboard.last_error is None, dashboard.last_error
        assert "RECOMPUTED" in dashboard.status.value
        after = hashlib.sha256(metrics.read_bytes()).hexdigest()
        assert before == after, "Published metrics changed during the interactive check."
        outputs = sorted(set((run / "interactive").glob("*")) - outputs_before)
        assert len(outputs) == 1 and (outputs[0] / "metrics.csv").exists()
        from nr2d.inference_display import load_metrics
        rerun_rows = load_metrics(outputs[0])
        assert all(any(r["method"] == m and _valid(r) for r in rerun_rows) for m in dashboard.methods.value)
        rerun = {"output": str(outputs[0]), "methods": list(dashboard.methods.value),
                 "channels": "basic", "window": "0.0", "action": "Slow press",
                 "published_metrics_sha256": before, "published_metrics_unchanged": True}
    result = {"status": "passed", "utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Functional native ipympl widget callbacks; not a frontend mouse-click test or final scientific analysis",
              "split": split, "history_id": history, "cached_conditions": checks,
              "independent_histories_available": len({r["history_id"] for r in records}),
              "history_noise_units_available": len(available_units),
              "planned_independent_histories": planned_histories,
              "planned_history_noise_units": planned_units,
              "all_planned_history_noise_units_available": len(available_units) == planned_units if planned_units is not None else None,
              "seed_checks": seed_checks, "cursor_and_widget_canvas": True,
              "real_recomputation": rerun, "wall_seconds": time.perf_counter()-started}
    output = run / "verification" / f"widget_{split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "history_id": history,
                      "cached_conditions": len(checks), "seed_checks": seed_checks,
                      "real_recomputation": rerun, "wall_seconds": result["wall_seconds"],
                      "record": str(output)}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", choices=["pilot", "old", "test"], default="test")
    parser.add_argument("--no-recompute", action="store_true", help="Verify recorded replay only")
    args = parser.parse_args()
    verify(args.run, args.split, not args.no_recompute)
