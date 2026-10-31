"""Recorded estimator comparisons for Matplotlib, VS Code and CLI exports."""
from pathlib import Path
from html import escape
import argparse
import csv
import json
import time

import numpy as np
import matplotlib.pyplot as plt

from .plotting import style, export_figure


METHODS = ["TRF published", "TRF calibrated", "GRU", "RME prior", "RME prior mean", "RME-inspired VI",
           "Fixed prior VI", "Neural-init TRF", "Snapshot MLP", "Projected truth",
           "Full replay", "True state"]
COLORS = dict(zip(METHODS, ["#969CA2", "#48515B", "#C27838", "#2866AB", "#7198C5", "#178479",
                          "#8267A0", "#698A43", "#B46179", "#6E97C5", "#39736B", "#202C3B"]))
REFERENCES = {"Projected truth", "Full replay", "True state"}
NUMBER_FIELDS = {"force_rmse_mN", "marker_rmse_um", "encode_seconds", "history_seconds",
                 "vi_seconds", "predict_seconds", "total_seconds", "force_coverage90",
                 "force_interval_width_mN", "force_crps_mN", "coefficient_coverage90",
                 "noisy_force_coverage90", "noisy_force_interval_width_mN",
                 "force_contact_coverage90", "force_contact_interval_width_mN",
                 "posterior_physics_violation_rate"}


def load_metrics(run, filename="metrics.csv"):
    """Read raw per-history rows without replacing failures by successful samples."""
    with (Path(run) / filename).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for name in NUMBER_FIELDS:
            if name in row:
                row[name] = float(row[name]) if row[name] not in ("", None) else np.nan
        if row.get("physics") and isinstance(row["physics"], str):
            row["physics"] = json.loads(row["physics"])
    return rows


def _window(value):
    return "contact" if str(value) == "contact" else f"{float(value):g}"


def _filter(rows, **conditions):
    selected = []
    for row in rows:
        keep = True
        for name, wanted in conditions.items():
            if wanted is None:
                continue
            actual = row.get(name, "")
            if name in ("training_seed", "training_size") and str(actual) in ("", "0", "-1"):
                continue  # Shared classical/reference estimate, not another training seed.
            if name == "action" and actual == "inference failure":
                continue  # An inference failure prevents every future-action rollout.
            if name == "window":
                keep &= _window(actual) == _window(wanted)
            else:
                keep &= str(actual) == str(wanted)
        if keep:
            selected.append(row)
    return selected


def _valid(row, metric="force_rmse_mN"):
    return str(row.get("success", "true")).lower() not in ("false", "0", "failed") and np.isfinite(row.get(metric, np.nan))


def _methods(rows, references=True):
    available = {row["method"] for row in rows}
    ordered = [m for m in METHODS if m in available] + sorted(available - set(METHODS))
    return ordered if references else [m for m in ordered if m not in REFERENCES]


def _history_values(rows, metric):
    from .inference_analysis import history_values
    return history_values([row for row in rows if _valid(row, metric)], metric)


def _interval(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.nan, np.array([np.nan, np.nan])
    if len(values) == 1:
        return float(values[0]), np.array([np.nan, np.nan])
    rng = np.random.default_rng(20261030)
    means = values[rng.integers(len(values), size=(2000, len(values)))].mean(axis=1)
    return float(values.mean()), np.quantile(means, [.025, .975])


def _curve(run, row):
    with np.load(Path(run) / row["curve_path"], allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def response_figure(run, history_id, noise_repeat=0, training_seed=11, channels="lateral",
                    window="0.8", action="Slow press", methods=None, rows=None, fig=None,
                    training_size=None):
    """Plot recorded force and shape responses for one shared information condition."""
    style()
    rows = load_metrics(run) if rows is None else rows
    rows = _filter(rows, history_id=history_id, noise_repeat=noise_repeat, training_seed=training_seed,
                   channels=channels, window=window, action=action, training_size=training_size)
    rows = [r for r in rows if _valid(r) and (methods is None or r["method"] in methods)]
    if fig is None:
        fig = plt.figure(figsize=(14, 4.9), layout="constrained")
    else:
        fig.clear()
    axes = fig.subplots(1, 3)
    if not rows:
        for ax in axes:
            ax.text(.5, .5, "No recorded prediction for this selection", ha="center", transform=ax.transAxes)
        return fig, rows
    first = _curve(run, rows[0])
    t = first["time"] - first["time"][0]
    axes[0].plot(t, first["true_force"], color="#14191E", lw=2.5, label="True response")
    axes[2].plot(t, first["true_markers"].mean(axis=1) * 1e6, color="#14191E", lw=2.5)
    shown, has_band = set(), False
    for row in rows:
        method = row["method"]
        if method in shown:
            continue
        shown.add(method)
        curve = _curve(run, row)
        color = COLORS.get(method, "#576979")
        axes[0].plot(t, curve["predicted_force"], color=color, lw=1.8, label=method,
                     ls=":" if method in REFERENCES else "-")
        axes[1].plot(t, (curve["predicted_force"] - curve["true_force"]) * 1e3, color=color, lw=1.8)
        axes[2].plot(t, curve["predicted_markers"].mean(axis=1) * 1e6, color=color, lw=1.8)
        if "force_lower" in curve and "force_upper" in curve:
            has_band = True
            axes[0].fill_between(t, curve["force_lower"], curve["force_upper"], color=color, alpha=.13)
    axes[0].set(ylabel="Contact force (N)", title="Command-conditioned response")
    axes[1].set(ylabel="Force error (mN)", title="Prediction minus true response")
    axes[2].set(ylabel="Mean outward lateral motion (um)", title="Side-marker response")
    axes[1].axhline(0, color="#555555", lw=.8)
    for ax in axes:
        ax.set_xlabel("Time after prediction (s)")
        ax.axvline(0, color="#7B7E83", ls="--", lw=1)
    axes[0].legend(fontsize=8, loc="best")
    sensor = "with lateral" if channels == "lateral" else "without lateral"
    sizes = sorted({int(r["training_size"]) for r in rows if int(r.get("training_size") or 0) > 0})
    training = f", train N={','.join(map(str, sizes))}, seed {training_seed}" if sizes else ""
    fig.suptitle(f"{history_id}, noise {noise_repeat}{training} | {sensor}, {_window(window)} s | {action}")
    point_note = "VI line: posterior state-mean rollout. " if any("VI" in m for m in shown) else "Shared-FEM point predictions. "
    band_note = "Shading: 90% noiseless force interval induced by state uncertainty." if has_band else "No probability band was recorded for this selection."
    fig.supxlabel(point_note + band_note, fontsize=10)
    return fig, rows


def information_figure(run, action="Slow press", split="test", rows=None):
    """Mean errors and three prespecified paired information gains, grouped by history."""
    style()
    rows = load_metrics(run) if rows is None else rows
    rows = _filter(rows, action=action, split=split)
    methods = _methods(rows, references=False)
    conditions = [(side, window) for side in ("basic", "lateral") for window in ("0", ".2", ".8")]
    matrix = np.full((len(methods), len(conditions)), np.nan)
    counts = np.zeros(matrix.shape, dtype=int)
    failed_cells = np.zeros(matrix.shape, dtype=bool)
    for i, method in enumerate(methods):
        for j, (side, window) in enumerate(conditions):
            condition_rows = _filter(rows, method=method, channels=side, window=window)
            values = _history_values(condition_rows, "force_rmse_mN")
            failed_cells[i, j] = any(not _valid(r) for r in condition_rows)
            if values:
                matrix[i, j] = np.mean(list(values.values()))
                counts[i, j] = len(values)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5., len(methods) * .58)), layout="constrained",
                             gridspec_kw={"width_ratios": [1.2, 1.]})
    finite = matrix[np.isfinite(matrix)]
    vmax = max(float(np.quantile(finite, .9)), .01) if len(finite) else 1.
    im = axes[0].imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap="Blues", vmin=0, vmax=vmax)
    axes[0].set_xticks(range(6), ["0", "0.2", "0.8", "0", "0.2", "0.8"])
    axes[0].set_yticks(range(len(methods)), methods)
    axes[0].set(xlabel="Window (s): first three without lateral; last three with lateral", title="Force RMSE (mN), complete-history means")
    for i in range(len(methods)):
        for j in range(6):
            axes[0].text(j, i, f"{matrix[i,j]:.2f}\nn={counts[i,j]}" if np.isfinite(matrix[i,j]) else
                         ("failed" if failed_cells[i,j] else "unrun"),
                         ha="center", va="center", fontsize=9,
                         color="white" if np.isfinite(matrix[i,j]) and matrix[i,j] > .6 * vmax else "#222222")
    clipped = bool(len(finite) and np.max(finite) > vmax)
    fig.colorbar(im, ax=axes[0], shrink=.75, label="Force RMSE (mN)", extend="max" if clipped else "neither")
    contrasts = [("Adding lateral at snapshot", ("basic", "0"), ("lateral", "0"), "o", "#2866AB"),
                 ("Adding 0.2 s with lateral", ("lateral", "0"), ("lateral", ".2"), "s", "#C27838"),
                 ("Adding 0.8 s with lateral", ("lateral", "0"), ("lateral", ".8"), "^", "#178479")]
    for label, before, after, marker, color in contrasts:
        from .inference_analysis import paired_values
        shift = {"o": -.2, "s": 0., "^": .2}[marker]
        for i, method in enumerate(methods):
            a = [r for r in _filter(rows, method=method, channels=before[0], window=before[1]) if _valid(r)]
            b = [r for r in _filter(rows, method=method, channels=after[0], window=after[1]) if _valid(r)]
            paired, _ = paired_values(a, b)
            if paired:
                mean, bounds = _interval([p["gain_mN"] for p in paired])
                axes[1].errorbar(mean, i + shift, xerr=[[mean-bounds[0]], [bounds[1]-mean]],
                                 fmt=marker, color=color, capsize=3, label=label if i == 0 else None)
    axes[1].set_yticks(range(len(methods)), methods)
    axes[1].set_ylim(len(methods)-.5, -1.7)
    axes[1].axvline(0, color="#555555", ls="--", lw=1)
    axes[1].set(xlabel="Paired error reduction (mN); positive is better", title="History-bootstrap 95% intervals")
    axes[1].legend(fontsize=8)
    failures = sum(not _valid(row) for row in rows)
    fig.suptitle(f"{split} histories | {action} | {failures} failed/nonfinite rows retained in metrics.csv")
    scale_note = f" Color saturates above {vmax:.2f} mN; labels show unclipped errors." if clipped else ""
    fig.supxlabel("Noise repeats and training seeds are averaged within history. Intervals require at least two histories.\n"
                  "Paired contrasts share history, noise, and training seed." + scale_note, fontsize=10)
    return fig


def latency_figure(run, action="Slow press", split="test", rows=None):
    """Join held-out accuracy with serial timing when available, not pool wall time."""
    style()
    rows = load_metrics(run) if rows is None else rows
    rows = _filter(rows, action=action, split=split)
    dedicated = (Path(run) / "latency.csv").exists()
    timing = _filter(load_metrics(run, "latency.csv"), action=action, temperature="warm") if dedicated else rows
    timing_label = "Median online phase-compute time (s), serial warm batch=1" if dedicated else "Parallel evaluation phase time (s), not dedicated latency"
    methods = _methods(rows, references=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), layout="constrained")
    markers = {"0": "o", "0.2": "s", "0.8": "^", "contact": "D"}
    for method in methods:
        for side in ("basic", "lateral"):
            for window, marker in markers.items():
                selected = [r for r in _filter(rows, method=method, channels=side, window=window) if _valid(r)]
                measured = _filter(timing, method=method, channels=side, window=window)
                times = [r["total_seconds"] for r in measured if _valid(r) and np.isfinite(r.get("total_seconds", np.nan)) and r["total_seconds"] > 0]
                values = _history_values(selected, "force_rmse_mN")
                if times and values:
                    axes[0].scatter(np.median(times), np.mean(list(values.values())), marker=marker, s=65,
                                    edgecolor=COLORS.get(method, "#576979"),
                                    facecolor=COLORS.get(method, "#576979") if side == "lateral" else "none")
        axes[0].plot([], [], "o", color=COLORS.get(method, "#576979"), label=method)
    accuracy_label = {"test": "New-test", "pilot": "Pilot", "old": "Old reanalysis"}.get(split, split)
    axes[0].set(xscale="log", xlabel=timing_label,
                ylabel=f"{accuracy_label} future force RMSE (mN)", title=f"{accuracy_label} accuracy and online phase compute")
    axes[0].legend(fontsize=8)
    keys = ["encode_seconds", "history_seconds", "vi_seconds", "predict_seconds"]
    labels = ["Encoding / initialization", "Fit / history replay", "VI refinement", "Future + Monte Carlo"]
    colors = ["#9BACBF", "#2866AB", "#C27838", "#178479"]
    left = np.zeros(len(methods))
    for key, label, color in zip(keys, labels, colors):
        values = []
        for method in methods:
            selected = _filter(timing, method=method, channels="lateral", window=".8")
            finite = [r.get(key, np.nan) for r in selected if _valid(r) and np.isfinite(r.get(key, np.nan))]
            values.append(float(np.median(finite)) if finite else 0.)
        axes[1].barh(range(len(methods)), values, left=left, label=label, color=color)
        left += values
    axes[1].set_yticks(range(len(methods)), methods)
    for i, value in enumerate(left):
        if value == 0.:
            axes[1].text(0., i, "not measured", va="center", fontsize=9, color="#777777")
    axes[1].invert_yaxis()
    axes[1].set(xlabel="Sum of median component times (s)", title="0.8 s with lateral, serial warm components" if dedicated else "0.8 s with lateral, pool evaluation components")
    axes[1].legend(fontsize=8)
    timing_note = "Timing: separate serial warm pilot suite." if dedicated else "Pool times are not dedicated online-latency measurements."
    app_note = " Application cold/resident-warm totals: latency_startup.csv." if (Path(run) / "latency_startup.csv").exists() else ""
    fig.supxlabel("Open/filled: no lateral/lateral. Circle/square/triangle/diamond: 0/0.2/0.8 s/contact. " + timing_note + "\n"
                  "Phase compute excludes context setup and experiment I/O." + app_note + " Offline training is separate.", fontsize=10)
    return fig


def uncertainty_figure(run, action="Slow press", split="test", rows=None):
    """Coverage, width, proper scoring and an explicitly post-hoc failure example."""
    style()
    rows = load_metrics(run) if rows is None else rows
    rows = _filter(rows, action=action, split=split)
    probability = [r for r in rows if _valid(r, "force_coverage90")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    if not probability:
        for ax in axes.flat:
            ax.text(.5, .5, "No recorded probability predictions", ha="center", transform=ax.transAxes)
        return fig
    groups = []
    for method in _methods(probability):
        for side, window in (("basic", ".2"), ("lateral", ".2"), ("lateral", ".8")):
            selected = _filter(probability, method=method, channels=side, window=window)
            if selected:
                groups.append((method, side, window, selected))
    for ax, field, title, ylabel in [(axes[0,0], "force_coverage90", "Noiseless force coverage (90% intervals)", "Fraction of force samples covered"),
                                     (axes[0,1], "force_interval_width_mN", "Predictive interval width", "Mean 90% width (mN)"),
                                     (axes[1,0], "force_crps_mN", "Predictive CRPS", "CRPS (mN), lower is better")]:
        for i, (method, side, window, selected) in enumerate(groups):
            values = _history_values(selected, field)
            if values:
                mean, bounds = _interval(list(values.values()))
                x = i-.10 if field == "force_coverage90" else i
                ax.errorbar(x, mean, yerr=[[mean-bounds[0]], [bounds[1]-mean]], fmt="o",
                            color=COLORS.get(method, "#576979"), capsize=3)
            if field == "force_coverage90":
                contact_values = _history_values(selected, "force_contact_coverage90")
                if contact_values:
                    mean, bounds = _interval(list(contact_values.values()))
                    ax.errorbar(i+.10, mean, yerr=[[mean-bounds[0]], [bounds[1]-mean]], fmt="^",
                                markerfacecolor="none", color=COLORS.get(method, "#576979"), capsize=3)
        ax.set_xticks(range(len(groups)), [f"{m}\n{'lateral' if s=='lateral' else 'no lateral'}, {_window(w)} s" for m,s,w,_ in groups], rotation=22, ha="right", fontsize=8)
        ax.set(title=title, ylabel=ylabel)
    axes[0,0].axhline(.9, color="#555555", ls="--", label="Nominal 90%")
    axes[0,0].plot([], [], "o", color="#555555", label="Full future horizon")
    if any(_valid(r, "force_contact_coverage90") for r in probability):
        axes[0,0].plot([], [], "^", color="#555555", markerfacecolor="none", label="Contact samples only")
    axes[0,0].set_ylim(0, 1.03)
    axes[0,0].legend(fontsize=8)
    worst = min(probability, key=lambda r: r["force_coverage90"])
    curve = _curve(run, worst)
    t = curve["time"] - curve["time"][0]
    axes[1,1].plot(t, curve["true_force"], color="#14191E", lw=2, label="True response")
    axes[1,1].plot(t, curve["predicted_force"], color=COLORS.get(worst["method"], "#178479"), label="State-mean rollout")
    if "force_lower" in curve:
        axes[1,1].fill_between(t, curve["force_lower"], curve["force_upper"], alpha=.2, color="#178479", label="90% interval")
    axes[1,1].set(xlabel="Time after prediction (s)", ylabel="Contact force (N)",
                   title=f"Lowest recorded coverage (post-hoc diagnostic)\n{worst['history_id']}, {worst['method']}, {worst['channels']}, {worst['window']} s")
    axes[1,1].legend(fontsize=8)
    audited = [r for r in probability if isinstance(r.get("physics"), dict)]
    overruns = sum(r["physics"].get("passed") is False for r in audited)
    physics_note = f" Retained simulated-forecast limit overruns: {overruns}/{len(audited)} records." if audited else ""
    mc_note = ""
    mc_path = Path(run) / "verification/finite_mc_reference.json"
    if mc_path.exists():
        reference = json.loads(mc_path.read_text(encoding="utf-8"))
        members = reference["design"]["members"]
        coverage = reference["results"]["coverage"]["mean"]
        mc_note = f"\nFinite MC ({members} members): Gaussian-reference coverage {coverage:.2%}; a diagnostic, not a FEM coverage correction. Analytic coefficient intervals are unaffected."
    fig.supxlabel("Full-horizon coverage includes zero-force flight; contact-only coverage is shown when recorded. Sensor-noise uncertainty is separate.\n"
                  "Coefficient coverage in run records targets the declared field projection." + physics_note + mc_note, fontsize=9)
    return fig


def export_inference_figures(run, action="Slow press", split="test", output=None):
    """Export four views and provenance from recorded metrics and response curves."""
    run = Path(run)
    rows = load_metrics(run)
    out = Path(output) if output is not None else run / "figures"
    out.mkdir(parents=True, exist_ok=True)
    snapshot_rows = _filter(rows, action=action, split=split)
    fields = list(dict.fromkeys(key for row in snapshot_rows for key in row))
    snapshot = out / "metrics_snapshot.csv"
    with snapshot.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                          for key, value in row.items()} for row in snapshot_rows)
    test = _filter(rows, split=split, action=action, channels="lateral", window=".8")
    valid = [r for r in test if _valid(r)]
    if not valid:
        raise ValueError(f"No valid {split}/{action}/lateral/0.8 s response is recorded; no example was manufactured.")
    first = min(valid, key=lambda r: (r["history_id"], int(r.get("noise_repeat", 0)), int(r.get("training_seed") or 0)))
    inferred_seeds = sorted({int(r["training_seed"]) for r in valid if str(r.get("training_seed", "")) not in ("", "0", "-1")})
    seed = inferred_seeds[0] if inferred_seeds else 0
    figures = [response_figure(run, first["history_id"], first["noise_repeat"], seed, action=action, rows=rows)[0],
               information_figure(run, action=action, split=split, rows=rows),
               latency_figure(run, action=action, split=split, rows=rows),
               uncertainty_figure(run, action=action, split=split, rows=rows)]
    stems = ["01_estimator_responses", "02_information_by_estimator", "03_accuracy_and_latency", "04_probability_checks"]
    snapshot_source = str(snapshot.relative_to(run)) if snapshot.is_relative_to(run) else str(snapshot.resolve())
    sources = ["metrics.csv", snapshot_source, "summary.json", "curves/"] + (["latency.csv"] if (run / "latency.csv").exists() else [])
    if (run / "verification/finite_mc_reference.json").exists():
        sources.append("verification/finite_mc_reference.json")
    if (run / "latency_startup.csv").exists():
        sources.append("latency_startup.csv")
    for fig, stem in zip(figures, stems):
        export_figure(fig, out / stem, run, sources)
        plt.close(fig)
    return [str(out / (stem + ".png")) for stem in stems]


class InferenceDashboard:
    """Dropdowns select cached data; only the explicit button can recompute FEM."""

    def __init__(self, run, recompute=None):
        import ipywidgets as W
        from .widgets import figure_widget
        self.run, self.recompute_callback = Path(run), recompute
        self.rows = load_metrics(self.run)
        self.fig, self.cursors, self.last_error = None, [], None
        test = [r for r in self.rows if r.get("split") == "test"] or self.rows
        total_histories = len({r["history_id"] for r in test})
        curve_directories = {Path(r["curve_path"]).parent for r in test if r.get("curve_path")}
        cached_directories = {directory for directory in curve_directories if (self.run / directory).is_dir()}
        test = [r for r in test if r.get("curve_path") and Path(r["curve_path"]).parent in cached_directories]
        if not test:
            raise ValueError("No replay curves are available for this run. Generate predictions before opening the dashboard.")
        replay_histories = len({r["history_id"] for r in test})
        self.history = W.Dropdown(options=sorted({r["history_id"] for r in test}), description="History")
        self.repeat = W.Dropdown(options=sorted({int(r["noise_repeat"]) for r in test}), description="Noise draw")
        seeds = sorted({int(r["training_seed"]) for r in test if str(r.get("training_seed", "")) not in ("", "0", "-1")})
        sizes = sorted({int(r["training_size"]) for r in test if str(r.get("training_size", "")) not in ("", "0", "-1")})
        self.seed = W.Dropdown(options=seeds or [0], description="Train seed")
        self.size = W.Dropdown(options=sizes or [0], value=max(sizes) if sizes else 0, description="Train N")
        self.channels = W.Dropdown(options=[("With lateral", "lateral"), ("Without lateral", "basic")], description="Sensors")
        self.window = W.Dropdown(options=[("Current sample", "0.0"), ("Recent 0.2 s", "0.2"), ("Recent 0.8 s", "0.8"), ("Earlier contact + propagation", "contact")], value="0.8", description="Window")
        actions = [a for a in ("Slow press", "Fast press") if any(r["action"] == a for r in test)]
        self.action = W.Dropdown(options=actions, value="Slow press" if "Slow press" in actions else actions[0], description="Future")
        available = _methods(test)
        preferred = [m for m in ("TRF calibrated", "GRU", "RME-inspired VI", "True state") if m in available]
        self.methods = W.SelectMultiple(options=available, value=tuple(preferred or available), description="Methods", rows=min(6, len(available)), layout=W.Layout(width="320px"))
        self.recompute = W.Checkbox(description="Recompute selected comparison", value=False, indent=False, layout=W.Layout(width="290px"))
        self.button = W.Button(description="Run selection", icon="play", button_style="primary")
        self.play = W.Play(value=0, min=0, max=100, interval=50, description="Replay")
        self.slider = W.IntSlider(value=0, min=0, max=100, description="Future frame", continuous_update=True, layout=W.Layout(width="600px"))
        self.link = W.jslink((self.play, "value"), (self.slider, "value"))
        self.status, self.details, self.plot = W.HTML(), W.HTML(), W.VBox()
        self.progress = W.IntProgress(value=0, max=3, description="Progress")
        self.figure_widget = figure_widget
        self.button.on_click(self._run)
        self.slider.observe(self._cursor, names="value")
        self.widget = W.VBox([W.HTML(f"Replay cache: {replay_histories} of {total_histories} histories. "
                                    "Static summaries use all recorded histories. Select information and methods, then press Run. "
                                    "Sliders replay; they never launch inference."),
                              W.HBox([self.history, self.repeat]), W.HBox([self.seed, self.size]),
                              W.HBox([self.channels, self.window, self.action]), self.methods,
                              W.HBox([self.button, self.recompute]), self.progress, self.status,
                              W.HBox([self.play, self.slider]), self.plot, self.details])
        self._run(None)

    def _cursor(self, _):
        if self.fig is not None and self.cursors:
            x = float(self.timeline[min(self.slider.value, len(self.timeline)-1)])
            for line in self.cursors:
                line.set_xdata([x, x])
            self.fig.canvas.draw_idle()

    def _run(self, _):
        from .widgets import table
        self.button.disabled = True
        self.last_error, self.progress.value = None, 0
        start = time.perf_counter()
        try:
            plot_run, rows = self.run, self.rows
            mode = "REPLAY: cached predictions; no solver call"
            if self.recompute.value:
                if self.recompute_callback is None:
                    raise RuntimeError("No recomputation callback is configured.")
                self.status.value = "<b>RECOMPUTING</b>: inference and shared FEM forecasts"
                result = self.recompute_callback(self.run, self.history.value, self.repeat.value, self.seed.value,
                                                  self.channels.value, self.window.value, self.action.value, list(self.methods.value))
                plot_run, rows = Path(result["run"]), result["rows"]
                mode = "RECOMPUTED: new interactive record; published results retained"
            self.progress.value = 1
            with plt.ioff():
                self.fig, selected = response_figure(plot_run, self.history.value, self.repeat.value, self.seed.value,
                    self.channels.value, self.window.value, self.action.value, list(self.methods.value), rows, self.fig, self.size.value)
            if not self.plot.children:
                self.plot.children = (self.figure_widget(self.fig),)
            self.cursors = []
            if selected:
                trace = _curve(plot_run, selected[0])
                self.timeline = trace["time"] - trace["time"][0]
                self.slider.max = self.play.max = len(self.timeline)-1
                self.slider.value = self.play.value = 0
                self.cursors = [ax.axvline(0, color="#B65E69", lw=.8) for ax in self.fig.axes]
            self.fig.canvas.draw_idle()
            self.progress.value = 2
            self.details.value = table(["Method", "Force RMSE (mN)", "Lateral RMSE (um)", "Recorded phase compute (s)"],
                [[r["method"], f"{r['force_rmse_mN']:.3f}", f"{r.get('marker_rmse_um', np.nan):.3f}", f"{r.get('total_seconds', np.nan):.3f}"] for r in selected])
            available = {r["method"] for r in selected}
            absent = [name for name in self.methods.value if name not in available]
            condition = _filter(rows, history_id=self.history.value, noise_repeat=self.repeat.value,
                                training_seed=self.seed.value, training_size=self.size.value,
                                channels=self.channels.value, window=self.window.value)
            failed = [r for r in condition if not _valid(r) and r["method"] in self.methods.value
                      and r.get("action") in (self.action.value, "inference failure")]
            if absent:
                self.details.value += "<p>Unrun or failed in this condition: " + escape(", ".join(absent)) + ".</p>"
            if failed:
                self.details.value += table(["Failed method", "Recorded error"],
                    [[r["method"], r.get("error", "See the run failure record")] for r in failed])
            self.details.value += table(["Method", "Optimizer status", "Forecast physical check"],
                [[r["method"], r.get("optimizer_status", "unreported"),
                  "passed" if r.get("physics", {}).get("passed") is True else
                  ("limit overrun (retained)" if r.get("physics", {}).get("passed") is False else "not recorded")]
                 for r in selected])
            self.details.value += "<p>VI point curve propagates the mean coefficient state; the interval propagates posterior samples. Privileged references receive simulator state or initial-state information.</p>"
            self.progress.value = 3
            self.status.value = f"<b>{mode}</b> | {time.perf_counter()-start:.2f} s | training histories: {self.size.value}; seed: {self.seed.value}"
        except Exception as exc:
            self.last_error = exc
            self.status.value = "<b>FAILED</b>: " + escape(str(exc))
        finally:
            self.button.disabled = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", choices=["pilot", "old", "test"], default="test")
    parser.add_argument("--action", choices=["Slow press", "Fast press"], default="Slow press")
    parser.add_argument("--output", type=Path, help="Separate preview directory; defaults to RUN/figures")
    args = parser.parse_args()
    for figure in export_inference_figures(args.run, action=args.action, split=args.split, output=args.output):
        print(figure)


if __name__ == "__main__":
    main()
