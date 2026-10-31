"""Compact replay bundles retain the complete statistical record."""
import csv

import pytest

from nr2d.inference_display import InferenceDashboard


def test_dashboard_lists_only_cached_histories_without_filtering_metrics(tmp_path, monkeypatch):
    rows = [dict(split="test", history_id=f"test_{index}", noise_repeat="0",
                 training_seed="11", training_size="128", method="GRU", action="Slow press",
                 curve_path=f"evaluation/test/test_{index}/repeat_0/curves/gru.npz")
            for index in (1, 2)]
    with (tmp_path / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    curve_directory = tmp_path / "evaluation/test/test_1/repeat_0/curves"
    curve_directory.mkdir(parents=True)
    monkeypatch.setattr(InferenceDashboard, "_run", lambda self, event: None)

    dashboard = InferenceDashboard(tmp_path)

    assert tuple(dashboard.history.options) == ("test_1",)
    assert {row["history_id"] for row in dashboard.rows} == {"test_1", "test_2"}
    assert "1 of 2 histories" in dashboard.widget.children[0].value
    assert tuple(dashboard.methods.value) == ("GRU",)

    curve_directory.rmdir()
    with pytest.raises(ValueError, match="No replay curves are available"):
        InferenceDashboard(tmp_path)
