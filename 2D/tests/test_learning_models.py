"""Causal information boundaries and executable neural training checks."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nr2d.learning_models import (StateRegressor, first_frame_inputs, load_model,
                                  predict_coefficients, train_model)


def samples(lengths):
    rng = np.random.default_rng(8)
    x = rng.normal(size=(len(lengths), 81, 12)).astype(np.float32)
    x[:, :, 7:] = 1.
    return x, np.asarray(lengths, np.int64)


@pytest.mark.parametrize("kind", ["gru", "mlp", "rme"])
def test_disabled_side_channels_and_padding_cannot_change_prediction(kind):
    torch.manual_seed(4)
    model = StateRegressor(kind)
    x, lengths = samples([1, 1] if kind == "mlp" else [1, 21])
    x[:, :, 10:12] = 0.
    changed = x.copy()
    changed[:, :, 3:5] = np.nan
    for i, length in enumerate(lengths):
        changed[i, length:] = np.inf
    a = predict_coefficients(model, x, lengths)
    b = predict_coefficients(model, changed, lengths)
    np.testing.assert_array_equal(a, b)
    assert np.all((a > 0.) & (a < 2.5))


def test_gru_does_not_retain_memory_between_calls():
    model = StateRegressor("gru")
    x, lengths = samples([81, 1])
    before = predict_coefficients(model, x[1:], lengths[1:])
    predict_coefficients(model, x[:1], lengths[:1])
    after = predict_coefficients(model, x[1:], lengths[1:])
    np.testing.assert_array_equal(before, after)


def test_rme_reuses_the_same_regression_block_three_times():
    model = StateRegressor("rme")
    calls = []
    hook = model.shared.register_forward_hook(lambda *args: calls.append(1))
    x, lengths = samples([1, 21, 81])
    prediction = predict_coefficients(model, x, lengths)
    hook.remove()
    assert len(calls) == 3
    assert prediction.shape == (3, 6)
    assert model.conv.kernel_size == (5,)
    assert model.attention.num_heads == 8


def test_snapshot_mlp_rejects_a_history_target():
    x, lengths = samples([21])
    with pytest.raises(ValueError, match="one-frame"):
        predict_coefficients(StateRegressor("mlp"), x, lengths)


def test_equal_label_snapshot_control_cannot_read_later_frames():
    model = StateRegressor("mlp")
    x, _ = samples([81, 21])
    changed = x.copy()
    changed[:, 1:] = np.nan
    a, lengths_a = first_frame_inputs(x)
    b, lengths_b = first_frame_inputs(changed)
    np.testing.assert_array_equal(predict_coefficients(model, a, lengths_a),
                                  predict_coefficients(model, b, lengths_b))
    assert len(a) == len(x)
    assert np.all(lengths_a == 1)


def test_training_overfits_a_small_diagnostic_and_roundtrips_checkpoint(tmp_path):
    x, lengths = samples([1] * 8)
    y = np.tile(np.array([.1, .3, .5, .7, .9, 1.1], np.float32), (8, 1))
    path = tmp_path / "diagnostic.pt"
    report = train_model(x, lengths, y, x, lengths, y, "gru", 11, path,
                         epochs=120, patience=30, batch_size=8, learning_rate=.01)
    assert report["best_validation_mse"] < 1e-3
    assert report["best_validation_mse"] < report["learning_curve"][0]["validation_mse"] / 10
    model = load_model(path)
    prediction = predict_coefficients(model, x, lengths)
    assert prediction.shape == y.shape
    assert np.all(np.asarray(model.report["prior_logit_scale"]) >= .15)
