"""Probability support and complete physical-likelihood gradient checks."""
from pathlib import Path

import numpy as np
import torch

from nr2d.variational import (UPPER, bounded_gaussian_mean, fem_nll,
                             gaussian_kl, infer_vi, negative_elbo)


class QuadraticProblem:
    def __init__(self, informative=True):
        self.target = np.linspace(.25, .75, 6)
        self.informative = informative
        self.calls = 0

    def nll(self, coefficients, gradient=False):
        self.calls += 1
        delta = np.asarray(coefficients) - self.target
        value = .5 * delta @ delta if self.informative else 4.2
        derivative = delta if self.informative else np.zeros(6)
        return (float(value), derivative) if gradient else float(value)


def test_physics_bridge_and_reparameterized_elbo_gradients():
    problem = QuadraticProblem()
    coefficients = torch.full((6,), .8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda c: fem_nll(c, problem), coefficients)
    prior_mu = torch.zeros(6, dtype=torch.float64)
    prior_scale = torch.ones(6, dtype=torch.float64)
    mu = torch.linspace(-.3, .2, 6, dtype=torch.float64, requires_grad=True)
    log_sigma = torch.full((6,), -.7, dtype=torch.float64, requires_grad=True)
    epsilon = torch.linspace(-.4, .4, 6, dtype=torch.float64)[None]
    assert torch.autograd.gradcheck(
        lambda a, b: negative_elbo(problem, a, b, prior_mu, prior_scale, epsilon)[0],
        (mu, log_sigma))


def test_common_transform_cancels_kl_jacobian():
    mu = torch.linspace(-.3, .3, 6, dtype=torch.float64)
    log_sigma = torch.full((6,), -.5, dtype=torch.float64)
    prior_mu = torch.zeros(6, dtype=torch.float64)
    prior_scale = torch.ones(6, dtype=torch.float64)
    q = torch.distributions.Normal(mu, log_sigma.exp())
    p = torch.distributions.Normal(prior_mu, prior_scale)
    z = mu + log_sigma.exp() * .2
    log_jacobian = np.log(UPPER) + torch.nn.functional.logsigmoid(z) + torch.nn.functional.logsigmoid(-z)
    torch.testing.assert_close((q.log_prob(z) - log_jacobian) - (p.log_prob(z) - log_jacobian),
                               q.log_prob(z) - p.log_prob(z))
    torch.testing.assert_close(gaussian_kl(mu, log_sigma, prior_mu, prior_scale),
                               torch.distributions.kl_divergence(q, p).sum())


def test_uninformative_likelihood_preserves_prior_and_sample_support():
    result = infer_vi(QuadraticProblem(False), np.full(6, .4), .5, seed=83,
                      steps=4, posterior_samples=200)
    np.testing.assert_allclose(result['q_mu'], result['prior_mu'], atol=1e-14)
    np.testing.assert_allclose(np.exp(result['q_log_sigma']), result['prior_scale'], atol=1e-14)
    np.testing.assert_allclose(result['coefficients'], result['prior_physical_mean'])
    assert np.all((result['coefficient_samples'] > 0) & (result['coefficient_samples'] < UPPER))
    assert all(abs(row['kl']) < 1e-12 for row in result['trace'])
    assert result['likelihood_fem_calls'] == 4
    # A physical posterior mean differs from transforming the latent mean.
    assert np.max(np.abs(result['coefficients'] - .4)) > .01
    np.testing.assert_allclose(bounded_gaussian_mean(np.zeros(6), np.full(6, 2.)), UPPER/2)


def test_actual_fem_to_variational_parameters_gradient():
    from nr2d.physics import Config, SpatialModel, Trace
    from nr2d.inference_physics import WindowProblem
    from nr2d.estimators import Observer
    root = Path(__file__).resolve().parents[1]
    trace = Trace.load(root / 'runs/information/channels_contact_20261026/histories/calibration_4000.npz')
    model = SpatialModel(Config(**trace.config))
    observed = Observer(model).observations(trace, noise=1., seed=993)
    problem = WindowProblem(model, trace, observed, .8, True)
    projected, _ = problem.projection()
    fraction = np.clip(projected/UPPER, 1e-5, 1-1e-5)
    prior_mu = torch.tensor(np.log(fraction/(1-fraction)), dtype=torch.float64)
    prior_scale = torch.full((6,), .4, dtype=torch.float64)
    mu = prior_mu.clone().requires_grad_()
    logs = torch.full((6,), -1.3, dtype=torch.float64, requires_grad=True)
    epsilon = torch.linspace(-.3, .3, 6, dtype=torch.float64)[None]
    loss = negative_elbo(problem, mu, logs, prior_mu, prior_scale, epsilon)[0]
    loss.backward()
    analytic = np.r_[mu.grad.numpy(), logs.grad.numpy()]
    parameters = np.r_[mu.detach().numpy(), logs.detach().numpy()]
    finite = []
    h = 1e-3
    for i in range(12):
        step = np.zeros(12); step[i] = h
        values = []
        for sign in (-1, 1):
            shifted = parameters + sign * step
            a = torch.tensor(shifted[:6], dtype=torch.float64)
            b = torch.tensor(shifted[6:], dtype=torch.float64)
            values.append(float(negative_elbo(problem, a, b, prior_mu, prior_scale, epsilon)[0]))
        finite.append((values[1] - values[0])/(2*h))
    relative = np.linalg.norm(analytic-finite)/max(np.linalg.norm(finite), 1e-12)
    assert np.linalg.norm(analytic[:6]) > 1e-3
    assert relative < 2e-3, relative
