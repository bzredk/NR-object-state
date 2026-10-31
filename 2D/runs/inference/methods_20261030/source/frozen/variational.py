"""RME-inspired VI with the unchanged FEM likelihood and bounded state support.

Prior and variational distributions are diagonal Gaussians in logit coordinates.
Both use c = 2.5 sigmoid(z), so their density Jacobians cancel in the KL. SciPy
is outside Torch; its exact implicit FEM derivatives supply the custom backward.
"""
import time

import numpy as np
from scipy.special import expit, logit
import torch


UPPER = 2.5


class _FEMNLL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coefficients, problem):
        value, derivative = problem.nll(
            coefficients.detach().cpu().numpy(), gradient=True)
        derivative = np.asarray(derivative, dtype=np.float64)
        if not np.isfinite(value) or not np.isfinite(derivative).all():
            raise FloatingPointError('Nonfinite FEM likelihood or derivative')
        ctx.save_for_backward(torch.as_tensor(
            derivative, dtype=coefficients.dtype, device=coefficients.device))
        return coefficients.new_tensor(value)

    @staticmethod
    def backward(ctx, output_gradient):
        derivative, = ctx.saved_tensors
        return output_gradient * derivative, None


def fem_nll(coefficients, problem):
    """Scalar Gaussian NLL with an explicit, nonzero physical backward path."""
    return _FEMNLL.apply(coefficients, problem)


def gaussian_kl(mu, log_sigma, prior_mu, prior_scale):
    """Exact KL in common unconstrained coordinates; no sample clipping."""
    return .5 * torch.sum(
        (torch.exp(2 * log_sigma) + (mu - prior_mu) ** 2) / prior_scale ** 2
        - 1 + 2 * (torch.log(prior_scale) - log_sigma))


def negative_elbo(problem, mu, log_sigma, prior_mu, prior_scale, epsilon):
    """Reparameterized sample-mean NLL plus exact KL, without likelihood tempering.

    epsilon is supplied explicitly so deterministic finite differences can check
    the complete variational-parameter to physical-likelihood gradient path.
    """
    if epsilon.ndim == 1:
        epsilon = epsilon[None]
    latent = mu[None] + torch.exp(log_sigma)[None] * epsilon
    coefficients = UPPER * torch.sigmoid(latent)
    nll = torch.stack([fem_nll(c, problem) for c in coefficients]).mean()
    kl = gaussian_kl(mu, log_sigma, prior_mu, prior_scale)
    return nll + kl, nll, kl


def bounded_gaussian_mean(mu, sigma, quadrature_points=64):
    """Marginal physical coefficient mean, not sigmoid of the latent mean."""
    nodes, weights = np.polynomial.hermite.hermgauss(quadrature_points)
    values = expit(np.asarray(mu)[..., None]
                   + np.sqrt(2) * np.asarray(sigma)[..., None] * nodes)
    return UPPER * (values @ weights) / np.sqrt(np.pi)


def infer_vi(problem, prior_coefficients, prior_scale, seed, steps=12,
             samples_per_step=1, lr=.05, posterior_samples=16,
             gradient_clip=10.):
    """Fit a mean-field logistic-normal distribution over window-start state.

    The caller records whether the supplied prior is neural/data-conditioned or
    fixed from training. q begins equal to that prior; steps is a calibrated fixed
    budget, not a convergence claim. Runtime includes VI and coefficient sampling,
    but history-to-current replay and future FEM predictions are timed by caller.
    """
    started = time.perf_counter()
    center = np.asarray(prior_coefficients, dtype=np.float64).reshape(6)
    scale = np.broadcast_to(np.asarray(prior_scale, dtype=np.float64), (6,)).copy()
    if not np.isfinite(center).all() or np.any((center < 0) | (center > UPPER)):
        raise ValueError('Prior coefficient center must lie in [0, 2.5]')
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Prior latent standard deviations must be finite and positive')
    if steps < 0 or samples_per_step < 1 or posterior_samples < 1:
        raise ValueError('Invalid VI or Monte Carlo budget')

    # This endpoint convention defines a finite prior; random samples are never
    # clipped. Interior neural outputs do not require this numerical adjustment.
    fraction = np.clip(center / UPPER, 1e-6, 1 - 1e-6)
    prior_mu = torch.tensor(logit(fraction), dtype=torch.float64)
    prior_std = torch.tensor(scale, dtype=torch.float64)
    mu = torch.nn.Parameter(prior_mu.clone())
    log_sigma = torch.nn.Parameter(torch.log(prior_std).clone())
    optimizer = torch.optim.Adam([mu, log_sigma], lr=lr)
    vi_seed = int(seed) % (2**63 - 1)
    prediction_seed = (vi_seed + 1_000_003) % (2**63 - 1)
    generator = torch.Generator(device='cpu').manual_seed(vi_seed)
    initial_calls = getattr(problem, 'calls', 0)
    trace = []
    for index in range(steps):
        epsilon = torch.randn((samples_per_step, 6), generator=generator,
                              dtype=torch.float64)
        optimizer.zero_grad(set_to_none=True)
        loss, nll, kl = negative_elbo(
            problem, mu, log_sigma, prior_mu, prior_std, epsilon)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite variational objective')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([mu, log_sigma], gradient_clip)
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite variational gradient')
        before = torch.cat((mu.detach().clone(), log_sigma.detach().clone()))
        optimizer.step()
        change = torch.linalg.vector_norm(
            torch.cat((mu.detach(), log_sigma.detach())) - before)
        trace.append({'step': index + 1, 'nll': float(nll.detach()),
                      'kl': float(kl.detach()), 'loss': float(loss.detach()),
                      'gradient_norm_before_clipping': float(norm),
                      'parameter_change': float(change)})

    posterior_generator = torch.Generator(device='cpu').manual_seed(prediction_seed)
    with torch.no_grad():
        epsilon = torch.randn((posterior_samples, 6), generator=posterior_generator,
                              dtype=torch.float64)
        samples = UPPER * torch.sigmoid(mu[None] + log_sigma.exp()[None] * epsilon)
    q_mu = mu.detach().numpy().copy()
    q_log_sigma = log_sigma.detach().numpy().copy()
    physical_mean = bounded_gaussian_mean(q_mu, np.exp(q_log_sigma))
    return {'coefficients': physical_mean,
            'coefficient_samples': samples.numpy().copy(),
            'q_mu': q_mu, 'q_log_sigma': q_log_sigma,
            'prior_mu': prior_mu.numpy().copy(), 'prior_scale': scale,
            'prior_coefficients_center': center,
            'prior_physical_mean': bounded_gaussian_mean(prior_mu.numpy(), scale),
            'trace': trace, 'seconds': time.perf_counter() - started,
            'optimizer_status': 'fixed_budget_completed', 'optimizer_steps': steps,
            'likelihood_fem_calls': getattr(problem, 'calls', 0) - initial_calls,
            'seed': vi_seed, 'posterior_sampling_seed': prediction_seed,
            'samples_per_step': samples_per_step, 'posterior_samples': posterior_samples,
            'learning_rate': lr, 'gradient_clip': gradient_clip,
            'initial_variational_distribution': 'same as prior',
            'family': 'diagonal Gaussian logits transformed by 2.5 sigmoid',
            'point_estimate': 'Gauss-Hermite physical coefficient mean, then FEM propagation',
            'quadrature_points': 64, 'likelihood_reduction': 'sum',
            'trace_evaluation': 'before optimizer update',
            'prior_endpoint_fraction': 1e-6,
            'gradient_path': 'exact FEM implicit sensitivity through custom Torch backward'}
