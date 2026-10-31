"""Shared window-start state, projection and exact-FEM observation sensitivities."""
from dataclasses import replace
import time

import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear
from scipy.sparse.linalg import splu

from .physics import State, Trace
from .estimators import Observer, NOISE_STD
from .information import contact_end, coverage


FEATURE_SCALE = np.array([.0005, .005, .25, .00005, .00005, .0005, 1.4])


class WindowProblem:
    """All estimators infer the SAME coefficients at begin, then replay to cut.

    The likelihood conditions on measured tool motion. Its Gaussian observation
    error includes force/side sensor noise, but not errors-in-variables treatment
    of the noisy motion input. No surrogate replaces the existing SpatialModel.
    """

    def __init__(self, model, trace, observed, window, lateral, basis=None):
        self.model, self.trace = model, trace
        self.observed = np.asarray(observed).copy()
        self.lateral = bool(lateral)
        if not lateral:
            self.observed[:, 3:] = 0.
        self.last = len(trace.time) - 1
        duration = .2 if str(window) == 'contact' else float(window)
        self.end = (contact_end(self.observed, model.c.dt, .2, .0045)
                    if str(window) == 'contact' else self.last)
        self.begin = self.end - min(self.end, round(duration / model.c.dt))
        self.window = str(window)
        self.channels = [2, 3, 4] if lateral else [2]
        self.sigma = NOISE_STD[self.channels]
        self.basis = Observer(model).basis if basis is None else basis
        self.calls = 0
        self.sensitivity_calls = 0
        self._basis_flat = self.basis.reshape(3, -1).T

    def inputs(self):
        obs = self.observed[self.begin:self.end + 1].copy()
        # Closed channels are zero before any scaling, not merely after encoding.
        mask = np.array([1., 1., 1., float(self.lateral), float(self.lateral)])
        obs *= mask
        raw = np.column_stack((obs, self.trace.command[self.begin:self.end + 1],
                               self.trace.time[self.begin:self.end + 1]))
        raw = raw / FEATURE_SCALE
        values = np.column_stack((raw, np.broadcast_to(mask, (len(raw), 5))))
        padded = np.zeros((81, 12), dtype=np.float32)
        padded[:len(values)] = values
        return padded, len(values)

    def internal(self, coefficients):
        return np.einsum('ij,jab->iab', np.asarray(coefficients).reshape(2, 3), self.basis)

    def initial(self, coefficients):
        z = self.internal(coefficients)
        d = float(self.observed[self.begin, 0])
        u = self.model.equilibrate(z, d)
        return State(u, z, d, float(self.observed[self.begin, 1]),
                     float(self.trace.time[self.begin]))

    def replay(self, coefficients, to_current=False):
        self.calls += 1
        stop = self.last if to_current else self.end
        return self.model.simulate(
            self.trace.command[self.begin + 1:stop + 1],
            self.trace.phase[self.begin + 1:stop + 1], state=self.initial(coefficients),
            actual_positions=self.observed[self.begin + 1:stop + 1, 0])

    def current(self, coefficients):
        replay = self.replay(coefficients, to_current=True)
        state = replay.checkpoint()
        state.d, state.v = map(float, self.observed[-1, :2])
        return state

    def projection(self):
        """Bounded least squares in element-volume, elastic-energy norm.

        The minimum field error is a privileged diagnostic, not a prediction-error
        lower bound. A rank-deficient basis need not identify unique coefficients.
        """
        m = self.model
        transform = np.linalg.cholesky(m.D)
        weight = np.sqrt(m.area * m.c.thickness)[:, None]
        a = np.stack([(b @ transform * weight).ravel() for b in self.basis], axis=1)
        target = self.trace.z[self.begin]
        coefficients = []
        for field in target:
            result = lsq_linear(a, (field @ transform * weight).ravel(),
                                bounds=(0., 2.5), tol=1e-12, max_iter=100)
            coefficients.extend(result.x)
        coefficients = np.array(coefficients)
        residual = self.internal(coefficients) - target
        numerator = sum(np.linalg.norm(f @ transform * weight) ** 2 for f in residual)
        denominator = sum(np.linalg.norm(f @ transform * weight) ** 2 for f in target)
        singular = np.linalg.svd(a, compute_uv=False)
        return coefficients, {'relative_field_error': float(np.sqrt(numerator / max(denominator, 1e-30))),
                              'basis_singular_values': singular.tolist(),
                              'basis_rank': int(np.linalg.matrix_rank(a)),
                              'coefficient_definition': 'bounded volume/elastic-energy projection at window start'}

    def _memory_load(self, dz, branch_weight):
        m = self.model
        stress = np.einsum('i,ab,kieb->kea', branch_weight, m.D, dz)
        local = np.einsum('eai,kea,e->kei', m.B, stress, m.area * m.c.thickness)
        out = np.zeros((m.nd, 6))
        for k in range(6):
            np.add.at(out[:, k], m.dofs.ravel(), local[k].ravel())
        return out

    def _contact_tangent(self, u, d):
        """Derivative of the actual residual, including circular-gap curvature.

        The forward Newton solver uses a Gauss-Newton contact tangent. Its fixed
        point is unchanged here; implicit sensitivities require the missing
        pressure-times-gap-Hessian term as well.
        """
        m = self.model
        _, tangent, _, gap, points, pressure = m.contact(u, d)
        root = np.sqrt(m.c.radius ** 2 - points[:, 0] ** 2)
        coefficient = -pressure * m.qweight * m.c.radius ** 2 / root ** 3
        dofs = 2 * m.edge_nodes
        rows = np.repeat(dofs, 2, axis=1).ravel()
        cols = np.tile(dofs, (1, 2)).ravel()
        vals = (coefficient[:, None, None] * m.N[:, :, None] * m.N[:, None, :]).ravel()
        curvature = sparse.coo_matrix((vals, (rows, cols)), shape=(m.nd, m.nd)).tocsc()
        return tangent[:-1, :-1].tocsc() + curvature, gap, points

    def observation_and_jacobian(self, coefficients):
        self.sensitivity_calls += 1
        m = self.model
        replay = self.replay(coefficients)
        dz = np.zeros((6, 2, len(m.tri), 3))
        for branch in range(2):
            for component in range(3):
                dz[branch * 3 + component, branch] = self.basis[component]
        values, jacobians = [], []
        a = m.tau / (m.tau + m.c.dt)
        for index, (u, d) in enumerate(zip(replay.u, replay.d)):
            weights = m.Ebranch if index == 0 else a * m.Ebranch
            modulus = m.c.E_inf + weights.sum()
            contact_tangent, gap, points = self._contact_tangent(u, d)
            rhs = self._memory_load(dz, weights)
            tangent = (modulus * m.Kunit + contact_tangent)[m.free, :][:, m.free].tocsc()
            du = np.zeros((m.nd, 6))
            du[m.free] = splu(tangent).solve(rhs[m.free])
            # Sum of normal contact forces; d is a measured prescribed input.
            x = points[:, 0]
            normal = np.column_stack((-x / np.sqrt(m.c.radius ** 2 - x*x), np.ones_like(x)))
            qp_du = np.einsum('qi,qijk->qjk', m.N, du.reshape(-1, 2, 6)[m.edge_nodes])
            dgap = np.einsum('qi,qik->qk', normal, qp_du)
            dforce = ((gap > 0) * m.c.penalty * m.qweight) @ dgap
            dmarkers = du[2 * m.marker_nodes] * m.marker_sign[:, None]
            full_jac = np.vstack((dforce, dmarkers))
            values.append(replay.observation[index, self.channels])
            jacobians.append(full_jac if self.lateral else full_jac[:1])
            if index > 0:
                de = np.einsum('eij,ejk->kei', m.B, du[m.dofs])
                dz = a[None, :, None, None] * dz + (1-a)[None, :, None, None] * de[:, None]
        return np.asarray(values), np.asarray(jacobians)

    def residual(self, coefficients, jacobian=False, normalize=False):
        if jacobian:
            values, jac = self.observation_and_jacobian(coefficients)
        else:
            values = self.replay(coefficients).observation[:, self.channels]
        residual = (values - self.observed[self.begin:self.end + 1, self.channels]) / self.sigma
        divisor = np.sqrt(len(residual)) if normalize else 1.
        if jacobian:
            return residual.ravel() / divisor, (jac / self.sigma[None, :, None]).reshape(-1, 6) / divisor
        return residual.ravel() / divisor

    def nll(self, coefficients, gradient=False):
        """SUM Gaussian negative log likelihood, with all normalizing constants."""
        if gradient:
            residual, jac = self.residual(coefficients, jacobian=True)
        else:
            residual = self.residual(coefficients)
        count = self.end - self.begin + 1
        value = .5 * residual @ residual + count * np.log(self.sigma * np.sqrt(2*np.pi)).sum()
        return (float(value), jac.T @ residual) if gradient else float(value)

    def record(self):
        t = self.trace.time
        record = {'window_start': float(t[self.begin]), 'update_end': float(t[self.end]),
                  'prediction_cut': float(t[-1]), 'update_samples': self.end-self.begin+1,
                  'propagation_seconds': float(t[-1]-t[self.end]),
                  'total_motion_span': float(t[-1]-t[self.begin]),
                  'channels': self.channels, 'motion_samples': self.last-self.begin+1,
                  'fixed_physical_scales': FEATURE_SCALE.tolist(),
                  **coverage(self.trace, self.observed, self.begin, self.end, .0045)}
        if self.window == 'contact':
            record['selection'] = {'channel':'observed force','threshold_N':.0045,
                                   'scanned_interval':[float(t[0]),float(t[-1])]}
        return record
