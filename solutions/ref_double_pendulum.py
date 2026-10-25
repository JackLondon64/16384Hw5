"""Reference double pendulum: the answer key for HW5's code question.

Point masses M1, M2 on massless rods L1, L2, both angles absolute and measured
from the downward vertical, no applied torque.

Everything here is a pure function of an explicit parameter dict, so nothing
depends on (or changes) module state. `ref_submission/double_pendulum.py` is
the same physics written the way the handout asks, with module constants.
"""

import numpy as np
from scipy.integrate import solve_ivp

PARAM_NAMES = ("M1", "M2", "L1", "L2", "G")
DEFAULTS = {"M1": 1.0, "M2": 1.0, "L1": 1.0, "L2": 1.0, "G": 9.81}

# Tighter than the handout's 1e-9, so the reference is the more accurate of
# the two by a wide margin.
RTOL = 1e-11
ATOL = 1e-11


def total_energy(p, q, qd):
    """T + V at each row of (N, 2) arrays q and qd; V is zero at the pivot height."""
    q, qd = np.asarray(q, dtype=float), np.asarray(qd, dtype=float)
    th1, th2 = q[:, 0], q[:, 1]
    th1d, th2d = qd[:, 0], qd[:, 1]
    m1, m2, l1, l2, g = (p[n] for n in PARAM_NAMES)
    T = (0.5 * (m1 + m2) * l1 ** 2 * th1d ** 2
         + 0.5 * m2 * l2 ** 2 * th2d ** 2
         + m2 * l1 * l2 * th1d * th2d * np.cos(th1 - th2))
    V = -(m1 + m2) * g * l1 * np.cos(th1) - m2 * g * l2 * np.cos(th2)
    return T + V


def dynamics(p, q, qd):
    """qdd from M(q) qdd = b(q, qd), the Euler-Lagrange equations."""
    th1, th2 = q
    th1d, th2d = qd
    m1, m2, l1, l2, g = (p[n] for n in PARAM_NAMES)
    c, s = np.cos(th1 - th2), np.sin(th1 - th2)
    M = np.array([[(m1 + m2) * l1 ** 2, m2 * l1 * l2 * c],
                  [m2 * l1 * l2 * c, m2 * l2 ** 2]])
    b = np.array([-m2 * l1 * l2 * s * th2d ** 2 - (m1 + m2) * g * l1 * np.sin(th1),
                  m2 * l1 * l2 * s * th1d ** 2 - m2 * g * l2 * np.sin(th2)])
    return np.linalg.solve(M, b)


def sample_times(T, dt):
    return np.arange(0, T + dt / 2, dt)


def simulate(p, q0, qd0, T, dt):
    """(t, q, qd) sampled every dt from 0 to T inclusive."""
    t = sample_times(T, dt)

    def f(_, x):
        return np.concatenate([x[2:], dynamics(p, x[:2], x[2:])])

    sol = solve_ivp(f, (0.0, t[-1]), np.concatenate([q0, qd0]), method="DOP853",
                    t_eval=t, rtol=RTOL, atol=ATOL)
    return t, sol.y[:2].T, sol.y[2:].T


def euler(p, q0, qd0, T, dt, semi_implicit=False):
    """Fixed-step Euler at the sample spacing: the simplest integrator the
    handout invites. The generator keeps only simulate cases this passes."""
    t = sample_times(T, dt)
    q = np.zeros((len(t), 2))
    qd = np.zeros((len(t), 2))
    q[0], qd[0] = q0, qd0
    for k in range(len(t) - 1):
        qdd = dynamics(p, q[k], qd[k])
        qd[k + 1] = qd[k] + dt * qdd
        q[k + 1] = q[k] + dt * (qd[k + 1] if semi_implicit else qd[k])
    return t, q, qd
