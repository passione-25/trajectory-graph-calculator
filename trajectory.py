"""Trajectory math: cubic/quintic boundary-condition polynomials and sampling.

Coefficients are returned ascending in ``t``, so ``coeffs[0]`` is the constant
term. ``trajectory_equation_type`` selects the polynomial order:

* ``'cubic'``   - 4 coefficients; matches position and velocity at both ends.
* ``'quintic'`` - 6 coefficients; also matches acceleration at both ends.

For a rest-to-rest move (zero boundary velocity and acceleration) the quintic
collapses to the normalized smoothstep ``s(u) = 10u^3 - 15u^4 + 6u^5`` on
``u = (t - t0) / (tf - t0)``, and the cubic to ``3u^2 - 2u^3``. Those closed
forms are used as exact fast paths; ``tests/test_trajectory.py`` checks them
against the general solver below.
"""

import numpy as np

EQUATION_TYPES = ('cubic', 'quintic')
DEFAULT_EQUATION_TYPE = 'quintic'


def clamp01(x):
    """Clamp a scalar to the closed interval [0, 1]."""
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def quintic_smoothstep(u):
    """Quintic smoothstep: 0 at u=0, 1 at u=1, zero first/second derivatives at both ends."""
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def quintic_smoothstep_derivative(u):
    """d/du of quintic_smoothstep; vanishes at u=0 and u=1."""
    return 30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4


def quintic_smoothstep_second_derivative(u):
    """d2/du2 of quintic_smoothstep; vanishes at u=0 and u=1."""
    return 60.0 * u - 180.0 * u**2 + 120.0 * u**3


def quintic_smoothstep_third_derivative(u):
    """d3/du3 of quintic_smoothstep; +60 at both ends and -30 at u=0.5.

    A quintic cannot have zero jerk at the boundaries, so this is intentionally
    non-zero at u=0 and u=1.
    """
    return 60.0 - 360.0 * u + 360.0 * u**2


def cubic_smoothstep(u):
    """Cubic smoothstep: 0 at u=0, 1 at u=1, zero first derivative at both ends."""
    return 3.0 * u**2 - 2.0 * u**3


def cubic_smoothstep_derivative(u):
    """d/du of cubic_smoothstep; vanishes at u=0 and u=1."""
    return 6.0 * u - 6.0 * u**2


def cubic_smoothstep_second_derivative(u):
    """d2/du2 of cubic_smoothstep; 6 at u=0 and -6 at u=1 (a cubic cannot zero this)."""
    return 6.0 - 12.0 * u


def cubic_smoothstep_third_derivative(u):
    """d3/du3 of cubic_smoothstep; constant, so jerk is a step at u=0 and u=1."""
    return -12.0


_SMOOTHSTEP_DERIVATIVES = {
    'cubic': (
        cubic_smoothstep,
        cubic_smoothstep_derivative,
        cubic_smoothstep_second_derivative,
        cubic_smoothstep_third_derivative,
    ),
    'quintic': (
        quintic_smoothstep,
        quintic_smoothstep_derivative,
        quintic_smoothstep_second_derivative,
        quintic_smoothstep_third_derivative,
    ),
}


def normalize_equation_type(trajectory_equation_type):
    """Validate `trajectory_equation_type` and return the canonical lower-case name."""
    kind = str(trajectory_equation_type).strip().lower()
    if kind not in EQUATION_TYPES:
        raise ValueError(
            f'trajectory_equation_type must be one of {EQUATION_TYPES}, '
            f'got {trajectory_equation_type!r}')
    return kind


def trajectory_coefficients(t0, tf, q0, qf, w0=0.0, wf=0.0, a0=0.0, af=0.0,
                            trajectory_equation_type=DEFAULT_EQUATION_TYPE):
    """Solve the boundary conditions for the polynomial coefficients.

    Parameters
    ----------
    t0, tf : float
        Start and end time; ``tf`` must be greater than ``t0``.
    q0, qf : float
        Position at ``t0`` and ``tf``.
    w0, wf : float
        Velocity at ``t0`` and ``tf``.
    a0, af : float
        Acceleration at ``t0`` and ``tf``. Only meaningful for ``'quintic'``;
        a cubic has too few coefficients, so passing a non-zero value with
        ``'cubic'`` raises ``ValueError``.
    trajectory_equation_type : {'cubic', 'quintic'}
        Polynomial order: 4 or 6 coefficients respectively.

    Returns
    -------
    list of float
        Ascending coefficients: ``c0 + c1*t + c2*t**2 + ...``.
    """
    kind = normalize_equation_type(trajectory_equation_type)
    t0 = float(t0)
    tf = float(tf)
    if not tf > t0:
        raise ValueError(f'tf must be greater than t0, got t0={t0}, tf={tf}')

    if kind == 'cubic':
        if a0 or af:
            raise ValueError(
                "'cubic' cannot match boundary acceleration; use 'quintic'")
        matrix = np.array([
            [1.0, t0, t0**2, t0**3],
            [0.0, 1.0, 2.0 * t0, 3.0 * t0**2],
            [1.0, tf, tf**2, tf**3],
            [0.0, 1.0, 2.0 * tf, 3.0 * tf**2],
        ])
        target = np.array([q0, w0, qf, wf], dtype=float)
    else:
        matrix = np.array([
            [1.0, t0, t0**2, t0**3, t0**4, t0**5],
            [0.0, 1.0, 2.0 * t0, 3.0 * t0**2, 4.0 * t0**3, 5.0 * t0**4],
            [0.0, 0.0, 2.0, 6.0 * t0, 12.0 * t0**2, 20.0 * t0**3],
            [1.0, tf, tf**2, tf**3, tf**4, tf**5],
            [0.0, 1.0, 2.0 * tf, 3.0 * tf**2, 4.0 * tf**3, 5.0 * tf**4],
            [0.0, 0.0, 2.0, 6.0 * tf, 12.0 * tf**2, 20.0 * tf**3],
        ])
        target = np.array([q0, w0, a0, qf, wf, af], dtype=float)

    return np.linalg.solve(matrix, target).tolist()


def differentiate(coeffs, order=1):
    """Return the ascending coefficients of the `order`-th derivative."""
    derived = np.asarray(coeffs, dtype=float)
    for _ in range(order):
        if derived.size <= 1:
            return [0.0]
        derived = derived[1:] * np.arange(1, derived.size)
    return derived.tolist()


def evaluate(coeffs, t):
    """Evaluate ascending coefficients at ``t``."""
    return np.polyval(np.asarray(coeffs, dtype=float)[::-1], t)


def format_equation(coeffs, tol=1e-12, name='q(t)'):
    """Render ascending coefficients as a readable ``q(t) = ...`` string."""
    terms = [f'{coeffs[0]:.6g}']
    for power, coefficient in enumerate(coeffs[1:], start=1):
        if abs(coefficient) < tol:
            continue
        terms.append(f'{coefficient:+.6g} * t**{power}')
    return f'{name} = ' + ' '.join(terms)


def _delta(q_start, q_goal, scale):
    """Per-joint displacement (q_goal - q_start) * scale."""
    q0 = np.asarray(q_start, dtype=float)
    qf = np.asarray(q_goal, dtype=float)
    return (qf - q0) * scale


def _sampler(q_start, q_goal, duration, elapsed, order, trajectory_equation_type):
    """Shared body for the per-derivative samplers."""
    kind = normalize_equation_type(trajectory_equation_type)
    if duration <= 0:
        return [0.0] * len(list(q_start))
    shape = _SMOOTHSTEP_DERIVATIVES[kind][order]
    u = clamp01(float(elapsed) / float(duration))
    return _delta(q_start, q_goal, shape(u) / float(duration) ** order).tolist()


def sample_trajectory(q_start, q_goal, duration, elapsed,
                      trajectory_equation_type=DEFAULT_EQUATION_TYPE):
    """Return the joint positions at `elapsed` seconds, clamped to [0, duration]."""
    kind = normalize_equation_type(trajectory_equation_type)
    if duration <= 0:
        return list(q_goal)
    u = clamp01(float(elapsed) / float(duration))
    shape = _SMOOTHSTEP_DERIVATIVES[kind][0]
    return (np.asarray(q_start, dtype=float)
            + _delta(q_start, q_goal, shape(u))).tolist()


def sample_velocity(q_start, q_goal, duration, elapsed,
                    trajectory_equation_type=DEFAULT_EQUATION_TYPE):
    """Return dq/dt at `elapsed` seconds; zero at and outside the endpoints."""
    return _sampler(q_start, q_goal, duration, elapsed, 1, trajectory_equation_type)


def sample_acceleration(q_start, q_goal, duration, elapsed,
                        trajectory_equation_type=DEFAULT_EQUATION_TYPE):
    """Return d2q/dt2 at `elapsed` seconds.

    Zero at the endpoints for ``'quintic'``; a cubic leaves a non-zero jump there.
    """
    return _sampler(q_start, q_goal, duration, elapsed, 2, trajectory_equation_type)


def sample_jerk(q_start, q_goal, duration, elapsed,
                trajectory_equation_type=DEFAULT_EQUATION_TYPE):
    """Return d3q/dt3 at `elapsed` seconds.

    Non-zero at the endpoints for a quintic by construction, and piecewise
    constant for a cubic.
    """
    return _sampler(q_start, q_goal, duration, elapsed, 3, trajectory_equation_type)
