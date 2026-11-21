"""Tests for quintic_equation.quintic_equation.

Runs standalone (``python tests/test_quintic_equation.py``) or under pytest.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trajectory import format_equation, trajectory_coefficients


def quintic_equation(t0, tf, q0, qf, w0, wf, a0, af):
    """Thin local alias so this file's call sites mirror the old public signature."""
    return trajectory_coefficients(t0, tf, q0, qf, w0, wf, a0, af, 'quintic')

# t0, tf, q0, qf, w0, wf, a0, af
CASES = [
    (0.0, 3.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 2.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0),
    (1.0, 5.0, 10.0, 20.0, 0.0, 0.0, 0.0, 0.0),
    (-3.0, 4.0, 2.5, -7.5, 0.0, 0.0, 0.0, 0.0),
    (0.0, 3.0, -1.0, 4.0, 0.5, -1.5, 2.0, -3.0),
]

H = 1e-3


def velocity(equation, t):
    """5-point central difference (O(h^4)) of the equation at t."""
    return (
        8 * (equation(t + H) - equation(t - H))
        - (equation(t + 2 * H) - equation(t - 2 * H))
    ) / (12 * H)


def acceleration(equation, t):
    """5-point central difference (O(h^4)) of the equation at t."""
    return (
        16 * (equation(t + H) + equation(t - H))
        - (equation(t + 2 * H) + equation(t - 2 * H))
        - 30 * equation(t)
    ) / (12 * H**2)


def reference_coefficients(t0, tf, q0, qf, w0, wf, a0, af):
    """Boundary-condition system solved here, independently of the module under test."""
    vander = np.array(
        [
            [1, t0, t0**2, t0**3, t0**4, t0**5],
            [0, 1, 2 * t0, 3 * t0**2, 4 * t0**3, 5 * t0**4],
            [0, 0, 2, 6 * t0, 12 * t0**2, 20 * t0**3],
            [1, tf, tf**2, tf**3, tf**4, tf**5],
            [0, 1, 2 * tf, 3 * tf**2, 4 * tf**3, 5 * tf**4],
            [0, 0, 2, 6 * tf, 12 * tf**2, 20 * tf**3],
        ]
    )
    return np.linalg.solve(vander, [q0, w0, a0, qf, wf, af])


def as_equation(coeffs):
    """Wrap ascending coefficients in a callable q(t), built independently of the module."""
    c = np.asarray(coeffs, dtype=float)

    def q(t):
        return sum(ci * t**i for i, ci in enumerate(c))

    return q


def test_returns_six_float_coefficients():
    coeffs = quintic_equation(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    assert isinstance(coeffs, list)
    assert len(coeffs) == 6
    assert all(isinstance(c, float) for c in coeffs)


def test_endpoints_match_boundary_positions():
    for t0, tf, q0, qf, w0, wf, a0, af in CASES:
        q = as_equation(quintic_equation(t0, tf, q0, qf, w0, wf, a0, af))
        np.testing.assert_allclose(
            [q(t0), q(tf)],
            [q0, qf],
            atol=1e-9,
            err_msg=f"positions wrong for t0={t0}, tf={tf}",
        )


def test_velocities_match_boundary_conditions():
    for t0, tf, q0, qf, w0, wf, a0, af in CASES:
        q = as_equation(quintic_equation(t0, tf, q0, qf, w0, wf, a0, af))
        np.testing.assert_allclose(
            [velocity(q, t0), velocity(q, tf)],
            [w0, wf],
            atol=1e-6,
            err_msg=f"velocities wrong for t0={t0}, tf={tf}",
        )


def test_accelerations_match_boundary_conditions():
    for t0, tf, q0, qf, w0, wf, a0, af in CASES:
        q = as_equation(quintic_equation(t0, tf, q0, qf, w0, wf, a0, af))
        np.testing.assert_allclose(
            [acceleration(q, t0), acceleration(q, tf)],
            [a0, af],
            atol=1e-5,
            err_msg=f"accelerations wrong for t0={t0}, tf={tf}",
        )


def test_known_smoothstep_coefficients():
    # Zero velocity/acceleration at both ends over t0=0, tf=1, q0=0, qf=1.
    coeffs = quintic_equation(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    np.testing.assert_allclose(coeffs, [0.0, 0.0, 0.0, 10.0, -15.0, 6.0], atol=1e-12)


def test_matches_reference_coefficients():
    for t0, tf, q0, qf, w0, wf, a0, af in CASES:
        coeffs = quintic_equation(t0, tf, q0, qf, w0, wf, a0, af)
        expected = reference_coefficients(t0, tf, q0, qf, w0, wf, a0, af)
        np.testing.assert_allclose(
            coeffs,
            expected,
            atol=1e-9,
            err_msg=f"coefficients differ for t0={t0}, tf={tf}",
        )


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")

    print()
    for t0, tf, q0, qf, w0, wf, a0, af in CASES:
        coeffs = reference_coefficients(t0, tf, q0, qf, w0, wf, a0, af)
        print(f"t0={t0}, tf={tf}, q0={q0}, qf={qf}, w0={w0}, wf={wf}, a0={a0}, af={af}")
        print(format_equation(coeffs) + "\n")

    print(f"{len(tests)} tests passed")
