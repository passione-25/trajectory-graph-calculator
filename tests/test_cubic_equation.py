"""Tests for cubic_equation.cubic_equation.

Runs standalone (``python tests/test_cubic_equation.py``) or under pytest.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trajectory import format_equation, trajectory_coefficients


def cubic_equation(t0, tf, q0, qf, w0, wf):
    """Thin local alias so this file's call sites mirror the old public signature."""
    return trajectory_coefficients(t0, tf, q0, qf, w0, wf, 0.0, 0.0, 'cubic')

# t0, tf, q0, qf, w0, wf
CASES = [
    (0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
    (0.0, 2.0, 1.0, 3.0, 0.0, 0.0),
    (1.0, 5.0, 10.0, 20.0, -1.0, 2.0),
    (-3.0, 4.0, 2.5, -7.5, 3.0, 1.0),
]

H = 1e-3


def velocity(equation, t):
    """5-point central difference (O(h^4)) of the equation at t."""
    return (
        8 * (equation(t + H) - equation(t - H))
        - (equation(t + 2 * H) - equation(t - 2 * H))
    ) / (12 * H)


def reference_coefficients(t0, tf, q0, qf, w0, wf):
    """Boundary-condition system solved here, independently of the module under test."""
    vander = np.array(
        [
            [1, t0, t0**2, t0**3],
            [0, 1, 2 * t0, 3 * t0**2],
            [1, tf, tf**2, tf**3],
            [0, 1, 2 * tf, 3 * tf**2],
        ]
    )
    return np.linalg.solve(vander, [q0, w0, qf, wf])


def as_equation(coeffs):
    """Wrap ascending coefficients in a callable q(t), built independently of the module."""
    c = np.asarray(coeffs, dtype=float)

    def q(t):
        return sum(ci * t**i for i, ci in enumerate(c))

    return q


def test_returns_four_float_coefficients():
    coeffs = cubic_equation(0.0, 1.0, 0.0, 1.0, 0.0, 0.0)
    assert isinstance(coeffs, list)
    assert len(coeffs) == 4
    assert all(isinstance(c, float) for c in coeffs)


def test_endpoints_match_boundary_positions():
    for t0, tf, q0, qf, w0, wf in CASES:
        q = as_equation(cubic_equation(t0, tf, q0, qf, w0, wf))
        np.testing.assert_allclose(
            [q(t0), q(tf)],
            [q0, qf],
            atol=1e-9,
            err_msg=f"positions wrong for t0={t0}, tf={tf}",
        )


def test_velocities_match_boundary_conditions():
    for t0, tf, q0, qf, w0, wf in CASES:
        q = as_equation(cubic_equation(t0, tf, q0, qf, w0, wf))
        np.testing.assert_allclose(
            [velocity(q, t0), velocity(q, tf)],
            [w0, wf],
            atol=1e-6,
            err_msg=f"velocities wrong for t0={t0}, tf={tf}",
        )


def test_known_smoothstep_coefficients():
    # Zero velocity at both ends over t0=0, tf=1, q0=0, qf=1.
    coeffs = cubic_equation(0.0, 1.0, 0.0, 1.0, 0.0, 0.0)
    np.testing.assert_allclose(coeffs, [0.0, 0.0, 3.0, -2.0], atol=1e-12)


def test_matches_reference_coefficients():
    for t0, tf, q0, qf, w0, wf in CASES:
        coeffs = cubic_equation(t0, tf, q0, qf, w0, wf)
        expected = reference_coefficients(t0, tf, q0, qf, w0, wf)
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
    for t0, tf, q0, qf, w0, wf in CASES:
        coeffs = reference_coefficients(t0, tf, q0, qf, w0, wf)
        print(f"t0={t0}, tf={tf}, q0={q0}, qf={qf}, w0={w0}, wf={wf}")
        print(format_equation(coeffs) + "\n")

    print(f"{len(tests)} tests passed")
