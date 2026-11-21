"""Tests for trajectory's coefficient solver and derivative samplers.

The reference is an independent 6x6 boundary-condition solve written here, whose
coefficients are differentiated exactly, so no finite differences are involved.

Runs standalone (``python tests/test_trajectory.py``) or under pytest.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trajectory import (
    differentiate as trajectory_differentiate,
    evaluate as evaluate_coefficients,
    sample_acceleration,
    sample_jerk,
    sample_trajectory,
    sample_velocity,
    trajectory_coefficients,
)

Q_START = [0.0, 0.5, 2.0]
Q_GOAL = [1.0, -1.5, 2.5]
DURATION = 3.0
TIMES = np.linspace(0.0, DURATION, 13)


def reference_coefficients(q0, qf):
    """Ascending coefficients from an independent 6x6 solve (rest-to-rest)."""
    t0, tf = 0.0, DURATION
    matrix = np.array([
        [1, t0, t0**2, t0**3, t0**4, t0**5],
        [0, 1, 2 * t0, 3 * t0**2, 4 * t0**3, 5 * t0**4],
        [0, 0, 2, 6 * t0, 12 * t0**2, 20 * t0**3],
        [1, tf, tf**2, tf**3, tf**4, tf**5],
        [0, 1, 2 * tf, 3 * tf**2, 4 * tf**3, 5 * tf**4],
        [0, 0, 2, 6 * tf, 12 * tf**2, 20 * tf**3],
    ], dtype=float)
    return np.linalg.solve(matrix, [q0, 0.0, 0.0, qf, 0.0, 0.0])


def differentiate(coeffs, order):
    """Exact d^order/dt^order of ascending polynomial coefficients."""
    derived = np.asarray(coeffs, dtype=float)
    for _ in range(order):
        derived = derived[1:] * np.arange(1, len(derived))
    return derived


def evaluate(coeffs, t):
    return np.polyval(np.asarray(coeffs, dtype=float)[::-1], t)


def expected(order, joint, t):
    coeffs = differentiate(reference_coefficients(Q_START[joint], Q_GOAL[joint]), order)
    return evaluate(coeffs, t)


def test_position_matches_quintic_solver():
    for joint in range(len(Q_START)):
        actual = [sample_trajectory(Q_START, Q_GOAL, DURATION, t)[joint] for t in TIMES]
        reference = [expected(0, joint, t) for t in TIMES]
        np.testing.assert_allclose(actual, reference, atol=1e-9,
                                   err_msg=f'position wrong for joint {joint}')


def test_velocity_matches_exact_derivative():
    for joint in range(len(Q_START)):
        actual = [sample_velocity(Q_START, Q_GOAL, DURATION, t)[joint] for t in TIMES]
        reference = [expected(1, joint, t) for t in TIMES]
        np.testing.assert_allclose(actual, reference, atol=1e-9,
                                   err_msg=f'velocity wrong for joint {joint}')


def test_acceleration_matches_exact_derivative():
    for joint in range(len(Q_START)):
        actual = [sample_acceleration(Q_START, Q_GOAL, DURATION, t)[joint] for t in TIMES]
        reference = [expected(2, joint, t) for t in TIMES]
        np.testing.assert_allclose(actual, reference, atol=1e-9,
                                   err_msg=f'acceleration wrong for joint {joint}')


def test_jerk_matches_exact_derivative():
    for joint in range(len(Q_START)):
        actual = [sample_jerk(Q_START, Q_GOAL, DURATION, t)[joint] for t in TIMES]
        reference = [expected(3, joint, t) for t in TIMES]
        np.testing.assert_allclose(actual, reference, atol=1e-9,
                                   err_msg=f'jerk wrong for joint {joint}')


def test_velocity_and_acceleration_vanish_at_ends():
    for t in (0.0, DURATION):
        np.testing.assert_allclose(sample_velocity(Q_START, Q_GOAL, DURATION, t),
                                   np.zeros(len(Q_START)), atol=1e-12)
        np.testing.assert_allclose(sample_acceleration(Q_START, Q_GOAL, DURATION, t),
                                   np.zeros(len(Q_START)), atol=1e-12)


def test_jerk_is_sixty_delta_over_duration_cubed_at_ends():
    ends = [60.0 * (goal - start) / DURATION**3
            for start, goal in zip(Q_START, Q_GOAL)]
    for t in (0.0, DURATION):
        np.testing.assert_allclose(sample_jerk(Q_START, Q_GOAL, DURATION, t), ends,
                                   atol=1e-9)


def test_jerk_is_minus_thirty_delta_over_duration_cubed_at_midpoint():
    midpoint = [-30.0 * (goal - start) / DURATION**3
                for start, goal in zip(Q_START, Q_GOAL)]
    np.testing.assert_allclose(
        sample_jerk(Q_START, Q_GOAL, DURATION, DURATION / 2.0), midpoint, atol=1e-9)


def test_velocity_and_acceleration_are_zero_outside_the_move():
    for t in (-1.0, DURATION + 1.0):
        np.testing.assert_allclose(sample_velocity(Q_START, Q_GOAL, DURATION, t),
                                   np.zeros(len(Q_START)), atol=1e-12)
        np.testing.assert_allclose(sample_acceleration(Q_START, Q_GOAL, DURATION, t),
                                   np.zeros(len(Q_START)), atol=1e-12)


def test_zero_duration_samplers_are_degenerate_but_safe():
    np.testing.assert_allclose(sample_trajectory(Q_START, Q_GOAL, 0.0, 1.0), Q_GOAL)
    for sampler in (sample_velocity, sample_acceleration, sample_jerk):
        np.testing.assert_allclose(sampler(Q_START, Q_GOAL, 0.0, 1.0),
                                   np.zeros(len(Q_START)), atol=1e-12)


def test_solved_quintic_matches_closed_form_sampler():
    """The general solver and the exact smoothstep fast path agree."""
    for joint in range(len(Q_START)):
        coeffs = reference_coefficients(Q_START[joint], Q_GOAL[joint])
        for t in TIMES:
            np.testing.assert_allclose(
                sample_trajectory(Q_START, Q_GOAL, DURATION, t)[joint],
                evaluate(coeffs, t), atol=1e-9)


def test_quintic_coefficients_honour_all_boundary_conditions():
    t0, tf, q0, qf, w0, wf, a0, af = 1.0, 4.0, 2.0, 5.0, 0.3, -0.7, 0.2, -0.4
    coeffs = trajectory_coefficients(t0, tf, q0, qf, w0, wf, a0, af, 'quintic')
    assert len(coeffs) == 6
    for order, start_value, end_value in ((0, q0, qf), (1, w0, wf), (2, a0, af)):
        derived = differentiate(coeffs, order)
        np.testing.assert_allclose(evaluate(derived, t0), start_value, atol=1e-9,
                                   err_msg=f'derivative {order} at t0')
        np.testing.assert_allclose(evaluate(derived, tf), end_value, atol=1e-9,
                                   err_msg=f'derivative {order} at tf')


def test_cubic_coefficients_honour_position_and_velocity():
    t0, tf, q0, qf, w0, wf = 1.0, 4.0, 2.0, 5.0, 0.3, -0.7
    coeffs = trajectory_coefficients(t0, tf, q0, qf, w0, wf,
                                     trajectory_equation_type='cubic')
    assert len(coeffs) == 4
    for order, start_value, end_value in ((0, q0, qf), (1, w0, wf)):
        derived = differentiate(coeffs, order)
        np.testing.assert_allclose(evaluate(derived, t0), start_value, atol=1e-9)
        np.testing.assert_allclose(evaluate(derived, tf), end_value, atol=1e-9)


def test_cubic_samplers_match_the_cubic_solver():
    coeffs = trajectory_coefficients(0.0, DURATION, Q_START[0], Q_GOAL[0],
                                     trajectory_equation_type='cubic')
    for t in TIMES:
        for order, sampler in enumerate(
                (sample_trajectory, sample_velocity, sample_acceleration, sample_jerk)):
            np.testing.assert_allclose(
                sampler(Q_START, Q_GOAL, DURATION, t, 'cubic')[0],
                evaluate(differentiate(coeffs, order), t),
                atol=1e-9,
                err_msg=f'cubic sampler {sampler.__name__} at t={t}')


def test_cubic_leaves_endpoint_acceleration_that_quintic_removes():
    q0, qf, duration = 0.0, 1.0, 2.0
    np.testing.assert_allclose(
        sample_acceleration([q0], [qf], duration, 0.0, 'quintic'), [0.0], atol=1e-12)
    np.testing.assert_allclose(
        sample_acceleration([q0], [qf], duration, 0.0, 'cubic'),
        [6.0 * (qf - q0) / duration**2], atol=1e-12)


def test_invalid_equation_type_is_rejected():
    for bad in ('quartic', '', None, 5):
        try:
            trajectory_coefficients(0.0, 1.0, 0.0, 1.0, trajectory_equation_type=bad)
        except ValueError:
            continue
        raise AssertionError(f'{bad!r} should have raised ValueError')


def test_cubic_rejects_acceleration_boundary_conditions():
    try:
        trajectory_coefficients(0.0, 1.0, 0.0, 1.0, a0=1.0,
                                trajectory_equation_type='cubic')
    except ValueError:
        return
    raise AssertionError('cubic with non-zero a0 should raise ValueError')


def test_non_increasing_time_span_is_rejected():
    for t0, tf in ((1.0, 1.0), (2.0, 1.0)):
        try:
            trajectory_coefficients(t0, tf, 0.0, 1.0)
        except ValueError:
            continue
        raise AssertionError(f'tf={tf}, t0={t0} should raise ValueError')


def test_evaluate_and_differentiate_round_trip():
    """The module's own evaluate/differentiate helpers agree with the closed form."""
    coeffs = trajectory_coefficients(0.0, 1.0, 0.0, 1.0)
    np.testing.assert_allclose(evaluate_coefficients(coeffs, 0.0), 0.0, atol=1e-12)
    np.testing.assert_allclose(evaluate_coefficients(coeffs, 0.5), 0.5, atol=1e-12)
    np.testing.assert_allclose(evaluate_coefficients(coeffs, 1.0), 1.0, atol=1e-12)

    velocity_coeffs = trajectory_differentiate(coeffs, 1)
    np.testing.assert_allclose(evaluate_coefficients(velocity_coeffs, 0.0), 0.0, atol=1e-9)
    np.testing.assert_allclose(evaluate_coefficients(velocity_coeffs, 0.5), 1.875, atol=1e-9)
    np.testing.assert_allclose(evaluate_coefficients(velocity_coeffs, 1.0), 0.0, atol=1e-9)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")

    print()
    print(f"{len(tests)} tests passed")
