"""Tests for bottle.BottleArm.generate_trajectory.

bottle.py is a ROS 2 node, so minimal stand-ins for rclpy / geometry_msgs /
hw_arm_bottle are installed before it is imported. Runs standalone
(``python tests/test_bottle.py``) or under pytest.
"""

import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _install_ros_stubs():
    """Install minimal stand-ins so bottle.py imports without a ROS install."""

    class ExternalShutdownException(Exception):
        pass

    class PoseStamped:
        pass

    class SimpleArm:
        def __init__(self, *args, **kwargs):
            self.q = [0.0, 0.0, 0.0]

    rclpy = types.ModuleType('rclpy')
    rclpy.init = lambda *args, **kwargs: None
    rclpy.spin = lambda *args, **kwargs: None
    rclpy.try_shutdown = lambda *args, **kwargs: None
    executors = types.ModuleType('rclpy.executors')
    executors.ExternalShutdownException = ExternalShutdownException
    rclpy.executors = executors

    geometry_msgs = types.ModuleType('geometry_msgs')
    geometry_msgs_msg = types.ModuleType('geometry_msgs.msg')
    geometry_msgs_msg.PoseStamped = PoseStamped
    geometry_msgs.msg = geometry_msgs_msg

    hw_arm_bottle = types.ModuleType('hw_arm_bottle')
    hw_arm = types.ModuleType('hw_arm_bottle.arm')
    hw_arm.SimpleArm = SimpleArm
    hw_scene = types.ModuleType('hw_arm_bottle.scene')
    hw_scene.BOTTLE_HEIGHT = 0.20
    hw_scene.LID_CLEARANCE = 0.05
    hw_scene.TRAJECTORY_DURATION = 3.0
    hw_arm_bottle.arm = hw_arm
    hw_arm_bottle.scene = hw_scene

    sys.modules.update({
        'rclpy': rclpy,
        'rclpy.executors': executors,
        'geometry_msgs': geometry_msgs,
        'geometry_msgs.msg': geometry_msgs_msg,
        'hw_arm_bottle': hw_arm_bottle,
        'hw_arm_bottle.arm': hw_arm,
        'hw_arm_bottle.scene': hw_scene,
    })


_install_ros_stubs()

from bottle import BottleArm
from trajectory import evaluate, trajectory_coefficients

Q_START = [0.0, 0.0, 0.0]
Q_GOAL = [1.0, 2.0, -3.0]
DURATION = 3.0
# Finite-difference steps chosen near each derivative's conditioning optimum:
# smaller h loses precision to cancellation, larger h to truncation.
H_VELOCITY = 1e-4
H_ACCELERATION = 1e-5


def sample(elapsed, q_start=Q_START, q_goal=Q_GOAL, duration=DURATION):
    return BottleArm.generate_trajectory(q_start, q_goal, duration, elapsed)


def smoothstep(u):
    """Closed-form quintic smoothstep, written independently of bottle.py."""
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def test_returns_three_python_floats():
    q = sample(1.0)
    assert isinstance(q, list)
    assert len(q) == 3
    assert all(isinstance(value, float) for value in q)


def test_starts_at_q_start():
    np.testing.assert_allclose(sample(0.0), Q_START, atol=1e-12)


def test_ends_at_q_goal():
    np.testing.assert_allclose(sample(DURATION), Q_GOAL, atol=1e-12)


def test_elapsed_before_zero_holds_q_start():
    np.testing.assert_allclose(sample(-1.5), Q_START, atol=1e-12)


def test_elapsed_after_duration_holds_q_goal():
    np.testing.assert_allclose(sample(DURATION + 5.0), Q_GOAL, atol=1e-12)


def test_halfway_is_exact_average():
    expected = (np.asarray(Q_START) + np.asarray(Q_GOAL)) / 2.0
    np.testing.assert_allclose(sample(DURATION / 2.0), expected, atol=1e-12)


def test_matches_closed_form_smoothstep():
    for u in np.linspace(0.0, 1.0, 17):
        expected = np.asarray(Q_START) + (np.asarray(Q_GOAL) - np.asarray(Q_START)) * smoothstep(u)
        np.testing.assert_allclose(
            sample(u * DURATION),
            expected,
            atol=1e-12,
            err_msg=f'trajectory differs from smoothstep at u={u}',
        )


def test_matches_repo_solver_with_zero_end_derivatives():
    """Cross-check the node's sampling against the shared boundary-condition solver."""
    times = np.linspace(0.0, DURATION, 13)
    for joint, (q0, qf) in enumerate(zip(Q_START, Q_GOAL)):
        coeffs = trajectory_coefficients(0.0, DURATION, q0, qf, 0.0, 0.0, 0.0, 0.0)
        expected = [evaluate(coeffs, t) for t in times]
        actual = [sample(t)[joint] for t in times]
        np.testing.assert_allclose(
            actual,
            expected,
            atol=1e-9,
            err_msg=f'joint {joint} differs from the solver',
        )


def test_velocity_is_zero_at_both_ends():
    h = H_VELOCITY
    start = (np.asarray(sample(h)) - np.asarray(sample(0.0))) / h
    end = (np.asarray(sample(DURATION)) - np.asarray(sample(DURATION - h))) / h
    np.testing.assert_allclose(start, np.zeros(3), atol=1e-6)
    np.testing.assert_allclose(end, np.zeros(3), atol=1e-6)


def test_acceleration_is_zero_at_both_ends():
    # A cubic would leave |q''| = 6*dq/T^2 = 2.0 here, so this catches a quintic
    # that was silently replaced by a cubic.
    h = H_ACCELERATION
    start = (np.asarray(sample(2 * h)) - 2 * np.asarray(sample(h)) + np.asarray(sample(0.0))) / h**2
    end = (
        np.asarray(sample(DURATION))
        - 2 * np.asarray(sample(DURATION - h))
        + np.asarray(sample(DURATION - 2 * h))
    ) / h**2
    np.testing.assert_allclose(start, np.zeros(3), atol=5e-3)
    np.testing.assert_allclose(end, np.zeros(3), atol=5e-3)


def test_no_overshoot_between_endpoints():
    low = np.minimum(Q_START, Q_GOAL)
    high = np.maximum(Q_START, Q_GOAL)
    for elapsed in np.linspace(0.0, DURATION, 61):
        q = np.asarray(sample(elapsed))
        assert np.all(q >= low - 1e-12) and np.all(q <= high + 1e-12)


def test_progress_is_monotonic_for_single_joint_move():
    positions = [sample(t, [0.0], [1.0])[0] for t in np.linspace(0.0, DURATION, 61)]
    assert all(a <= b + 1e-12 for a, b in zip(positions, positions[1:]))


def test_zero_duration_returns_q_goal():
    np.testing.assert_allclose(sample(0.0, duration=0.0), Q_GOAL, atol=1e-12)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")

    print()
    print(f"{len(tests)} tests passed")
