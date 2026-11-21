"""Tests for the JSON trajectory data packet built by trajectory_data.py.

Runs standalone (``python tests/test_trajectory_data.py``) or under pytest.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trajectory_data import to_json, trajectory_data
from trajectory import (
    sample_acceleration,
    sample_jerk,
    sample_trajectory,
    sample_velocity,
)

Q_START = [0.0, 0.5, 2.0]
Q_GOAL = [1.0, -1.5, 2.5]
DURATION = 3.0
SAMPLES = 13
QUANTITIES = {
    'position': sample_trajectory,
    'velocity': sample_velocity,
    'acceleration': sample_acceleration,
    'jerk': sample_jerk,
}


def build(**overrides):
    kwargs = dict(q_start=Q_START, q_goal=Q_GOAL, duration=DURATION,
                  sample_count=SAMPLES, trajectory_equation_type='quintic')
    kwargs.update(overrides)
    return trajectory_data(**kwargs)


def test_packet_has_expected_keys_and_metadata():
    data = build()
    for key in ('equation_type', 'duration', 'sample_count', 'joint_count',
                'q_start', 'q_goal', 'units', 'time', 'series'):
        assert key in data, f'missing key {key!r}'
    assert data['equation_type'] == 'quintic'
    assert data['duration'] == DURATION
    assert data['sample_count'] == SAMPLES
    assert data['joint_count'] == len(Q_START)
    assert data['q_start'] == Q_START
    assert data['q_goal'] == Q_GOAL
    assert list(data['series']) == list(QUANTITIES)
    assert list(data['units']) == list(QUANTITIES)


def test_series_shapes_match_the_sample_grid():
    data = build()
    assert len(data['time']) == SAMPLES
    for quantity, per_joint in data['series'].items():
        assert len(per_joint) == len(Q_START), f'{quantity}: wrong joint count'
        for joint_values in per_joint:
            assert len(joint_values) == SAMPLES, f'{quantity}: wrong sample count'
    np.testing.assert_allclose(data['time'], np.linspace(0.0, DURATION, SAMPLES))


def test_packet_values_match_the_samplers():
    data = build()
    for quantity, sampler in QUANTITIES.items():
        for joint in range(len(Q_START)):
            expected = [sampler(Q_START, Q_GOAL, DURATION, t, 'quintic')[joint]
                        for t in data['time']]
            np.testing.assert_allclose(
                data['series'][quantity][joint], expected, atol=1e-12,
                err_msg=f'{quantity} joint {joint} differs from the sampler')


def test_packet_endpoints_hold_the_boundary_positions():
    data = build()
    for joint, (q0, qf) in enumerate(zip(Q_START, Q_GOAL)):
        assert data['series']['position'][joint][0] == q0
        assert data['series']['position'][joint][-1] == qf
        assert data['series']['velocity'][joint][0] == 0.0
        assert data['series']['velocity'][joint][-1] == 0.0


def test_packet_carries_the_requested_equation_type():
    data = build(trajectory_equation_type='cubic')
    assert data['equation_type'] == 'cubic'
    assert data['series']['acceleration'][0][0] != 0.0


def test_packet_reports_radian_units():
    data = build()
    assert data['units'] == {
        'position': 'rad',
        'velocity': 'rad/s',
        'acceleration': 'rad/s\u00b2',
        'jerk': 'rad/s\u00b3',
    }


def test_packet_is_json_serialisable_and_round_trips():
    data = build()
    restored = json.loads(to_json(data))
    assert restored == data


def test_invalid_inputs_are_rejected():
    bad_calls = (
        dict(q_start=[0.0], q_goal=[0.0, 1.0]),
        dict(duration=0.0),
        dict(duration=-1.0),
        dict(sample_count=1),
        dict(trajectory_equation_type='quartic'),
    )
    for overrides in bad_calls:
        try:
            build(**overrides)
        except ValueError:
            continue
        raise AssertionError(f'{overrides} should have raised ValueError')


def test_data_and_plot_clis_share_the_same_packet_defaults():
    """Both CLIs must expose identical packet options (one shared helper)."""
    import generate_plot
    import trajectory_data

    data_args = vars(trajectory_data.parse_args([]))
    plot_args = vars(generate_plot.parse_args([]))
    for key in ('equation_type', 'duration', 'q_start', 'q_goal', 'samples'):
        assert key in data_args, f'data CLI missing {key!r}'
        assert key in plot_args, f'plot CLI missing {key!r}'
        assert data_args[key] == plot_args[key], f'CLIs disagree on {key!r}'


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")

    print()
    print(f"{len(tests)} tests passed")
