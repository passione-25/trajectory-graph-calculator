"""Tests for quantity selection in generate_plot.py.

Runs standalone (``python tests/test_plot_selection.py``) or under pytest.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate_plot import (
    DEFAULT_QUANTITIES,
    QUANTITY_SYMBOLS,
    build_figure,
    select_quantities,
)
from trajectory_data import trajectory_data

Q_START = [0.0, 0.0]
Q_GOAL = [1.0, -2.0]
DURATION = 2.0
SYMBOLS = {'position': '\u03b8', 'velocity': '\u03c9',
           'acceleration': '\u03b1', 'jerk': '\u03b6'}


def build_data():
    return trajectory_data(Q_START, Q_GOAL, DURATION, sample_count=9)


def test_default_selection_is_position_only():
    assert DEFAULT_QUANTITIES == ('position',)
    assert select_quantities() == ('position',)


def test_any_combination_is_accepted():
    assert select_quantities(['position', 'acceleration', 'jerk']) == \
        ('position', 'acceleration', 'jerk')
    assert select_quantities(['position', 'jerk']) == ('position', 'jerk')
    assert select_quantities(['velocity']) == ('velocity',)
    assert select_quantities(['velocity', 'position']) == ('position', 'velocity')
    assert select_quantities(['jerk', 'acceleration', 'velocity', 'position']) == \
        ('position', 'velocity', 'acceleration', 'jerk')


def test_single_string_and_duplicates_are_handled():
    assert select_quantities('velocity') == ('velocity',)
    assert select_quantities(['jerk', 'jerk', 'position']) == ('position', 'jerk')


def test_unknown_or_empty_selection_is_rejected():
    for bad in (['bogus'], [], 'nope', [None]):
        try:
            select_quantities(bad)
        except ValueError:
            continue
        raise AssertionError(f'{bad!r} should have raised ValueError')


def test_quantity_missing_from_packet_is_rejected():
    data = build_data()
    partial = {'position': data['units']['position'], 'jerk': data['units']['jerk']}
    assert select_quantities(['jerk'], list(partial)) == ('jerk',)
    try:
        select_quantities(['velocity'], list(partial))
    except ValueError:
        pass
    else:
        raise AssertionError('velocity absent from the packet should raise ValueError')


def test_figure_has_one_panel_per_selected_quantity():
    data = build_data()
    for selection in (None, ['position'], ['velocity'], ['position', 'jerk'],
                      ['position', 'acceleration', 'jerk'],
                      ['position', 'velocity', 'acceleration', 'jerk']):
        expected = select_quantities(selection, list(data['units']))
        fig = build_figure(data, selection)
        try:
            axes = fig.get_axes()
            assert len(axes) == len(expected), \
                f'{selection}: expected {len(expected)} panels, got {len(axes)}'
            for axes_obj, quantity in zip(axes, expected):
                assert axes_obj.get_ylabel().split('\n')[0] == SYMBOLS[quantity]
            for axes_obj in axes:
                assert len(axes_obj.get_lines()) == len(Q_START) + 1  # + zero line
        finally:
            plt.close(fig)


def test_axis_labels_use_the_greek_symbols_and_radian_units():
    fig = build_figure(build_data(), ['position', 'velocity', 'acceleration', 'jerk'])
    try:
        labels = [axes.get_ylabel() for axes in fig.get_axes()]
    finally:
        plt.close(fig)
    assert labels == [
        '\u03b8\n[rad]',
        '\u03c9\n[rad/s]',
        '\u03b1\n[rad/s\u00b2]',
        '\u03b6\n[rad/s\u00b3]',
    ]
    assert list(QUANTITY_SYMBOLS.values()) == ['\u03b8', '\u03c9', '\u03b1', '\u03b6']


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")

    print()
    print(f"{len(tests)} tests passed")
