#!/usr/bin/env python3
"""Build a JSON-serialisable trajectory data packet.

``trajectory_data`` returns a plain dict of sampled series, which can be posted
to a remote server, written to disk as JSON, or handed to ``generate_plot.py``
for rendering. Nothing here imports matplotlib, so this layer is safe to use on
a machine that only needs the numbers.

    python trajectory_data.py                     # JSON to stdout
    python trajectory_data.py --output out.json
    python trajectory_data.py --equation-type cubic
"""

import argparse
import json
from pathlib import Path

import numpy as np

from trajectory import (
    DEFAULT_EQUATION_TYPE,
    EQUATION_TYPES,
    normalize_equation_type,
    sample_acceleration,
    sample_jerk,
    sample_trajectory,
    sample_velocity,
)

# Re-exported so the plotting layer depends on this module alone.
__all__ = [
    'DEFAULT_DURATION',
    'DEFAULT_EQUATION_TYPE',
    'DEFAULT_Q_GOAL',
    'DEFAULT_Q_START',
    'DEFAULT_SAMPLES',
    'EQUATION_TYPES',
    'QUANTITIES',
    'add_packet_arguments',
    'display',
    'parse_args',
    'to_json',
    'trajectory_data',
    'validate_packet_arguments',
]

DEFAULT_Q_START = [0.0, 0.0, 0.0]
DEFAULT_Q_GOAL = [1.0, 2.0, -3.0]
DEFAULT_DURATION = 3.0
DEFAULT_SAMPLES = 241

QUANTITIES = (
    ('position', sample_trajectory, 'unit'),
    ('velocity', sample_velocity, 'unit/s'),
    ('acceleration', sample_acceleration, 'unit/s\u00b2'),
    ('jerk', sample_jerk, 'unit/s\u00b3'),
)


def trajectory_data(q_start, q_goal, duration, sample_count=DEFAULT_SAMPLES,
                    trajectory_equation_type=DEFAULT_EQUATION_TYPE):
    """Return every sampled series for a joint move as a plain dict.

    Structure::

        {
          'equation_type': 'quintic',
          'duration': 3.0,
          'sample_count': 241,
          'joint_count': 3,
          'q_start': [...], 'q_goal': [...],
          'units': {'position': 'unit', 'velocity': 'unit/s', ...},
          'time': [...],
          'series': {
              'position': [[joint 0 samples], [joint 1 samples], ...],
              'velocity': [...], 'acceleration': [...], 'jerk': [...],
          }
        }

    All values are plain Python ``float``/``int``/``str`` so the result passes
    straight through ``json.dumps``.
    """
    kind = normalize_equation_type(trajectory_equation_type)
    duration = float(duration)
    if duration <= 0:
        raise ValueError(f'duration must be positive, got {duration}')
    if len(q_start) != len(q_goal):
        raise ValueError(
            f'q_start and q_goal need the same number of joints, '
            f'got {len(q_start)} and {len(q_goal)}')
    if sample_count < 2:
        raise ValueError(f'sample_count must be at least 2, got {sample_count}')

    times = np.linspace(0.0, duration, int(sample_count))
    joint_count = len(q_start)

    series = {}
    for name, sampler, _unit in QUANTITIES:
        rows = [sampler(q_start, q_goal, duration, t, kind) for t in times]
        series[name] = [[float(row[joint]) for row in rows] for joint in range(joint_count)]

    return {
        'equation_type': kind,
        'duration': duration,
        'sample_count': int(sample_count),
        'joint_count': joint_count,
        'q_start': [float(value) for value in q_start],
        'q_goal': [float(value) for value in q_goal],
        'units': {name: unit for name, _sampler, unit in QUANTITIES},
        'time': [float(t) for t in times],
        'series': series,
    }


def to_json(data, indent=2):
    """Serialise a packet from :func:`trajectory_data`."""
    return json.dumps(data, indent=indent)


def add_packet_arguments(parser):
    """Add the shared packet arguments to ``parser`` and return it.

    Both CLIs use this so the data and plotting commands can never disagree
    about defaults, choices or option names.
    """
    parser.add_argument('--equation-type', choices=EQUATION_TYPES,
                        default=DEFAULT_EQUATION_TYPE,
                        help=f'polynomial order (default: {DEFAULT_EQUATION_TYPE})')
    parser.add_argument('--duration', type=float, default=DEFAULT_DURATION,
                        help=f'move duration in seconds (default: {DEFAULT_DURATION})')
    parser.add_argument('--q-start', type=float, nargs='+', default=DEFAULT_Q_START,
                        metavar='Q', help='start joint positions')
    parser.add_argument('--q-goal', type=float, nargs='+', default=DEFAULT_Q_GOAL,
                        metavar='Q', help='goal joint positions')
    parser.add_argument('--samples', type=int, default=DEFAULT_SAMPLES,
                        help=f'samples per series (default: {DEFAULT_SAMPLES})')
    return parser


def validate_packet_arguments(parser, args):
    """Reject inconsistent packet arguments through ``parser.error``."""
    if len(args.q_start) != len(args.q_goal):
        parser.error('--q-start and --q-goal need the same number of joints')
    if args.duration <= 0:
        parser.error('--duration must be positive')
    if args.samples < 2:
        parser.error('--samples must be at least 2')
    return args


def display(path):
    """Render ``path`` relative to the working directory when possible."""
    try:
        return Path(path).relative_to(Path.cwd()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description='Emit a JSON trajectory data packet (no plotting).')
    add_packet_arguments(parser)
    parser.add_argument('--output', metavar='PATH',
                        help='write JSON to PATH instead of stdout')
    parser.add_argument('--indent', type=int, default=2,
                        help='JSON indent (default: 2; use 0 for compact output)')
    args = parser.parse_args(argv)
    return validate_packet_arguments(parser, args)


def main(argv=None):
    args = parse_args(argv)
    data = trajectory_data(args.q_start, args.q_goal, args.duration,
                           args.samples, args.equation_type)
    text = to_json(data, indent=args.indent or None)

    if args.output:
        path = Path(args.output).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + '\n', encoding='utf-8')
        print(f'saved {display(path)}')
    else:
        print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
