#!/usr/bin/env python3
"""Render a trajectory data packet as a PNG.

The data comes from ``trajectory_data.trajectory_data`` (or a JSON file produced
by it), so this script only handles matplotlib. Output goes to ``plots/``.

    python generate_plot.py
    python generate_plot.py --equation-type cubic
    python generate_plot.py --data out.json --output plots/remote.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from trajectory_data import (
    add_packet_arguments,
    display,
    trajectory_data,
    validate_packet_arguments,
)

COLORS = ('#4c9be8', '#e8954c', '#5fc46e', '#c76fd0', '#d4c04a')
PLOTS_DIR = Path(__file__).resolve().parent / 'plots'


def joint_color(joint, total):
    """Per-joint colour; falls back to a continuous colormap past the palette size."""
    if total <= len(COLORS):
        return COLORS[joint]
    return matplotlib.colormaps['viridis'](joint / max(1, total - 1))


def build_figure(data):
    """Build the position/velocity/acceleration/jerk figure for a data packet."""
    times = np.asarray(data['time'], dtype=float)
    joint_count = int(data['joint_count'])
    quantities = list(data['units'])
    duration = float(data['duration'])

    fig, axes_list = plt.subplots(
        len(quantities), 1, figsize=(8.0, 11.0), dpi=150, sharex=True)
    if len(quantities) == 1:
        axes_list = [axes_list]

    for axes, quantity in zip(axes_list, quantities):
        for joint in range(joint_count):
            if quantity == 'position':
                name = (f"joint {joint}: "
                        f"{data['q_start'][joint]:g} \u2192 {data['q_goal'][joint]:g}")
            else:
                name = f'joint {joint}'
            axes.plot(times, np.asarray(data['series'][quantity][joint], dtype=float),
                      color=joint_color(joint, joint_count), linewidth=1.6, label=name)
        axes.set_ylabel(f"{quantity}\n[{data['units'][quantity]}]")
        axes.set_xlim(0.0, duration)
        axes.axhline(0.0, color='#9aa0a6', linewidth=0.8)
        axes.grid(True, alpha=0.3)

    axes_list[0].set_title(
        f"{data['equation_type']} trajectory over {duration:g} s")
    axes_list[0].legend(loc='best', fontsize=8)
    axes_list[-1].set_xlabel('time [s]')

    fig.tight_layout()
    return fig


def resolve_output(raw, equation_type):
    if raw is None:
        return PLOTS_DIR / f'trajectory_{equation_type}.png'
    path = Path(raw).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def save_plot(data, output=None):
    """Render ``data`` to ``output`` and return the path that was written."""
    path = resolve_output(output, data['equation_type'])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure(data)
    fig.savefig(path)
    plt.close(fig)
    return path


def load_packet(raw):
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return json.loads(path.read_text(encoding='utf-8'))


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description='Render a trajectory data packet as a PNG.')
    parser.add_argument('--data', metavar='JSON',
                        help='plot a data packet from this JSON file instead of sampling')
    add_packet_arguments(parser)
    parser.add_argument('--output', metavar='PATH',
                        help='PNG path (default: plots/trajectory_<type>.png)')
    args = parser.parse_args(argv)
    if args.data is None:
        validate_packet_arguments(parser, args)
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.data:
        data = load_packet(args.data)
    else:
        data = trajectory_data(args.q_start, args.q_goal, args.duration,
                               args.samples, args.equation_type)

    path = save_plot(data, args.output)
    print(f"saved {display(path)} "
          f"({data['sample_count']} samples, {data['joint_count']} joints)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
