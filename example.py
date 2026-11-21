#!/usr/bin/env python3
"""Example: print the trajectory data packet, then save its plot.

    python example.py

Runs the full chain end to end:

    trajectory.py  ->  trajectory_data.py  ->  generate_plot.py

The packet printed here is the same dict a remote server would receive.
"""

from generate_plot import save_plot
from trajectory_data import display, to_json, trajectory_data

Q_START = [0.0, 0.0, 0.0]
Q_GOAL = [1.0, 2.0, -3.0]
DURATION = 3.0


def main():
    data = trajectory_data(Q_START, Q_GOAL, DURATION)

    print(to_json(data))
    print()

    print(f'saved {display(save_plot(data))} '
          f"({data['sample_count']} samples, {data['joint_count']} joints)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
