# trajectory-graph-calculator

Cubic and quintic trajectory generation that satisfies position, velocity and
acceleration boundary conditions, with sampling and plotting.

## Module

All the math lives in `trajectory.py`:

- `trajectory_coefficients(t0, tf, q0, qf, w0=0, wf=0, a0=0, af=0,
  trajectory_equation_type='quintic')` returns ascending coefficients
  `[c0, c1, ...]`, i.e. `c0 + c1*t + c2*t**2 + ...`.
  - `'quintic'` (default): 6 coefficients; matches position, velocity **and**
    acceleration at both ends.
  - `'cubic'`: 4 coefficients; matches position and velocity at both ends. A
    non-zero `a0`/`af` raises `ValueError`, since a cubic cannot honour them.
  - Any `t0`/`tf` and boundary values are accepted; `tf` must exceed `t0`.
- `evaluate(coeffs, t)` - value of the polynomial at `t`.
- `differentiate(coeffs, order=1)` - ascending coefficients of the n-th derivative.
- `format_equation(coeffs)` - readable `q(t) = ...` string.
- `sample_trajectory` / `sample_velocity` / `sample_acceleration` / `sample_jerk`
  `(q_start, q_goal, duration, elapsed, trajectory_equation_type='quintic')` -
  per-joint sampling clamped to `[0, duration]`.

Rest-to-rest moves (zero boundary velocity and acceleration) use exact closed
forms: `10u^3 - 15u^4 + 6u^5` for the quintic and `3u^2 - 2u^3` for the cubic,
with `u = elapsed / duration`. The tests check these against the general solver.

## Scripts

- `example.py` - prints the data packet as JSON and saves its plot on every run.
- `trajectory_data.py` - builds the **data packet**. `trajectory_data(...)` returns
  a plain dict of every sampled series (JSON-serialisable, safe to POST to a
  remote server); the CLI prints it as JSON or writes it with `--output`. No
  matplotlib import, so this layer works on a numbers-only machine.
- `generate_plot.py` - renders a packet as `plots/trajectory_<type>.png`. It uses
  `trajectory_data` directly, or `--data packet.json` to plot a received packet.
  `--plot` chooses which panels to draw (any combination, default `position`);
  positions use `theta`, velocity `omega`, acceleration `alpha`, jerk `zeta`.

  | quantity | panel label | unit |
  |---|---|---|
  | position | `θ` (theta) | rad |
  | velocity | `ω` (omega) | rad/s |
  | acceleration | `α` (alpha) | rad/s² |
  | jerk | `ζ` (zeta) | rad/s³ |

  Units are labels only: the samplers work on plain numbers and never convert.
  The project's convention is radians, so pass joint values in radians. A
  consumer wanting degrees scales by `180 / pi` per derivative order.
- `bottle.py` - ROS 2 node; needs a `rclpy` / `hw_arm_bottle` workspace, so it is
  not importable without ROS. It depends on `trajectory.py` only.

Layer order, strictly one direction (each layer is usable without the next):

```
trajectory.py  ->  trajectory_data.py  ->  generate_plot.py
      \\                                   /
       +-> bottle.py, example.py, tests -+
```

`trajectory_data.add_packet_arguments` is shared by both CLIs, so the data and
plotting commands cannot drift apart.

### Data packet

```python
from trajectory_data import trajectory_data

data = trajectory_data([0.0, 0.0, 0.0], [1.0, 2.0, -3.0], 3.0)
# {
#   'equation_type': 'quintic', 'duration': 3.0, 'sample_count': 241,
#   'joint_count': 3, 'q_start': [...], 'q_goal': [...],
#   'units': {'position': 'rad', 'velocity': 'rad/s', ...},
#   'time': [0.0, ...],
#   'series': {'position': [[joint 0], [joint 1], [joint 2]],
#              'velocity': [...], 'acceleration': [...], 'jerk': [...]},
# }
```

## Example

```powershell
.\.venv\Scripts\python.exe example.py                      # packet + plot
.\.venv\Scripts\python.exe trajectory_data.py --output plots/packet.json
.\.venv\Scripts\python.exe generate_plot.py                          # position only
.\.venv\Scripts\python.exe generate_plot.py --plot position jerk
.\.venv\Scripts\python.exe generate_plot.py --plot position acceleration jerk
.\.venv\Scripts\python.exe generate_plot.py --plot velocity --equation-type cubic
.\.venv\Scripts\python.exe generate_plot.py --data plots/packet.json
```

`--plot` accepts any combination of `position velocity acceleration jerk` (panels
are always drawn in that order). Both CLIs accept `--equation-type`, `--duration`,
`--q-start`, `--q-goal` and `--samples`, plus `--output`, and create the output
directory if needed.

## Testing

```powershell
.\.venv\Scripts\python.exe tests\test_trajectory.py
.\.venv\Scripts\python.exe tests\test_trajectory_data.py
.\.venv\Scripts\python.exe tests\test_quintic_equation.py
.\.venv\Scripts\python.exe tests\test_cubic_equation.py
.\.venv\Scripts\python.exe tests\test_bottle.py
```

`test_bottle.py` stubs the ROS imports (`rclpy`, `geometry_msgs`, `hw_arm_bottle`)
so the trajectory generator can be tested without a ROS install.

The test files use plain `assert` checks, so they also run under pytest if
installed (`python -m pytest tests`).
