# prpl-dexmate

![workflow](https://github.com/Princeton-Robot-Planning-and-Learning/prpl-dexmate/actions/workflows/ci.yml/badge.svg)

PRPL code for the Dexmate robot.

## Installation

```bash
pip install -e ".[develop]"
```

(Recommended: use a virtualenv, e.g. `uv venv && source .venv/bin/activate`.)

## Usage

Example scripts live in `examples/`. For instance:

```bash
python examples/read_joint_current.py
```

## Remote execution

Planning runs on an orchestrator machine (a lab laptop or workstation);
anything latency-sensitive runs on the Vega's onboard Jetson. The
orchestrator sends the Jetson one directive per skill invocation (e.g. a
whole joint trajectory) over the RPC boundary in `prpl_dexmate.remote`,
and the Jetson's skill server executes it locally at the full control
rate. Nothing streams per-control-step commands across the network.

Prerequisites, one-time:

- A clone of this repo on the Jetson at `~/prpl-dexmate` (override with
  `PRPL_REMOTE_REPO_DIR`), with `uv` installed there.
- A `~/.prpl_robot_env` file on the Jetson exporting the robot's
  identity and end-effector configuration, which dexcontrol needs to
  construct a `Robot()` (the launcher sources it in every pane;
  `vega_1u_gripper` is from `dexbot cfg list` and matches the mounted
  DexGripper S end effectors):

  ```bash
  export ROBOT_NAME=dm/vg78194c5120-1u
  export ROBOT_CONFIG=vega_1u_gripper
  export DEXCONTROL_COMM_CFG_PATH="$HOME/.dexmate/comm/zenoh/dm_vg78194c5120-1u.dzcfg"
  ```
- Passwordless SSH to the Jetson under an alias named `vega` (override
  with `PRPL_VEGA_HOST`) in `~/.ssh/config`:

  ```
  Host vega
    HostName 192.168.0.169
    User dexmate
  ```

- `tmuxinator` on the orchestrator (`brew install tmuxinator`).

Then:

```bash
scripts/launch.sh [branch-name]
```

This opens a tmux session with a shell pane on the Jetson, the skill
server pane, and a local orchestrator shell. With a branch name given,
the Jetson checkout is first hard-reset to `origin/<branch-name>` and
its dependencies re-synced; the Jetson is an ephemeral mirror of origin,
never a place where code is edited. The client and server exchange a
protocol version hash at connect time, so mismatched checkouts fail at
startup rather than subtly at runtime.

During development, prefer wired ethernet to the robot; the architecture
tolerates WiFi, but debugging is much easier wired.

### Session workflow

With the skill server running on the robot:

```bash
# Session start: unfold from the shipping fold to the model home
# (the resting pose while powered, and the pipeline's planning start).
python scripts/park_arms.py --to home --host <robot>

# Rollouts. Every directive shows a summary and a shadow-sim preview
# video at a confirm gate before anything moves.
python scripts/run_pipeline.py env=vega_motion3d mode=remote \
    env.pipelines.remote.real_env.host=<robot>

# Session end, before power-off: fold back. The folded arms rest on
# mechanical end-stops, so nothing sags when motor power cuts. (Whether
# the joints hold position unpowered in other poses is unverified —
# fold before every power-off.)
python scripts/park_arms.py --to fold --host <robot>

# Once grippers are mounted, pass --grippers everywhere (collision
# checks then include gripper geometry) and use the storage pose — the
# shipping fold self-collides with grippers and is refused:
python scripts/park_arms.py --to storage --grippers --host <robot>

# Then tear down the server and the tmux session (verifies nothing
# survived; exits non-zero otherwise).
scripts/stop.sh
```

`park_arms.py` observes the arms' actual positions, routes each arm
through home one at a time, collision-checks every straight-line segment
in sim before moving, and asks for confirmation per motion.

## Development

Run all CI checks locally with:

```bash
./run_ci_checks.sh
```
