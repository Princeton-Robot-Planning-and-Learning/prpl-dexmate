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

## Development

Run all CI checks locally with:

```bash
./run_ci_checks.sh
```
