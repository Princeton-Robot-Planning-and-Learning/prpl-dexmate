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

## Development

Run all CI checks locally with:

```bash
./run_ci_checks.sh
```
