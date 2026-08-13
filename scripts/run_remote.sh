#!/usr/bin/env bash
# Launch a command on the Vega's Jetson with the repo venv activated.
# Usage: scripts/run_remote.sh <ssh-target> <command...>
#
# Ported from prpl-tidybot's run_remote.sh, minus the opencv wheel hack,
# and with `uv venv` + `uv pip install -e .` in place of `uv sync`
# (this repo has no committed uv.lock).

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <ssh-target> <command...>" >&2
    exit 64
fi

host="$1"
shift
repo_dir="${PRPL_REMOTE_REPO_DIR:-~/prpl-dexmate}"
branch="${PRPL_BRANCH:-}"

# `uv`'s installer drops the binary in ~/.local/bin/, which is typically only
# added to PATH by the user's interactive shell rc (~/.bashrc). The SSH command
# below runs a non-interactive non-login shell that doesn't source any rc file,
# so `uv` isn't visible without this explicit prepend. The `$HOME` here is
# evaluated on the remote, not locally (single-quoted).
#
# ~/.prpl_robot_env on the robot exports the robot's identity
# (ROBOT_NAME, DEXCONTROL_COMM_CFG_PATH), which dexcontrol needs to
# construct a Robot(); non-interactive shells never source ~/.bashrc, so
# without this the server pane dies with "Variant not specified".
remote_path_prefix='export PATH="$HOME/.local/bin:$PATH" && { [ ! -f ~/.prpl_robot_env ] || . ~/.prpl_robot_env; } && '

# Bootstrap chained into the SSH command via `&&` so any step's failure
# aborts the pane instead of starting the server (or shell) against
# stale code or a stale venv.
#
# 1. Optional git sync. When PRPL_BRANCH is set, force the Jetson
#    checkout into exact alignment with origin/<branch>. The Jetson is
#    NOT a development box — it's an ephemeral mirror of whatever was
#    most recently pushed — so origin is unambiguously the source of
#    truth and the local branch must always re-snap to it (which also
#    recovers from force-pushes).
#
#    Before snapping, detect and refuse uncommitted modifications or
#    untracked files (`git status --porcelain` non-empty), so local
#    edits are never silently discarded.
#
#    `git checkout -B <branch> origin/<branch>` always (re)creates the
#    local branch pointing at origin/<branch>, regardless of any prior
#    state of the local branch.
#
# 2. Dependency sync: create the venv if missing, then reinstall the
#    package so pyproject.toml changes on the just-synced branch take
#    effect before any Python code runs. Fast when nothing changed
#    (uv's cache covers the pinned git dependencies).
#
# Concurrent-pane handling: set PRPL_SKIP_SYNC=1 on every pane except
# the one that should do the syncing. The sync pane holds an exclusive
# flock on .git/run_remote.lock for the duration of git + uv; the
# skipping panes acquire a shared flock on the same file (which blocks
# until the exclusive holder releases) and then go straight to
# launching the command. This way the skippers never run their command
# against a half-synced tree or venv.
deps="([ -d .venv ] || uv venv --python 3.10) && uv pip install -e ."
if [[ -n "${PRPL_SKIP_SYNC:-}" ]]; then
    sync="echo 'run_remote.sh: PRPL_SKIP_SYNC set; waiting for the sync pane to finish before launching...' && (flock -s 9) 9>.git/run_remote.lock && echo 'run_remote.sh: sync pane finished; launching.' && "
elif [[ -n "$branch" ]]; then
    sync="(flock -x 9 && git fetch origin $branch && if [ -n \"\$(git status --porcelain)\" ]; then echo 'run_remote.sh: ERROR: remote checkout has uncommitted modifications or untracked files; refusing to overwrite. Resolve manually before re-launching.' >&2; git status --short >&2; exit 1; fi && git checkout -B $branch origin/$branch && $deps) 9>.git/run_remote.lock && "
else
    sync="$deps && "
fi

# -tt forces a PTY in both directions so closing the local pane delivers
# SIGHUP to the remote shell, killing the Python process cleanly.
if [[ "$*" == "bash" ]]; then
    # Interactive shell: use --rcfile so .bashrc loads first and the venv
    # activation runs after, otherwise .bashrc resets PS1 (and possibly
    # PATH) and the venv prefix disappears from the prompt.
    exec ssh -tt "$host" "${remote_path_prefix}cd $repo_dir && ${sync}exec bash --rcfile <(echo \"[ -f ~/.bashrc ] && source ~/.bashrc; cd $repo_dir; source .venv/bin/activate\")"
else
    exec ssh -tt "$host" "${remote_path_prefix}cd $repo_dir && ${sync}source .venv/bin/activate && exec $*"
fi
