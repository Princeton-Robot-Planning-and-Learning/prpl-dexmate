#!/usr/bin/env bash
# Launch the prpl-dexmate tmuxinator session, optionally syncing the
# Jetson to a specific branch first.
#
# Usage:
#   scripts/launch.sh                  # no branch sync
#   scripts/launch.sh <branch-name>    # hard-sync the Jetson to origin/<branch>
#
# This is a thin wrapper around `tmuxinator start ./.tmuxinator.yml`,
# which (unlike `tmuxinator local`) forwards positional args to the
# project's ERB as `@args`.

set -euo pipefail

cd "$(dirname "$0")/.."

exec tmuxinator start ./.tmuxinator.yml ${1:+"$1"}
