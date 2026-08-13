#!/usr/bin/env bash
# Tear down the skill server and the tmuxinator session.
# Verifies each step and exits non-zero if anything survived.
#
# Ported from prpl-tidybot's stop_servers.sh, with one addition: a
# best-effort RPC stop first, so a directive mid-execution is halted
# through the server's own safe-stop path before the process is killed.
# (Killing the server mid-directive is also safe — streaming ceases and
# the robot holds position — but the RPC stop is the polite version.)
#
# This stops software only. If you are about to power the robot off,
# fold the arms FIRST (scripts/park_arms.py --to fold), while the
# server is still running.

set -uo pipefail

cd "$(dirname "$0")/.."

HOST="${PRPL_VEGA_HOST:-vega}"
SESSION="${PRPL_TMUX_SESSION:-prpl-dexmate}"
SERVER_PATTERN="prpl_dexmate.remote.server"

# This is an orchestrator-side script: it reaches the robot over the SSH
# alias and kills the local tmux session. Refuse to run on the robot
# itself (the alias and the session both live on the orchestrator).
if [ -f ~/.prpl_robot_env ]; then
    echo "ERROR: this looks like the robot (~/.prpl_robot_env exists)." >&2
    echo "Run scripts/stop.sh on the orchestrator machine instead." >&2
    exit 64
fi

# Refuse to run from inside the session being killed: the final
# kill-session would take this script's own shell down mid-run.
if [ -n "${TMUX:-}" ] && [ "$(tmux display-message -p '#S' 2>/dev/null)" = "$SESSION" ]; then
    echo "ERROR: you are inside the '$SESSION' tmux session this script kills." >&2
    echo "Detach first (Ctrl-b d), then run scripts/stop.sh from that terminal." >&2
    exit 64
fi

failed=0

err() {
    echo "ERROR: $*" >&2
    failed=1
}

# Best-effort graceful stop of any running directive via the RPC surface.
if [[ -x .venv/bin/python ]]; then
    echo "Sending RPC stop (best effort)..."
    .venv/bin/python - <<'EOF' 2>/dev/null || echo "  (no reachable server; skipping)"
from prpl_dexmate.remote.client import SkillClient
from prpl_dexmate.remote.server import DEFAULT_PORT
import os

client = SkillClient(os.environ.get("PRPL_VEGA_IP", "192.168.0.169"), DEFAULT_PORT,
                     connect_timeout=3.0, request_timeout=3.0)
client.stop()
client.close()
print("  stop sent.")
EOF
fi

# Remote script: kills python processes matching $1, then prints any survivors
# (filtered by comm == python so the bash shell running this script — whose
# argv contains the pattern — doesn't show up as a survivor).
read -r -d '' REMOTE_SCRIPT <<'REMOTE' || true
pattern="$1"
is_python() {
    case "$(cat /proc/"$1"/comm 2>/dev/null)" in
        python*) return 0 ;;
    esac
    return 1
}
survivors() {
    for pid in $(pgrep -f "$pattern"); do
        is_python "$pid" && echo "$pid"
    done
}
for pid in $(survivors); do
    kill "$pid" 2>/dev/null
done
# The server shuts down dexcontrol/zenoh on exit, which can take a few
# seconds; give SIGTERM a grace period before escalating to SIGKILL.
for _ in 1 2 3 4 5; do
    [ -z "$(survivors)" ] && break
    sleep 1
done
for pid in $(survivors); do
    kill -9 "$pid" 2>/dev/null
done
sleep 1
for pid in $(survivors); do
    tr '\0' ' ' < /proc/"$pid"/cmdline 2>/dev/null
    echo
done
REMOTE

echo "Stopping '$SERVER_PATTERN' on $HOST..."
out=$(ssh -o ConnectTimeout=10 "$HOST" bash -s "$SERVER_PATTERN" <<<"$REMOTE_SCRIPT" 2>&1)
rc=$?
if [[ $rc -ne 0 ]]; then
    err "ssh to $HOST failed (exit $rc):"
    if [[ -n "$out" ]]; then
        printf '%s\n' "$out" | sed 's/^/  /' >&2
    else
        echo "  (no output — likely auth failure; run 'ssh $HOST true' to debug)" >&2
    fi
elif [[ -n "$out" ]]; then
    err "processes matching '$SERVER_PATTERN' still running on $HOST:"
    printf '%s\n' "$out" | sed 's/^/  /' >&2
fi

echo "Killing tmux session '$SESSION'..."
if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        err "tmux session '$SESSION' still exists after kill-session"
    fi
else
    echo "  (no session named '$SESSION')"
fi

if [[ $failed -ne 0 ]]; then
    echo >&2
    echo "FAILED: one or more cleanup steps did not complete." >&2
    exit 1
fi

echo "All clean."
