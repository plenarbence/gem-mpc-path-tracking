#!/usr/bin/env bash

set -euo pipefail

export DISPLAY="${DISPLAY:-:1}"

SCREEN_GEOMETRY="${SCREEN_GEOMETRY:-1600x900x24}"
VNC_PORT="${VNC_PORT:-5900}"
NOVNC_PORT="${NOVNC_PORT:-6080}"
DISPLAY_NUMBER="${DISPLAY#:}"
X11_SOCKET="/tmp/.X11-unix/X${DISPLAY_NUMBER}"

pids=()

cleanup() {
    trap - EXIT INT TERM
    if ((${#pids[@]})); then
        kill "${pids[@]}" 2>/dev/null || true
        wait "${pids[@]}" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

Xvfb "$DISPLAY" \
    -screen 0 "$SCREEN_GEOMETRY" \
    -ac \
    -nolisten tcp \
    +extension GLX \
    +render \
    -noreset &
pids+=("$!")

for _ in $(seq 1 50); do
    if [[ -S "$X11_SOCKET" ]]; then
        break
    fi
    sleep 0.1
done

if [[ ! -S "$X11_SOCKET" ]]; then
    echo "Virtual display $DISPLAY did not become ready." >&2
    exit 1
fi

fluxbox -display "$DISPLAY" &
pids+=("$!")

x11vnc \
    -display "$DISPLAY" \
    -forever \
    -shared \
    -nopw \
    -noxdamage \
    -listen 0.0.0.0 \
    -rfbport "$VNC_PORT" &
pids+=("$!")

websockify \
    --web=/usr/share/novnc \
    "$NOVNC_PORT" \
    "localhost:$VNC_PORT" &
pids+=("$!")

set +e
wait -n "${pids[@]}"
status=$?
set -e

exit "$status"
