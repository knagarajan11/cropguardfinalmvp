#!/usr/bin/env bash

set -u

# ============================================================
# CropGuard Final MVP - HOST-SIDE LIFECYCLE SCRIPT
#
# Run this script from the DGX host, inside the Slurm allocation.
#
# Architecture:
#   Host ~/CropGuard
#          |
#          +--> Apptainer /workspace
#                  |
#                  +--> FastAPI :8002
#                  +--> Streamlit :8502
#                  +--> Cloudflared
#
# Only the current Slurm job's recorded PIDs are stopped.
# Port 8000 is NEVER touched.
# ============================================================

set -o pipefail

# ------------------------------------------------------------
# Host paths
# ------------------------------------------------------------

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RUN_DIR="$ROOT/run"
LOG_DIR="$ROOT/logs"

mkdir -p "$RUN_DIR" "$LOG_DIR"

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

API_PORT="${CROPGUARD_API_PORT:-8002}"
UI_PORT="${CROPGUARD_UI_PORT:-8502}"

CONTAINER="${CROPGUARD_CONTAINER:-$HOME/nemo_sandbox}"

CF="${CROPGUARD_CLOUDFLARED:-$HOME/bin/cloudflared}"
CF_TOKEN_FILE="${CROPGUARD_CF_TOKEN_FILE:-$ROOT/secrets/cloudflare_tunnel_token}"
CF_PUBLIC_HOSTNAME="${CROPGUARD_CF_HOSTNAME:-demo.cropguard.in}"

JOB_ID="${SLURM_JOB_ID:-no-slurm}"
JOB_TAG="cropguard_${JOB_ID}"

API_PID="$RUN_DIR/${JOB_TAG}_fastapi.pid"
UI_PID="$RUN_DIR/${JOB_TAG}_streamlit.pid"
CF_PID="$RUN_DIR/${JOB_TAG}_cloudflared.pid"

API_LOG="$LOG_DIR/${JOB_TAG}_fastapi.log"
UI_LOG="$LOG_DIR/${JOB_TAG}_streamlit.log"
CF_LOG="$LOG_DIR/${JOB_TAG}_cloudflared.log"

# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

validate_environment() {

    if [[ ! -d "$ROOT" ]]; then
        echo "ERROR: CropGuard root does not exist:"
        echo "$ROOT"
        return 1
    fi

    if [[ ! -f "$ROOT/api/main.py" ]]; then
        echo "ERROR: CropGuard API not found:"
        echo "$ROOT/api/main.py"
        return 1
    fi

    if [[ ! -d "$CONTAINER" ]]; then
        echo "ERROR: Apptainer sandbox not found:"
        echo "$CONTAINER"
        return 1
    fi

    if ! command -v apptainer >/dev/null 2>&1; then
        echo "ERROR: apptainer command not found."
        return 1
    fi

    if ! command -v curl >/dev/null 2>&1; then
        echo "ERROR: curl command not found."
        return 1
    fi

    if ! command -v lsof >/dev/null 2>&1; then
        echo "ERROR: lsof command not found."
        return 1
    fi

    if [[ ! -x "$CF" ]]; then
        echo "ERROR: cloudflared not found or not executable:"
        echo "$CF"
        return 1
    fi

    if [[ ! -f "$CF_TOKEN_FILE" ]]; then
        echo "ERROR: Cloudflare token file not found:"
        echo "$CF_TOKEN_FILE"
        return 1
    fi

    if [[ ! -r "$CF_TOKEN_FILE" ]]; then
        echo "ERROR: Cloudflare token file is not readable:"
        echo "$CF_TOKEN_FILE"
        return 1
    fi

    if [[ "$JOB_ID" == "no-slurm" ]]; then
        echo
        echo "WARNING: SLURM_JOB_ID is not set."
        echo "CropGuard should normally be started inside an srun allocation."
        echo
    fi

    return 0
}

# ------------------------------------------------------------
# Process helpers
# ------------------------------------------------------------

pid_alive() {

    local pid="$1"

    [[ -n "$pid" ]] || return 1

    kill -0 "$pid" 2>/dev/null
}

read_pid() {

    local file="$1"

    [[ -f "$file" ]] || return 1

    cat "$file"
}

kill_owned_process() {

    local name="$1"
    local pidfile="$2"

    if [[ ! -f "$pidfile" ]]; then
        echo "$name: not running"
        return 0
    fi

    local pid
    pid="$(cat "$pidfile" 2>/dev/null || true)"

    if [[ -z "$pid" ]]; then
        echo "$name: invalid PID file"
        rm -f "$pidfile"
        return 0
    fi

    if ! pid_alive "$pid"; then
        echo "$name: already stopped (PID $pid)"
        rm -f "$pidfile"
        return 0
    fi

    echo "Stopping $name (PID $pid)..."

    kill "$pid" 2>/dev/null || true

    for _ in $(seq 1 15); do

        if ! pid_alive "$pid"; then
            echo "$name: stopped"
            rm -f "$pidfile"
            return 0
        fi

        sleep 1
    done

    echo "$name: did not stop gracefully; sending SIGKILL to PID $pid"

    kill -9 "$pid" 2>/dev/null || true

    rm -f "$pidfile"

    echo "$name: stopped"
}

# ------------------------------------------------------------
# Apptainer command
# ------------------------------------------------------------

apptainer_exec() {

    apptainer exec --nv \
        --bind "$ROOT:/workspace" \
        --pwd /workspace \
        "$CONTAINER" \
        "$@"
}

# ------------------------------------------------------------
# Port helpers
# ------------------------------------------------------------

port_pids() {

    local port="$1"

    lsof -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true
}

port_free() {

    local port="$1"

    [[ -z "$(port_pids "$port")" ]]
}

# ------------------------------------------------------------
# RECOMMENDATION INDEX
# ------------------------------------------------------------

build_recommendation_index() {

    local mode="$1"

    echo
    echo "========================================"
    echo "       RECOMMENDATION INDEX"
    echo "========================================"
    echo "Mode: $mode"
    echo

    if ! apptainer_exec \
        /opt/venv/bin/python \
        /workspace/index/recommendation_index.py \
        > "$LOG_DIR/${JOB_TAG}_recommendation_index.log" 2>&1; then

        echo
        echo "ERROR: Recommendation index $mode failed."
        echo
        echo "Last index log:"
        tail -150 "$LOG_DIR/${JOB_TAG}_recommendation_index.log"

        return 1
    fi

    echo
    echo "Recommendation index $mode completed successfully."
    echo
    tail -80 "$LOG_DIR/${JOB_TAG}_recommendation_index.log"
    echo
    echo "Recommendation index: READY"
    echo

    return 0
}

# ------------------------------------------------------------
# START
# ------------------------------------------------------------

start() {

    # Default behavior:
    #   ./scripts/cropguard.sh start
    #   = --no-build-index

    local index_mode="none"

    shift || true

    while [[ $# -gt 0 ]]; do

        case "$1" in

            --no-build-index)
                index_mode="none"
                ;;

            --build-index)
                index_mode="build"
                ;;

            --refresh-index)
                index_mode="refresh"
                ;;

            *)
                echo
                echo "ERROR: Unknown start option: $1"
                usage
                return 1
                ;;

        esac

        shift
    done

    echo
    echo "========================================"
    echo "       CROP GUARD FINAL MVP"
    echo "              START"
    echo "========================================"
    echo "Node:       $(hostname)"
    echo "Slurm ID:   $JOB_ID"
    echo "Host root:  $ROOT"
    echo "Container:  $CONTAINER"
    echo

    if ! validate_environment; then
        return 1
    fi

    # --------------------------------------------------------
    # Check dedicated ports
    # --------------------------------------------------------

    if ! port_free "$API_PORT"; then
        echo "ERROR: CropGuard API port $API_PORT is already occupied."
        echo "Run:"
        echo "  $0 clean"
        echo "Then:"
        echo "  $0 start"
        return 1
    fi

    if ! port_free "$UI_PORT"; then
        echo "ERROR: CropGuard UI port $UI_PORT is already occupied."
        echo "Run:"
        echo "  $0 clean"
        echo "Then:"
        echo "  $0 start"
        return 1
    fi

    # --------------------------------------------------------
    # Recommendation index
    # --------------------------------------------------------

    case "$index_mode" in

        none)
            echo "Recommendation index: EXISTING INDEX"
            echo "Index build skipped (--no-build-index)."
            echo
            ;;

        build)
            if ! build_recommendation_index "BUILD"; then
                echo
                echo "ERROR: Startup aborted because index build failed."
                return 1
            fi
            ;;

        refresh)
            if ! build_recommendation_index "REFRESH"; then
                echo
                echo "ERROR: Startup aborted because index refresh failed."
                return 1
            fi
            ;;

    esac

    # --------------------------------------------------------
    # FastAPI
    # --------------------------------------------------------

    echo "Starting FastAPI inside Apptainer..."


    nohup bash -c '
        apptainer exec --nv \
            --bind "$1:/workspace" \
            --pwd /workspace \
            "$2" \
            /opt/venv/bin/python -m uvicorn api.main:app \
            --host 0.0.0.0 \
            --port "$3"
    ' _ "$ROOT" "$CONTAINER" "$API_PORT" \
        > "$API_LOG" 2>&1 &

    API_PID_VALUE=$!

    echo "$API_PID_VALUE" > "$API_PID"

    echo "FastAPI launcher PID: $API_PID_VALUE"
    echo "Waiting for model/API startup..."

    READY=0

    for _ in $(seq 1 600); do

        if curl -fsS \
            "http://127.0.0.1:${API_PORT}/health" \
            >/dev/null 2>&1; then

            READY=1
            break
        fi

        if ! pid_alive "$API_PID_VALUE"; then
            break
        fi

        sleep 1
    done

    if [[ "$READY" != "1" ]]; then

        echo
        echo "ERROR: FastAPI did not become healthy."
        echo
        echo "Last FastAPI log:"
        tail -100 "$API_LOG"

        kill_owned_process "FastAPI" "$API_PID"

        echo
        echo "Run:"
        echo "  $0 clean"
        echo "Then:"
        echo "  $0 start"

        return 1
    fi

    echo "FastAPI: READY"
    curl -s "http://127.0.0.1:${API_PORT}/health"
    echo
    echo

    # --------------------------------------------------------
    # Streamlit
    # --------------------------------------------------------

    echo "Starting Streamlit inside Apptainer..."

    nohup bash -c '
        apptainer exec --nv \
            --bind "$1:/workspace" \
            --pwd /workspace \
            "$2" \
            /opt/venv/bin/python -m streamlit run \
            /workspace/ui/app.py \
            --server.port "$3" \
            --server.address 0.0.0.0 \
            --server.headless true \
            --server.enableCORS false
    ' _ "$ROOT" "$CONTAINER" "$UI_PORT" \
        > "$UI_LOG" 2>&1 &

    UI_PID_VALUE=$!

    echo "$UI_PID_VALUE" > "$UI_PID"

    echo "Streamlit launcher PID: $UI_PID_VALUE"

    READY=0

    for _ in $(seq 1 60); do

        if curl -fsS \
            "http://127.0.0.1:${UI_PORT}" \
            >/dev/null 2>&1; then

            READY=1
            break
        fi

        if ! pid_alive "$UI_PID_VALUE"; then
            break
        fi

        sleep 1
    done

    if [[ "$READY" != "1" ]]; then

        echo
        echo "ERROR: Streamlit did not become ready."
        echo
        echo "Last Streamlit log:"
        tail -100 "$UI_LOG"

        kill_owned_process "Streamlit" "$UI_PID"

        echo
        echo "Run:"
        echo "  $0 clean"
        echo "Then:"
        echo "  $0 start"

        return 1
    fi

    echo "Streamlit: READY"
    echo

    # --------------------------------------------------------
    # Cloudflare Named Tunnel
    # --------------------------------------------------------

    echo "Starting named Cloudflare Tunnel..."
    echo "Public hostname: https://${CF_PUBLIC_HOSTNAME}"

    nohup "$CF" tunnel run \
        --token-file "$CF_TOKEN_FILE" \
        > "$CF_LOG" 2>&1 &

    CF_PID_VALUE=$!

    echo "$CF_PID_VALUE" > "$CF_PID"

    echo "Cloudflared launcher PID: $CF_PID_VALUE"

    sleep 5

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    echo
    echo "Checking final services..."

    if ! curl -fsS \
        "http://127.0.0.1:${API_PORT}/health" \
        >/dev/null 2>&1; then

        echo "ERROR: FastAPI became unavailable after startup."

        kill_owned_process "Cloudflared" "$CF_PID"
        kill_owned_process "Streamlit" "$UI_PID"
        kill_owned_process "FastAPI" "$API_PID"

        echo
        echo "Run:"
        echo "  $0 clean"
        echo "Then:"
        echo "  $0 start"

        return 1
    fi

    if ! curl -fsS \
        "http://127.0.0.1:${UI_PORT}" \
        >/dev/null 2>&1; then

        echo "ERROR: Streamlit became unavailable after startup."

        kill_owned_process "Cloudflared" "$CF_PID"
        kill_owned_process "Streamlit" "$UI_PID"
        kill_owned_process "FastAPI" "$API_PID"

        echo
        echo "Run:"
        echo "  $0 clean"
        echo "Then:"
        echo "  $0 start"

        return 1
    fi

    echo
    echo "========================================"
    echo "        CROP GUARD IS READY"
    echo "========================================"
    echo "Node:       $(hostname)"
    echo "Slurm ID:   $JOB_ID"
    echo
    echo "FastAPI:    http://127.0.0.1:${API_PORT}"
    echo "Streamlit:  http://127.0.0.1:${UI_PORT}"
    echo
    echo "Health:"
    curl -s "http://127.0.0.1:${API_PORT}/health"
    echo
    echo
    echo "Cloudflare URL:"
    echo "https://${CF_PUBLIC_HOSTNAME}"
    echo
    echo "Cloudflare log:"
    echo "$CF_LOG"
    echo
    echo "PID files:"
    echo "  $API_PID"
    echo "  $UI_PID"
    echo "  $CF_PID"
    echo
    echo "Logs:"
    echo "  $API_LOG"
    echo "  $UI_LOG"
    echo "  $CF_LOG"
    echo
    echo "========================================"

    return 0
}

# ------------------------------------------------------------
# STOP
# ------------------------------------------------------------

stop() {

    echo
    echo "========================================"
    echo "       CROP GUARD FINAL MVP"
    echo "              STOP"
    echo "========================================"
    echo "Node:     $(hostname)"
    echo "Slurm ID: $JOB_ID"
    echo

    kill_owned_process "Cloudflared" "$CF_PID"
    kill_owned_process "Streamlit" "$UI_PID"
    kill_owned_process "FastAPI" "$API_PID"

    echo
    echo "CropGuard processes for Slurm job $JOB_ID stopped."
    echo "Port 8000 was NOT touched."
    echo "========================================"
}

# ------------------------------------------------------------
# CLEAN
# ------------------------------------------------------------

clean() {

    echo
    echo "========================================"
    echo "       CROP GUARD FINAL MVP"
    echo "              CLEAN"
    echo "========================================"
    echo "Node:     $(hostname)"
    echo "Slurm ID: $JOB_ID"
    echo

    echo "Stopping current-job recorded processes..."

    kill_owned_process "Cloudflared" "$CF_PID"
    kill_owned_process "Streamlit" "$UI_PID"
    kill_owned_process "FastAPI" "$API_PID"

    echo
    echo "Checking dedicated CropGuard ports..."

    for port in "$API_PORT" "$UI_PORT"; do

        pids="$(port_pids "$port")"

        if [[ -n "$pids" ]]; then

            echo "Port $port occupied by PID(s): $pids"

            for pid in $pids; do

                echo "Stopping PID $pid on CropGuard port $port..."

                kill "$pid" 2>/dev/null || true

            done

        else

            echo "Port $port is free."

        fi
    done

    rm -f "$API_PID" "$UI_PID" "$CF_PID"

    echo
    echo "Waiting for ports to clear..."

    sleep 3

    echo
    echo "Final port check:"

    for port in "$API_PORT" "$UI_PORT"; do

        if port_free "$port"; then
            echo "  OK: port $port is free."
        else
            echo "  WARNING: port $port is still occupied."
        fi

    done

    echo
    echo "========================================"
    echo "CLEAN COMPLETE"
    echo "========================================"
    echo
    echo "You may now run:"
    echo "  $0 start"
    echo
}

# ------------------------------------------------------------
# STATUS
# ------------------------------------------------------------

status() {

    echo
    echo "========================================"
    echo "       CROP GUARD STATUS"
    echo "========================================"
    echo "Node:       $(hostname)"
    echo "Slurm ID:   $JOB_ID"
    echo "Host root:  $ROOT"
    echo "Container:  $CONTAINER"
    echo

    echo "FastAPI:"

    if curl -fsS \
        "http://127.0.0.1:${API_PORT}/health" \
        2>/dev/null; then

        echo
        echo "  READY"

    else

        echo "  NOT RESPONDING"

    fi

    echo
    echo "Streamlit:"

    if curl -fsS \
        "http://127.0.0.1:${UI_PORT}" \
        >/dev/null 2>&1; then

        echo "  READY"

    else

        echo "  NOT RESPONDING"

    fi

    echo
    echo "Recorded PIDs:"

    for file in "$API_PID" "$UI_PID" "$CF_PID"; do

        if [[ -f "$file" ]]; then

            pid="$(cat "$file" 2>/dev/null || true)"

            if pid_alive "$pid"; then
                echo "  RUNNING  PID=$pid  $file"
            else
                echo "  STOPPED  PID=$pid  $file"
            fi

        else

            echo "  NONE     $file"

        fi

    done

    echo
    echo "Dedicated ports:"

    for port in "$API_PORT" "$UI_PORT"; do

        if port_free "$port"; then
            echo "  $port: FREE"
        else
            echo "  $port: OCCUPIED"
        fi

    done

    echo
    echo "Port 8000 is not managed by CropGuard."
    echo
    echo "========================================"
}

# ------------------------------------------------------------
# RESTART
# ------------------------------------------------------------

restart() {

    shift || true

    stop
    sleep 2
    start "start" "$@"
}

# ------------------------------------------------------------
# COMMAND
# ------------------------------------------------------------

case "${1:-}" in

    start)
        start "$@"
        ;;

    stop)
        stop
        ;;

    status)
        status
        ;;

    clean)
        clean
        ;;

    restart)
        restart "$@"
        ;;

    -h|--help|help)
        echo
        echo "CropGuard Final MVP"
        echo
        echo "Usage:"
        echo
        echo "  $0 start"
        echo "      Start using the existing recommendation index."
        echo "      DEFAULT = --no-build-index"
        echo
        echo "  $0 start --no-build-index"
        echo "      Reuse the existing persisted recommendation index."
        echo
        echo "  $0 start --build-index"
        echo "      Build the recommendation index before startup."
        echo
        echo "  $0 start --refresh-index"
        echo "      Refresh/rebuild the recommendation index before startup."
        echo
        echo "  $0 stop"
        echo "      Stop CropGuard services for the current Slurm job."
        echo
        echo "  $0 status"
        echo "      Show FastAPI, Streamlit, Cloudflare and port status."
        echo
        echo "  $0 clean"
        echo "      Stop CropGuard and clean ports 8002 and 8502."
        echo
        echo "  $0 restart"
        echo "      Restart using the existing recommendation index."
        echo "      DEFAULT = --no-build-index"
        echo
        echo "  $0 restart --no-build-index"
        echo "  $0 restart --build-index"
        echo "  $0 restart --refresh-index"
        echo
        echo "Cloudflare:"
        echo "  https://${CF_PUBLIC_HOSTNAME}"
        echo "  Token file: $CF_TOKEN_FILE"
        echo
        echo "Port 8000 is NEVER managed by CropGuard."
        echo
        ;;

    *)
        echo
        echo "Usage:"
        echo "  $0 start [--no-build-index|--build-index|--refresh-index]"
        echo "  $0 stop"
        echo "  $0 status"
        echo "  $0 clean"
        echo "  $0 restart [--no-build-index|--build-index|--refresh-index]"
        echo "  $0 --help"
        echo
        exit 1
        ;;

esac
