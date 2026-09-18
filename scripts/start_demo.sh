#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace"
API_PORT=8002
UI_PORT=8502

LOG_DIR="$ROOT/logs"
RUN_DIR="$ROOT/run"

mkdir -p "$LOG_DIR" "$RUN_DIR"

echo "========================================"
echo "       CROP GUARD FINAL MVP"
echo "========================================"
echo "DGX Node: $(hostname)"
echo

# --------------------------------------------------
# FastAPI
# --------------------------------------------------

if curl -fsS "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then
    echo "FastAPI already running on :${API_PORT}"
else
    echo "Starting FastAPI..."

    nohup /opt/venv/bin/python -m uvicorn api.main:app \
        --host 0.0.0.0 \
        --port "$API_PORT" \
        > "$LOG_DIR/fastapi.log" 2>&1 &

    echo $! > "$RUN_DIR/fastapi.pid"

    echo "Waiting for FastAPI/model startup..."

    for i in $(seq 1 600); do
        if curl -fsS "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then
            break
        fi

        if [ "$i" -eq 600 ]; then
            echo "ERROR: FastAPI did not become healthy."
            tail -100 "$LOG_DIR/fastapi.log"
            exit 1
        fi

        sleep 1
    done

    echo "FastAPI: READY"
fi

# --------------------------------------------------
# Streamlit
# --------------------------------------------------

if curl -fsS "http://127.0.0.1:${UI_PORT}" >/dev/null 2>&1; then
    echo "Streamlit already running on :${UI_PORT}"
else
    echo "Starting Streamlit..."

    nohup /opt/venv/bin/python -m streamlit run \
        "$ROOT/ui/app.py" \
        --server.port "$UI_PORT" \
        --server.address 0.0.0.0 \
        --server.headless true \
        --server.enableCORS false \
        > "$LOG_DIR/streamlit.log" 2>&1 &

    echo $! > "$RUN_DIR/streamlit.pid"

    echo "Waiting for Streamlit..."

    for i in $(seq 1 60); do
        if curl -fsS "http://127.0.0.1:${UI_PORT}" >/dev/null 2>&1; then
            break
        fi

        if [ "$i" -eq 60 ]; then
            echo "ERROR: Streamlit did not start."
            tail -100 "$LOG_DIR/streamlit.log"
            exit 1
        fi

        sleep 1
    done

    echo "Streamlit: READY"
fi

# --------------------------------------------------
# Cloudflare
# --------------------------------------------------

CF="$HOME/bin/cloudflared"

if [ ! -x "$CF" ]; then
    echo "ERROR: $CF not found."
    exit 1
fi

if pgrep -f "cloudflared tunnel --url http://127.0.0.1:${UI_PORT}" >/dev/null 2>&1; then
    echo "Cloudflared already running."
else
    echo "Starting Cloudflare Quick Tunnel..."

    nohup "$CF" tunnel \
        --url "http://127.0.0.1:${UI_PORT}" \
        > "$LOG_DIR/cloudflared.log" 2>&1 &

    echo $! > "$RUN_DIR/cloudflared.pid"

    sleep 5
fi

echo
echo "========================================"
echo "        CROP GUARD IS READY"
echo "========================================"
echo
echo "DGX Node:"
hostname
echo
echo "FastAPI:"
echo "http://127.0.0.1:${API_PORT}"
echo
echo "Streamlit:"
echo "http://127.0.0.1:${UI_PORT}"
echo
echo "FastAPI health:"
curl -s "http://127.0.0.1:${API_PORT}/health"
echo
echo

echo "Cloudflare URL:"
grep -Eo 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' \
    "$LOG_DIR/cloudflared.log" | tail -1 || \
    echo "URL not available yet. Run: cat $LOG_DIR/cloudflared.log"

echo
echo "Logs:"
echo "  $LOG_DIR/fastapi.log"
echo "  $LOG_DIR/streamlit.log"
echo "  $LOG_DIR/cloudflared.log"
echo
echo "========================================"
