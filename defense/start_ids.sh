
#!/usr/bin/env bash
set -Eeuo pipefail

# Run from the repository root, regardless of the launch directory.
cd "$(dirname "$0")/.."
ROOT="$PWD"

INTERFACE="enp3s0"
FLOW_CSV="$ROOT/defense/flows/live_flows.csv"
ALERT_DIR="$ROOT/defense/alerts"
LOG_DIR="$ROOT/defense/logs"

HTTP_LOG="$LOG_DIR/http_server.log"
FLOW_LOG="$LOG_DIR/cicflowmeter.log"
IDS_LOG="$LOG_DIR/live_ids.log"

mkdir -p "$ROOT/defense/flows" "$ALERT_DIR" "$LOG_DIR"

# Check the required files.
for file in \
    "$ROOT/.venv/bin/cicflowmeter" \
    "$ROOT/.venv/bin/python" \
    "$ROOT/defense/live_ids.py"
do
    if [[ ! -f "$file" ]]; then
        echo "ERROR: Required file not found: $file"
        exit 1
    fi
done

# Stop rather than accidentally starting duplicate services.
if ss -ltn | grep -qE '(^|[[:space:]])[^[:space:]]*:9999[[:space:]]'; then
    echo "ERROR: Port 9999 is already in use."
    echo "Stop the existing HTTP server before running this script."
    exit 1
fi

echo "Checking packet-capture permissions..."
if ! sudo -n true; then
    echo "ERROR: Passwordless sudo is unavailable."
    exit 1
fi

# Start the receiver's HTTP test service.
"$ROOT/.venv/bin/python" -m http.server 9999 \
    --bind 0.0.0.0 >"$HTTP_LOG" 2>&1 &
HTTP_PID=$!

# Give the HTTP server time to start.
sleep 1
if ! kill -0 "$HTTP_PID" 2>/dev/null; then
    echo "ERROR: HTTP server failed to start. See $HTTP_LOG"
    exit 1
fi

# Start live packet capture and flow extraction.
rm -f "$FLOW_CSV"

sudo -n "$ROOT/.venv/bin/cicflowmeter" \
    -i "$INTERFACE" \
    -c "$FLOW_CSV" >"$FLOW_LOG" 2>&1 &
FLOW_PID=$!

# Wait for CICFlowMeter to create its output file.
echo "Waiting for CICFlowMeter to start..."
for i in {1..15}; do
    if [[ -f "$FLOW_CSV" ]]; then
        break
    fi
    if ! kill -0 "$FLOW_PID" 2>/dev/null; then
        echo "ERROR: CICFlowMeter exited. See $FLOW_LOG"
        kill "$HTTP_PID" 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

if [[ ! -f "$FLOW_CSV" ]]; then
    echo "ERROR: Flow CSV was not created. See $FLOW_LOG"
    kill "$HTTP_PID" "$FLOW_PID" 2>/dev/null || true
    exit 1
fi

# Start live model predictions and alert generation.
"$ROOT/.venv/bin/python" "$ROOT/defense/live_ids.py" \
    >"$IDS_LOG" 2>&1 &
IDS_PID=$!

cleanup() {
    echo
    echo "Stopping IDS processes..."

    kill "$IDS_PID" "$HTTP_PID" 2>/dev/null || true

    # Stop the specific privileged CICFlowMeter process.
    sudo pkill -TERM -f \
        "$ROOT/.venv/bin/cicflowmeter.*live_flows.csv" \
        2>/dev/null || true

    wait "$IDS_PID" "$HTTP_PID" 2>/dev/null || true
    echo "Stopped. Logs are in $LOG_DIR"
}

trap cleanup EXIT INT TERM

echo
echo "IDS is running."
echo "Interface:   $INTERFACE"
echo "Flow CSV:    $FLOW_CSV"
echo "Predictions: $ALERT_DIR/live_predictions.csv"
echo "Alerts:      $ALERT_DIR/live_alerts.csv"
echo
echo "Logs:"
echo "  $FLOW_LOG"
echo "  $IDS_LOG"
echo
echo "Press Ctrl+C to stop."
echo "Follow predictions with: tail -f $IDS_LOG"

# Keep this script alive while its services run.
while kill -0 "$HTTP_PID" 2>/dev/null &&
      kill -0 "$FLOW_PID" 2>/dev/null &&
      kill -0 "$IDS_PID" 2>/dev/null; do
    sleep 2
done

echo "ERROR: One of the IDS processes stopped unexpectedly."
