#!/bin/bash

# Configuration
SERVER_BIN="./bin/AgilityForgeServer"
CLIENT_APP="./bin/AgilityForgeGame.app"

# Force unbuffered output
export PYTHONUNBUFFERED=1

# Check binaries
if [ ! -f "$SERVER_BIN" ]; then
    echo "❌ Error: Server binary not found at $SERVER_BIN"
    exit 1
fi

echo "🚀 Starting AgilityForge Server..."

# create temp log
LOG_FILE=$(mktemp)

# Start server in background, writing to file
"$SERVER_BIN" > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

# Trap Ctrl+C to clean up
cleanup() {
    echo ""
    echo "🛑 Shutting down..."
    kill $SERVER_PID 2>/dev/null
    rm "$LOG_FILE"
    exit 0
}
trap cleanup SIGINT

# Show logs in background
tail -f "$LOG_FILE" &
TAIL_PID=$!

echo "⏳ Waiting for server port (8765) to open..."

# Wait for port 8765 to become active (checking every 1 second)
# Loop indefinitely until the server is ready (compilation can take time)
while ! nc -z localhost 8765; do
    sleep 1
done

echo ""
echo "✅ Server is ready (Port 8765 is open)!"
echo "🎮 Launching AgilityForge Game..."

# Kill the tail process so logs don't clutter the game output (optional, but cleaner)
# Actually, user might want to see runtime logs. Let's keep it running.

open "$CLIENT_APP"

echo "Press Ctrl+C to stop the server."
wait $SERVER_PID
