#!/usr/bin/env bash
# install.sh — One-command setup for agent-memory on a VPS
# Usage: ./install.sh [--openclaw-profile <profile>] [--agents-dir <path>]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENCLAW_PROFILE="${OPENCLAW_PROFILE:-alt}"
AGENTS_DIR="${AGENTS_DIR:-$HOME/.openclaw-${OPENCLAW_PROFILE}/agents}"
MCP_PORT="${MCP_PORT:-8002}"

echo "🧠 PowerMoves Agent Memory — Installer"
echo "======================================="
echo "Install dir:   $SCRIPT_DIR"
echo "Agents dir:    $AGENTS_DIR"
echo "MCP port:      $MCP_PORT"
echo ""

# Step 1: Python venv
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
  echo "📦 Creating Python virtual environment..."
  python3 -m venv "$SCRIPT_DIR/.venv"
fi

echo "📦 Installing dependencies..."
"$SCRIPT_DIR/.venv/bin/pip" install --quiet --upgrade pip
"$SCRIPT_DIR/.venv/bin/pip" install --quiet \
  "cognee==0.5.5" \
  "langchain-cognee>=0.1.0" \
  "mistralai>=1.9.10,<2.0.0" \
  "mcp>=1.0.0" \
  "fastmcp>=2.0.0" \
  "python-dotenv>=1.0.0" \
  "httpx>=0.27.0"

# Step 2: .env setup
if [ ! -f "$SCRIPT_DIR/.env" ]; then
  echo ""
  echo "⚙️  Setting up configuration..."
  cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
  echo "   Created .env from template."
  echo "   ⚠️  Edit $SCRIPT_DIR/.env with your API keys before continuing."
  echo ""
  read -p "Press Enter after editing .env to continue..."
fi

# Step 3: Data directory
mkdir -p "$SCRIPT_DIR/data"
mkdir -p "$SCRIPT_DIR/cognee-data"

# Step 4: mcporter registration
MCPORTER_CONFIG="$HOME/.mcporter/mcporter.json"
if [ -f "$MCPORTER_CONFIG" ]; then
  echo "🔌 Updating mcporter config to use HTTP endpoint..."
  python3 - << PYEOF
import json
from pathlib import Path

cfg_path = Path("$MCPORTER_CONFIG")
cfg = json.loads(cfg_path.read_text())
cfg.setdefault("mcpServers", {})
cfg["mcpServers"]["memory-gateway"] = {"url": "http://127.0.0.1:${MCP_PORT}/mcp/"}
cfg_path.write_text(json.dumps(cfg, indent=2))
print("   ✅ mcporter updated")
PYEOF
else
  echo "   ⚠️  mcporter not found at $MCPORTER_CONFIG — skipping"
fi

# Step 5: systemd service
SERVICE_FILE="/etc/systemd/system/memory-gateway.service"
if command -v systemctl &>/dev/null; then
  echo "🚀 Installing systemd service..."
  VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"
  SERVER_PY="$SCRIPT_DIR/server.py"
  
  sudo tee "$SERVICE_FILE" > /dev/null << EOF
[Unit]
Description=Memory Gateway MCP Server (Cognee)
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$SCRIPT_DIR
ExecStart=$VENV_PYTHON $SERVER_PY
EnvironmentFile=$SCRIPT_DIR/.env
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
  
  sudo systemctl daemon-reload
  sudo systemctl enable memory-gateway
  sudo systemctl restart memory-gateway
  sleep 3
  
  if systemctl is-active --quiet memory-gateway; then
    echo "   ✅ memory-gateway service running"
  else
    echo "   ❌ Service failed to start — check: journalctl -u memory-gateway -n 20"
    exit 1
  fi
else
  echo "   ⚠️  systemd not available — start manually: .venv/bin/python server.py"
fi

# Step 6: BOOT.md for each agent
if [ -d "$AGENTS_DIR" ]; then
  echo "📝 Installing BOOT.md for each agent..."
  for AGENT_DIR in "$AGENTS_DIR"/*/; do
    AGENT_NAME=$(basename "$AGENT_DIR")
    WORKSPACE=$(python3 -c "
import json, sys
try:
  cfg = json.load(open('$HOME/.openclaw-${OPENCLAW_PROFILE}/openclaw.json'))
  agents = cfg.get('agents', {}).get('list', [])
  for a in agents:
    if a.get('id') == '$AGENT_NAME':
      print(a.get('workspace', ''))
      sys.exit(0)
except: pass
print('')
" 2>/dev/null)
    
    if [ -n "$WORKSPACE" ] && [ -d "$WORKSPACE" ] && [ ! -f "$WORKSPACE/BOOT.md" ]; then
      sed "s/{{AGENT_ID}}/$AGENT_NAME/g" "$SCRIPT_DIR/templates/BOOT.md.template" > "$WORKSPACE/BOOT.md"
      echo "   ✅ $AGENT_NAME → $WORKSPACE/BOOT.md"
    fi
  done
fi

# Step 7: Install ingest cron
echo "⏰ Installing session ingest cron (every 30 min)..."
CRON_CMD="*/30 * * * * cd $SCRIPT_DIR && $SCRIPT_DIR/.venv/bin/python ingest-sessions.py >> $SCRIPT_DIR/data/ingest.log 2>&1"
( crontab -l 2>/dev/null | grep -v "ingest-sessions"; echo "$CRON_CMD" ) | crontab -
echo "   ✅ Cron installed"

# Step 8: First-time backfill
echo ""
echo "🔄 Running initial session backfill..."
echo "   (This ingests all existing sessions — may take a while)"
"$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/ingest-sessions.py" &
BACKFILL_PID=$!
echo "   Backfill running in background (PID $BACKFILL_PID)"
echo "   Monitor: tail -f $SCRIPT_DIR/data/ingest.log"

echo ""
echo "✅ Installation complete!"
echo ""
echo "Quick check:"
echo "  curl http://127.0.0.1:${MCP_PORT}/api/memory/health"
echo "  mcporter list | grep memory-gateway"
echo ""
echo "Docs: ./docs/deployment.md"
