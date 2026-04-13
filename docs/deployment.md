# Deployment Guide

## Requirements

- Ubuntu 20.04+ or similar Linux
- Python 3.11+
- OpenClaw alt gateway already installed
- mcporter CLI installed (`npm install -g mcporter`)
- An LLM API key (Azure OpenAI, OpenAI, or Anthropic)

## One-Command Install

```bash
git clone https://github.com/powermovesunlimited/agent-memory
cd agent-memory
chmod +x install.sh
./install.sh
```

The installer:
1. Creates Python venv and installs dependencies
2. Sets up `.env` from template (prompts for API keys)
3. Registers the MCP server with mcporter (HTTP mode)
4. Installs systemd service (auto-starts on boot)
5. Drops BOOT.md into each agent workspace
6. Installs the 30-minute session ingest cron
7. Runs initial backfill of existing session transcripts

## Manual Setup

### 1. Python environment

```bash
python3 -m venv .venv
.venv/bin/pip install cognee==0.5.5 langchain-cognee "mistralai>=1.9.10,<2.0.0" mcp fastmcp python-dotenv httpx
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your LLM API keys
```

### 3. Start as systemd service

```bash
sudo cp systemd/memory-gateway.service /etc/systemd/system/
# Edit the service file to set User= and WorkingDirectory=
sudo systemctl daemon-reload
sudo systemctl enable memory-gateway
sudo systemctl start memory-gateway
```

### 4. Register with mcporter

Add to `~/.mcporter/mcporter.json`:
```json
{
  "mcpServers": {
    "memory-gateway": {
      "url": "http://127.0.0.1:8002/mcp/"
    }
  }
}
```

### 5. Session ingestion cron

```bash
crontab -e
# Add:
*/30 * * * * cd /path/to/agent-memory && .venv/bin/python ingest-sessions.py >> data/ingest.log 2>&1
```

## Verifying the Installation

```bash
# Check service is running
sudo systemctl status memory-gateway

# Check MCP tools are accessible
mcporter list | grep memory-gateway

# Test a write
mcporter call memory-gateway.memory_store agent_id="test" data_class="session.context" content="Test entry"

# Test a read
mcporter call memory-gateway.memory_search agent_id="test" query="test entry" search_type="chunks"

# Check audit log
mcporter call memory-gateway.memory_stats agent_id="test"
```

## Storage Configuration

### Starter (default): Local LanceDB + Kuzu

No additional setup needed. Data stored in `cognee-data/` directory.

### Standard: PostgreSQL + pgvector

```bash
# In .env:
DATABASE_URL=postgresql://user:pass@host:5432/dbname
ENABLE_BACKEND_ACCESS_CONTROL=false

# In PostgreSQL:
CREATE EXTENSION IF NOT EXISTS vector;
```

### Enterprise: Add FalkorDB for graph

```bash
# Run FalkorDB
docker run -p 6379:6379 falkordb/falkordb:latest

# In .env:
GRAPH_DATABASE_PROVIDER=falkordb
FALKORDB_HOST=localhost
FALKORDB_PORT=6379
```

## Troubleshooting

### Service won't start
```bash
journalctl -u memory-gateway -n 50
```

### Port already in use
```bash
fuser -k 8002/tcp
sudo systemctl restart memory-gateway
```

### mcporter timeout on memory_store
Cognee ingestion can take 20-30s per write. This is normal.
For faster writes, set `COGNEE_SKIP_COGNIFY=true` in `.env` (disables graph construction, keeps vector search).

### Lock errors on concurrent writes
Ensure `SHARED_KUZU_LOCK=true` and `CACHING=true` are set in `.env`.
