# Memory Gateway — Agent Memory Layer

**Enterprise-grade persistent memory for OpenClaw agents.** Powered by Cognee (vector + graph search), deployed as a local MCP server that runs alongside your OpenClaw gateway.

## What it does

Every conversation your agents have gets automatically remembered and recalled — without the agents needing to do anything. Agents wake up each session with full context from previous conversations, decisions made, and user preferences.

```
User talks to agent
       ↓
Active Memory (OpenClaw built-in) — searches recent memory files
       ↓
memory-gateway (this) — deep recall: weeks/months of conversation history
       ↓
Auto-ingestion cron — new sessions ingested every 30 minutes, automatically
```

## Architecture

- **MCP Server** (`server.py`) — 12 memory tools exposed via streamable HTTP
- **Session Ingester** (`ingest-sessions.py`) — cron job that ingests OpenClaw JSONL sessions into Cognee
- **Cognee** — vector (LanceDB/pgvector) + graph (Kuzu/FalkorDB) hybrid storage
- **Per-agent isolation** — each agent gets its own dataset, no cross-contamination

## Quick Start

```bash
git clone https://github.com/powermovesunlimited/agent-memory
cd agent-memory
cp .env.example .env
# Edit .env with your LLM API keys
./install.sh
```

## Storage Tiers

| Tier | Vector Store | Graph Store | Best for |
|------|-------------|-------------|----------|
| Starter | LanceDB (local file) | Kuzu (local file) | Dev, single agent |
| Standard | PostgreSQL + pgvector | Kuzu | Multi-agent, production |
| Enterprise | PostgreSQL + pgvector | FalkorDB | High concurrency, scale |

## Memory Tools (MCP)

| Tool | Description |
|------|-------------|
| `memory_search` | Semantic search across agent's history |
| `memory_store` | Store a durable fact |
| `memory_session_record` | Log a conversation turn |
| `memory_working_set_refresh` | Load context for current topic |
| `memory_session_search` | Search within current session |
| `memory_compact` | Flush session to long-term storage |
| `memory_feedback` | Report recall quality |
| `memory_anchor_set` | Pin critical facts |
| `memory_graph_query` | Direct graph queries |
| `memory_stats` | Engine health and counts |
| `memory_audit` | Audit log access |

## Requirements

- Python 3.11+
- An LLM API key (Azure OpenAI, OpenAI, or Anthropic)
- mcporter (for MCP tool access from agents)
- OpenClaw alt gateway

## Deployment

See [docs/deployment.md](docs/deployment.md) for VPS setup guide.

## Status

Currently deployed on PowerMoves internal infrastructure, powering 19 agents across all OpenClaw workspaces.

## License

Private — PowerMoves Unlimited
