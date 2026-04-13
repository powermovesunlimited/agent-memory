#!/usr/bin/env python3
"""
Automatic session transcript ingester for Cognee memory gateway.

Scans OpenClaw agent session JSONL files, extracts conversation turns,
and pushes them to the memory-gateway MCP server via streamable HTTP.

Runs as a cron job — only ingests sessions modified since last run.
State tracked in: data/ingest-watermark.json

Usage:
  Normal (incremental): python ingest-sessions.py
  Full backfill:        python ingest-sessions.py --full
"""

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=True)

from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

# --- Config ---
AGENTS_DIR = Path("/home/oem/.openclaw-alt/agents")
GATEWAY_URL = "http://127.0.0.1:8002/mcp"
WATERMARK_FILE = Path(__file__).parent / "data" / "ingest-watermark.json"
AGENT_ID_MAP = {
    "main": "milo",
    "campaigns": "campaign-milo",
}
MIN_MESSAGES = 3
MAX_CHARS = 3000
# Skip these in message text (heartbeats, boot checks, cron invocations)
SKIP_PREFIXES = ("[cron:", "HEARTBEAT_OK", "Follow BOOT.md", "NO_REPLY", "You are running a boot check")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("ingest-sessions")


def load_watermark() -> dict:
    if WATERMARK_FILE.exists():
        try:
            return json.loads(WATERMARK_FILE.read_text())
        except Exception:
            pass
    return {}


def save_watermark(wm: dict):
    WATERMARK_FILE.parent.mkdir(parents=True, exist_ok=True)
    WATERMARK_FILE.write_text(json.dumps(wm, indent=2))


def extract_turns(jsonl_path: Path) -> list[dict]:
    """Extract meaningful user/assistant turns from an OpenClaw v3 session JSONL."""
    turns = []
    try:
        for line in jsonl_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue

            # OpenClaw v3: type="message", message.role = user|assistant
            if e.get("type") != "message":
                continue
            msg = e.get("message", {})
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            if role not in ("user", "assistant"):
                continue

            content = msg.get("content", [])
            if isinstance(content, list):
                text = " ".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ).strip()
            elif isinstance(content, str):
                text = content.strip()
            else:
                text = ""

            if not text:
                continue

            # Skip noise: heartbeats, boot checks, cron invocations
            if any(text.startswith(skip) for skip in SKIP_PREFIXES):
                continue
            if len(text) < 10:
                continue

            turns.append({"role": role, "content": text})
    except Exception as exc:
        log.warning(f"Parse error {jsonl_path.name}: {exc}")
    return turns


def format_summary(agent_id: str, session_id: str, turns: list[dict], mtime: float) -> str:
    ts = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"[agent:{agent_id}] [session:{session_id[:8]}] [{ts}]", ""]
    total = len(lines[0])
    for turn in turns:
        line = f"{turn['role']}: {turn['content'][:400]}"
        if total + len(line) > MAX_CHARS:
            lines.append("...(truncated)")
            break
        lines.append(line)
        total += len(line)
    return "\n".join(lines)


async def run(full: bool = False):
    watermark = {} if full else load_watermark()
    new_watermark = dict(watermark)

    to_ingest = []
    for agent_dir in sorted(AGENTS_DIR.iterdir()):
        if not agent_dir.is_dir():
            continue
        agent_name = agent_dir.name
        cognee_agent_id = AGENT_ID_MAP.get(agent_name, agent_name)
        sessions_dir = agent_dir / "sessions"
        if not sessions_dir.exists():
            continue

        last_mtime = watermark.get(agent_name, 0.0)
        latest_mtime = last_mtime

        for jsonl_file in sorted(sessions_dir.glob("*.jsonl"), key=lambda f: f.stat().st_mtime):
            if "active-memory" in str(jsonl_file.parent):
                continue
            try:
                mtime = jsonl_file.stat().st_mtime
            except OSError:
                continue

            latest_mtime = max(latest_mtime, mtime)

            if not full and mtime <= last_mtime:
                continue

            turns = extract_turns(jsonl_file)
            if len(turns) < MIN_MESSAGES:
                continue

            summary = format_summary(cognee_agent_id, jsonl_file.stem, turns, mtime)
            to_ingest.append((agent_name, cognee_agent_id, jsonl_file.name[:24], summary))

        # Update watermark to latest seen mtime regardless of ingestion
        if latest_mtime > last_mtime:
            new_watermark[agent_name] = latest_mtime

    if not to_ingest:
        log.info("Nothing new to ingest")
        save_watermark(new_watermark)
        return

    log.info(f"Ingesting {len(to_ingest)} sessions...")
    ok_count = 0
    fail_count = 0

    async with streamablehttp_client(GATEWAY_URL) as (read, write, _):
        async with ClientSession(read, write) as sess:
            await sess.initialize()
            for agent_name, agent_id, fname, content in to_ingest:
                try:
                    result = await sess.call_tool("memory_store", {
                        "agent_id": agent_id,
                        "data_class": "contact.history",
                        "content": content,
                    })
                    r = json.loads(result.content[0].text) if result.content else {}
                    if r.get("status") in ("ok", "stored"):
                        log.info(f"  ✅ {agent_name}/{fname} → {agent_id}")
                        ok_count += 1
                    else:
                        log.warning(f"  ⚠️  {agent_name}/{fname}: {r}")
                        fail_count += 1
                except Exception as exc:
                    log.warning(f"  ❌ {agent_name}/{fname}: {exc}")
                    fail_count += 1

    save_watermark(new_watermark)
    log.info(f"Done — ok={ok_count} fail={fail_count}")


if __name__ == "__main__":
    full = "--full" in sys.argv
    if full:
        log.info("Full backfill mode — ignoring watermark")
    log.info("=== Session ingester starting ===")
    asyncio.run(run(full=full))
