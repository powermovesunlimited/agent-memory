#!/usr/bin/env python3
"""
Automatic session transcript ingester for Cognee memory gateway.

Scans OpenClaw agent session JSONL files, extracts conversation turns,
pushes raw sessions AND extracted structured facts to the memory-gateway.

Runs as a cron job — only ingests sessions modified since last run.
State tracked in: data/ingest-watermark.json

Usage:
  Normal (incremental):  python ingest-sessions.py
  Full backfill:         python ingest-sessions.py --full
  Skip fact extraction:  python ingest-sessions.py --no-extract
"""

import asyncio
import json
import logging
import os
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
SKIP_PREFIXES = (
    "[cron:", "HEARTBEAT_OK", "Follow BOOT.md", "NO_REPLY",
    "You are running a boot check", "Read HEARTBEAT.md",
    "Pre-compaction memory flush",
)

# Fact extraction config
EXTRACT_MODEL = os.getenv("LLM_MODEL", "azure/gpt-4o-mini")
EXTRACT_API_KEY = os.getenv("LLM_API_KEY", "")
EXTRACT_ENDPOINT = os.getenv("LLM_ENDPOINT", "")
EXTRACT_API_VERSION = os.getenv("LLM_API_VERSION", "")
EXTRACT_MAX_TURNS = 20  # Only send last N turns to extraction (cost control)

EXTRACTION_PROMPT = """You are a memory extraction assistant. Extract durable facts from this conversation that would be useful to remember in future sessions.

For each fact, return a JSON object with:
- "fact": the fact as a concise statement (max 100 chars)
- "category": one of: preference | decision | goal | context | identity | skill | project
- "data_class": one of: contact.preferences | contact.identity | org.config | case.state | reasoning.pattern | session.context
- "confidence": one of: high | medium | low

Rules:
- Only extract facts that are DURABLE (stable across sessions, not just this moment)
- Skip small talk, greetings, procedural steps, and one-off task outputs
- Skip anything already obvious from context
- Return 0-5 facts max. Quality over quantity.
- Return a JSON array only, no other text. If nothing worth extracting, return []

Conversation:
"""

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
            if not text or len(text) < 10:
                continue
            if any(text.startswith(skip) for skip in SKIP_PREFIXES):
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


async def extract_facts(turns: list[dict]) -> list[dict]:
    """Run LLM fact extraction on a session's turns. Returns list of fact dicts."""
    if not EXTRACT_API_KEY:
        return []

    # Use last N turns only (cost control)
    sample = turns[-EXTRACT_MAX_TURNS:]
    convo_text = "\n".join(f"{t['role']}: {t['content'][:300]}" for t in sample)

    try:
        import httpx

        # Build request for Azure OpenAI or standard OpenAI
        if EXTRACT_ENDPOINT and "azure" in EXTRACT_MODEL:
            # Azure OpenAI
            url = f"{EXTRACT_ENDPOINT}/chat/completions?api-version={EXTRACT_API_VERSION}"
            headers = {"api-key": EXTRACT_API_KEY, "Content-Type": "application/json"}
            model_name = EXTRACT_MODEL.replace("azure/", "")
        else:
            # Standard OpenAI
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {EXTRACT_API_KEY}", "Content-Type": "application/json"}
            model_name = EXTRACT_MODEL.replace("azure/", "")

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": convo_text},
            ],
            "temperature": 0.1,
            "max_tokens": 500,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                log.warning(f"Fact extraction API error: {resp.status_code}")
                return []

            data = resp.json()
            raw = data["choices"][0]["message"]["content"].strip()

            # Parse JSON response
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            facts = json.loads(raw)
            if not isinstance(facts, list):
                return []
            return [f for f in facts if isinstance(f, dict) and "fact" in f]

    except Exception as e:
        log.warning(f"Fact extraction failed: {e}")
        return []


async def run(full: bool = False, do_extract: bool = True):
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
            to_ingest.append((agent_name, cognee_agent_id, jsonl_file.name[:24], summary, turns))

        if latest_mtime > last_mtime:
            new_watermark[agent_name] = latest_mtime

    if not to_ingest:
        log.info("Nothing new to ingest")
        save_watermark(new_watermark)
        return

    log.info(f"Ingesting {len(to_ingest)} sessions (fact_extraction={'on' if do_extract else 'off'})...")
    ok_count = 0
    fail_count = 0
    facts_extracted = 0

    async with streamablehttp_client(GATEWAY_URL) as (read, write, _):
        async with ClientSession(read, write) as sess:
            await sess.initialize()

            for agent_name, agent_id, fname, summary, turns in to_ingest:
                # 1. Store raw session summary
                try:
                    result = await sess.call_tool("memory_store", {
                        "agent_id": agent_id,
                        "data_class": "contact.history",
                        "content": summary,
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
                    continue

                # 2. Extract and store structured facts
                if do_extract and EXTRACT_API_KEY:
                    facts = await extract_facts(turns)
                    for fact in facts:
                        fact_text = (
                            f"[agent:{agent_id}] [extracted_fact] "
                            f"[category:{fact.get('category','unknown')}] "
                            f"[confidence:{fact.get('confidence','medium')}] "
                            f"{fact['fact']}"
                        )
                        data_class = fact.get("data_class", "session.context")
                        try:
                            result = await sess.call_tool("memory_store", {
                                "agent_id": agent_id,
                                "data_class": data_class,
                                "content": fact_text,
                            })
                            r = json.loads(result.content[0].text) if result.content else {}
                            if r.get("status") in ("ok", "stored"):
                                log.info(f"    💡 fact: {fact['fact'][:60]} [{data_class}]")
                                facts_extracted += 1
                        except Exception as exc:
                            log.warning(f"    ⚠️  fact store failed: {exc}")

    save_watermark(new_watermark)
    log.info(f"Done — sessions ok={ok_count} fail={fail_count} | facts extracted={facts_extracted}")


if __name__ == "__main__":
    full = "--full" in sys.argv
    do_extract = "--no-extract" not in sys.argv
    if full:
        log.info("Full backfill mode — ignoring watermark")
    if not do_extract:
        log.info("Fact extraction disabled")
    log.info("=== Session ingester starting ===")
    asyncio.run(run(full=full, do_extract=do_extract))
