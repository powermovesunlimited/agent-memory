"""
Memory Gateway MCP Server — Full Spec Implementation
Spec: enterprise-memory-layer-v0.4.md (all phases)
Engine: Cognee 0.5.5
Transport: stdio (FastMCP)

Phase 0: memory_search, memory_store, memory_graph_query, memory_stats
Phase 1: memory_working_set_refresh, memory_session_search
Phase 2: memory_feedback, memory_compact
Phase 3: memory_audit
"""

import asyncio
import json
import logging
import os
import sqlite3
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env before any cognee imports
load_dotenv(Path(__file__).parent / ".env", override=True)

from mcp.server.fastmcp import FastMCP

import cognee
from cognee import SearchType

# Point Cognee data storage at project directory (not site-packages)
COGNEE_DATA = Path(__file__).parent / "cognee-data"
COGNEE_DATA.mkdir(exist_ok=True)
cognee.config.system_root_directory(str(COGNEE_DATA))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("memory-gateway")

# ---------------------------------------------------------------------------
# Database paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
AUDIT_DB = DATA_DIR / "audit.db"
SESSION_DB = DATA_DIR / "sessions.db"
WORKING_SET_DB = DATA_DIR / "working_sets.db"

# ---------------------------------------------------------------------------
# Audit log (SQLite, append-only)
# ---------------------------------------------------------------------------
def _get_audit_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(AUDIT_DB))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS audit_log (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            operation TEXT NOT NULL,
            details TEXT
        )"""
    )
    return conn


def _audit_log(agent_id: str, operation: str, details: dict | None = None):
    """Fire-and-forget audit entry."""
    try:
        conn = _get_audit_conn()
        conn.execute(
            "INSERT INTO audit_log (timestamp, agent_id, operation, details) VALUES (?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                agent_id,
                operation,
                json.dumps(details) if details else None,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning(f"Audit log write failed: {e}")


# ---------------------------------------------------------------------------
# Session store (SQLite - short-term memory + compaction events)
# ---------------------------------------------------------------------------
def _get_session_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(SESSION_DB))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS session_turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata TEXT
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS compaction_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            summary TEXT NOT NULL,
            data_class TEXT,
            facts_compacted INTEGER DEFAULT 0,
            dataset TEXT
        )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_session_turns_session ON session_turns(session_id, agent_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_compaction_session ON compaction_events(session_id, agent_id)"
    )
    return conn


# ---------------------------------------------------------------------------
# Working set store (SQLite - active context facts per agent)
# ---------------------------------------------------------------------------
def _get_ws_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(WORKING_SET_DB))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS working_set (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            fact_id TEXT NOT NULL UNIQUE,
            content TEXT NOT NULL,
            data_class TEXT,
            relevance_score REAL DEFAULT 0.5,
            last_accessed TEXT NOT NULL,
            is_anchor INTEGER DEFAULT 0,
            loaded_at TEXT NOT NULL
        )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ws_agent ON working_set(agent_id, session_id)"
    )
    return conn


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DATASET = "shared"
AGENT_DATASETS = {}  # reserved for future static overrides

# Serialize all cognee.add/cognify calls — KuzuDB only allows one writer at a time.
_COGNEE_WRITE_LOCK = asyncio.Lock()

# Skip cognify (graph construction) for high-frequency writes.
# Graph building uses KuzuDB which has an exclusive file lock — concurrent writes fail.
# Vector search via LanceDB works fine without cognify and is sufficient for agent recall.
# Set COGNEE_SKIP_COGNIFY=true to disable graph construction (not recommended — breaks search).
_SKIP_COGNIFY = os.getenv("COGNEE_SKIP_COGNIFY", "false").lower() != "false"

SEARCH_TYPE_MAP = {
    "graph_completion": SearchType.GRAPH_COMPLETION,
    "rag_completion": SearchType.RAG_COMPLETION,
    "chunks": SearchType.CHUNKS,
    "summaries": SearchType.SUMMARIES,
    "graph_summary_completion": SearchType.GRAPH_SUMMARY_COMPLETION,
}

# Working set defaults (from spec section 6.3)
MAX_WORKING_SET_SIZE = 15

# ---------------------------------------------------------------------------
# Agent dataset routing helpers
# ---------------------------------------------------------------------------
def _agent_dataset(agent_id: str) -> str:
    """Return the private dataset name for an agent. Falls back to 'shared'."""
    if not agent_id or agent_id in ("default", "shared"):
        return DEFAULT_DATASET
    return agent_id  # agent_id IS the dataset name (e.g. "artemis", "milo")


def _search_datasets(agent_id: str, include_shared: bool = True) -> list[str]:
    """Return the ordered list of datasets to search for an agent.
    Always searches the agent's private dataset first, then shared (if requested
    and the agent is not already scoped to shared).
    """
    private = _agent_dataset(agent_id)
    if private == DEFAULT_DATASET or not include_shared:
        return [private]
    return [private, DEFAULT_DATASET]
DEFAULT_EVICTION_THRESHOLD = 0.3
DRIFT_THRESHOLD = 0.65  # cosine distance threshold for topic shift

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_results(results) -> list[dict]:
    """Normalize cognee search results to dicts."""
    formatted = []
    if not results:
        return formatted
    if not hasattr(results, '__iter__'):
        return formatted
    for r in results:
        if isinstance(r, dict):
            formatted.append(r)
        elif isinstance(r, str):
            formatted.append({"content": r})
        else:
            d = {}
            for attr in ("id", "type", "text", "name", "description"):
                val = getattr(r, attr, None)
                if val is not None:
                    d[attr] = str(val)
            if not d.get("text") and not d.get("content"):
                d["content"] = str(r)[:2000]
            formatted.append(d)
    return formatted


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "memory-gateway",
    instructions="Enterprise Memory Layer MCP server powered by Cognee. Spec: enterprise-memory-layer-v0.4.md",
)


# ========================================================================
# PHASE 0: Core Tools
# ========================================================================

@mcp.tool()
async def memory_search(
    query: str,
    agent_id: str = "default",
    search_type: str = "chunks",
    top_k: int = 10,
    dataset: str = "",
) -> str:
    """Search the knowledge graph and vector store.

    Automatically searches the agent's private dataset + shared dataset.
    Pass dataset to override (e.g. dataset="shared" to search only shared).

    Args:
        query: Natural language search query
        agent_id: Agent identifier for scoping and audit
        search_type: One of: chunks, graph_completion, rag_completion, summaries, graph_summary_completion
        top_k: Max results to return (default 10)
        dataset: Override dataset (default: search agent's dataset + shared)

    Returns:
        JSON with search results array, count, and timing.
    """
    t0 = time.monotonic()
    datasets = [dataset] if dataset else _search_datasets(agent_id)
    log.info(f"memory_search | agent={agent_id} | datasets={datasets} | query={query!r} | type={search_type}")

    st = SEARCH_TYPE_MAP.get(search_type)
    if st is None:
        return json.dumps({
            "error": f"Unknown search_type '{search_type}'. Valid: {list(SEARCH_TYPE_MAP.keys())}",
            "results": [], "count": 0,
        })

    try:
        results = await cognee.search(
            query_text=query, query_type=st, datasets=datasets, top_k=top_k,
        )
        elapsed = time.monotonic() - t0
        formatted = _format_results(results)

        # Extract relevance scores for observability
        scores = [r.get("score") or r.get("feedback_weight") for r in formatted if r.get("score") or r.get("feedback_weight")]
        avg_score = round(sum(float(s) for s in scores) / len(scores), 3) if scores else None
        empty = len(formatted) == 0

        _audit_log(agent_id, "memory_search", {
            "query": query, "search_type": search_type,
            "datasets": datasets, "results_count": len(formatted),
            "elapsed_ms": round(elapsed * 1000),
            "avg_relevance_score": avg_score,
            "returned_empty": empty,
            "top_result_preview": (formatted[0].get("text") or formatted[0].get("content", ""))[:100] if formatted else None,
        })

        return json.dumps({
            "results": formatted, "count": len(formatted),
            "search_type": search_type, "datasets": datasets,
            "elapsed_ms": round(elapsed * 1000),
            "avg_relevance_score": avg_score,
            "returned_empty": empty,
        }, default=str)

    except Exception as e:
        log.error(f"memory_search error: {e}\n{traceback.format_exc()}")
        _audit_log(agent_id, "memory_search_error", {"query": query, "error": str(e)})
        return json.dumps({"error": str(e), "results": [], "count": 0})


@mcp.tool()
async def memory_store(
    content: str,
    agent_id: str = "default",
    data_class: str = "session.context",
    dataset: str = "",
    metadata: str = "{}",
) -> str:
    """Store content into the knowledge graph via Cognee's ECL pipeline.

    By default stores into the agent's private dataset. Pass dataset="shared"
    to store into the shared org-wide dataset.

    Args:
        content: Text content to store
        agent_id: Agent identifier for scoping and audit
        data_class: Memory taxonomy class (contact.identity, contact.preferences, contact.history, case.state, case.outcome, org.config, reasoning.pattern, session.context)
        dataset: Dataset to store into (default: agent's private dataset)
        metadata: Optional JSON string with additional metadata

    Returns:
        JSON with storage confirmation and timing.
    """
    t0 = time.monotonic()
    target_dataset = dataset if dataset else _agent_dataset(agent_id)
    log.info(f"memory_store | agent={agent_id} | class={data_class} | dataset={target_dataset} | len={len(content)}")

    tagged_content = f"[agent:{agent_id}] [class:{data_class}] {content}"

    try:
        async with _COGNEE_WRITE_LOCK:
            await cognee.add(tagged_content, dataset_name=target_dataset)
            if not _SKIP_COGNIFY:
                await cognee.cognify(datasets=[target_dataset])
        elapsed = time.monotonic() - t0

        _audit_log(agent_id, "memory_store", {
            "data_class": data_class, "dataset": target_dataset,
            "content_length": len(content), "elapsed_ms": round(elapsed * 1000),
        })

        return json.dumps({
            "status": "stored", "dataset": target_dataset, "data_class": data_class,
            "content_length": len(content), "elapsed_ms": round(elapsed * 1000),
        }, default=str)

    except Exception as e:
        log.error(f"memory_store error: {e}\n{traceback.format_exc()}")
        _audit_log(agent_id, "memory_store_error", {"error": str(e)})
        return json.dumps({"error": str(e), "status": "failed"})


@mcp.tool()
async def memory_graph_query(
    entity: str,
    agent_id: str = "default",
    dataset: str = "",
    depth: int = 1,
) -> str:
    """Traverse the knowledge graph from a specific entity.

    Args:
        entity: Entity name to query from (e.g. "Kevin Kurono", "Smart Support")
        agent_id: Agent identifier for scoping and audit
        dataset: Override dataset (default: search agent's dataset + shared)
        depth: Traversal depth (used as top_k multiplier, default 1)

    Returns:
        JSON with entity name, relationships array, count, and timing.
    """
    t0 = time.monotonic()
    datasets = [dataset] if dataset else _search_datasets(agent_id)
    log.info(f"memory_graph_query | agent={agent_id} | datasets={datasets} | entity={entity!r}")

    try:
        results = await cognee.search(
            query_text=f"What is connected to {entity}? What are all the relationships involving {entity}?",
            query_type=SearchType.GRAPH_COMPLETION,
            datasets=datasets, top_k=5 * depth,
        )
        elapsed = time.monotonic() - t0
        formatted = _format_results(results)

        _audit_log(agent_id, "memory_graph_query", {
            "entity": entity, "datasets": datasets,
            "results_count": len(formatted), "elapsed_ms": round(elapsed * 1000),
        })

        return json.dumps({
            "entity": entity, "relationships": formatted,
            "count": len(formatted), "elapsed_ms": round(elapsed * 1000),
        }, default=str)

    except Exception as e:
        log.error(f"memory_graph_query error: {e}\n{traceback.format_exc()}")
        _audit_log(agent_id, "memory_graph_query_error", {"entity": entity, "error": str(e)})
        return json.dumps({"error": str(e), "entity": entity, "relationships": [], "count": 0})


@mcp.tool()
async def memory_stats(agent_id: str = "default") -> str:
    """Get memory gateway health and statistics.

    Args:
        agent_id: Agent identifier for audit

    Returns:
        JSON with health status, engine info, storage backends, and counts.
    """
    log.info(f"memory_stats | agent={agent_id}")

    try:
        # Audit count
        audit_count = 0
        if AUDIT_DB.exists():
            conn = _get_audit_conn()
            row = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()
            audit_count = row[0] if row else 0
            conn.close()

        # Working set count
        ws_count = 0
        if WORKING_SET_DB.exists():
            conn = _get_ws_conn()
            row = conn.execute("SELECT COUNT(*) FROM working_set").fetchone()
            ws_count = row[0] if row else 0
            conn.close()

        # Session turn count
        session_count = 0
        if SESSION_DB.exists():
            conn = _get_session_conn()
            row = conn.execute("SELECT COUNT(*) FROM session_turns").fetchone()
            session_count = row[0] if row else 0
            conn.close()

        result = {
            "status": "healthy",
            "engine": "cognee",
            "engine_version": "0.5.5",
            "server_version": "0.4.0",
            "spec": "enterprise-memory-layer-v0.4.md",
            "phase": "full-dogfood",
            "llm_model": os.getenv("LLM_MODEL", "unknown"),
            "embedding_model": os.getenv("EMBEDDING_MODEL", "unknown"),
            "caching": os.getenv("CACHING", "false"),
            "cache_backend": os.getenv("CACHE_BACKEND", "none"),
            "counts": {
                "audit_entries": audit_count,
                "working_set_facts": ws_count,
                "session_turns": session_count,
            },
            "storage": {
                "vector": "lancedb (local)",
                "graph": "kuzu (local)",
                "relational": "sqlite",
                "session_cache": os.getenv("CACHE_BACKEND", "fs"),
            },
        }

        _audit_log(agent_id, "memory_stats", None)
        return json.dumps(result, default=str)

    except Exception as e:
        log.error(f"memory_stats error: {e}")
        return json.dumps({"status": "error", "error": str(e)})


@mcp.tool()
async def memory_metrics(
    agent_id: str = "",
    days: int = 7,
) -> str:
    """Get memory usage metrics and lookup quality stats.

    Shows search quality (empty result rate, avg relevance), store success/fail,
    sessions ingested, and per-agent breakdowns. Use this to monitor memory health.

    Args:
        agent_id: Filter to a specific agent (default: all agents)
        days: Lookback window in days (default: 7)

    Returns:
        JSON with search quality, store stats, and per-agent breakdown.
    """
    try:
        if not AUDIT_DB.exists():
            return json.dumps({"error": "No audit data yet"})

        conn = _get_audit_conn()
        since = f"datetime('now', '-{days} days')"
        where_agent = f" AND agent_id = '{agent_id}'" if agent_id else ""

        # Search stats
        search_rows = conn.execute(f"""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN json_extract(details, '$.returned_empty') = 1 THEN 1 ELSE 0 END) as empty_count,
                AVG(json_extract(details, '$.elapsed_ms')) as avg_elapsed_ms,
                AVG(json_extract(details, '$.results_count')) as avg_results,
                AVG(json_extract(details, '$.avg_relevance_score')) as avg_relevance
            FROM audit_log
            WHERE operation = 'memory_search'
            AND timestamp > {since}
            {where_agent}
        """).fetchone()

        search_errors = conn.execute(f"""
            SELECT COUNT(*) FROM audit_log
            WHERE operation = 'memory_search_error'
            AND timestamp > {since}
            {where_agent}
        """).fetchone()[0]

        # Store stats
        store_rows = conn.execute(f"""
            SELECT
                COUNT(*) as total,
                AVG(json_extract(details, '$.elapsed_ms')) as avg_elapsed_ms
            FROM audit_log
            WHERE operation = 'memory_store'
            AND timestamp > {since}
            {where_agent}
        """).fetchone()

        store_errors = conn.execute(f"""
            SELECT COUNT(*) FROM audit_log
            WHERE operation = 'memory_store_error'
            AND timestamp > {since}
            {where_agent}
        """).fetchone()[0]

        # Per-agent breakdown
        agent_rows = conn.execute(f"""
            SELECT
                agent_id,
                SUM(CASE WHEN operation = 'memory_search' THEN 1 ELSE 0 END) as searches,
                SUM(CASE WHEN operation = 'memory_search_error' THEN 1 ELSE 0 END) as search_errors,
                SUM(CASE WHEN operation = 'memory_store' THEN 1 ELSE 0 END) as stores,
                SUM(CASE WHEN operation = 'memory_store_error' THEN 1 ELSE 0 END) as store_errors,
                SUM(CASE WHEN operation = 'memory_search' AND json_extract(details, '$.returned_empty') = 1 THEN 1 ELSE 0 END) as empty_searches,
                MAX(timestamp) as last_activity
            FROM audit_log
            WHERE timestamp > {since}
            {where_agent}
            GROUP BY agent_id
            ORDER BY searches DESC
        """).fetchall()

        conn.close()

        # Recent empty searches (for alerting)
        conn2 = _get_audit_conn()
        empty_examples = conn2.execute(f"""
            SELECT agent_id, json_extract(details, '$.query') as query, timestamp
            FROM audit_log
            WHERE operation = 'memory_search'
            AND json_extract(details, '$.returned_empty') = 1
            AND timestamp > datetime('now', '-1 days')
            {where_agent}
            ORDER BY timestamp DESC LIMIT 5
        """).fetchall()
        conn2.close()

        result = {
            "period_days": days,
            "agent_filter": agent_id or "all",
            "searches": {
                "total": search_rows[0] or 0,
                "errors": search_errors,
                "empty_results": int(search_rows[1] or 0),
                "empty_rate_pct": round((search_rows[1] or 0) / max(search_rows[0] or 1, 1) * 100, 1),
                "avg_results_returned": round(search_rows[3] or 0, 1),
                "avg_elapsed_ms": round(search_rows[2] or 0),
                "avg_relevance_score": round(search_rows[4], 3) if search_rows[4] else None,
            },
            "stores": {
                "total": store_rows[0] or 0,
                "errors": store_errors,
                "avg_elapsed_ms": round(store_rows[1] or 0),
            },
            "recent_empty_searches": [
                {"agent": r[0], "query": r[1], "at": r[2][:19]}
                for r in empty_examples
            ],
            "by_agent": [
                {
                    "agent": r[0],
                    "searches": r[1],
                    "search_errors": r[2],
                    "stores": r[3],
                    "store_errors": r[4],
                    "empty_searches": r[5],
                    "last_activity": r[6][:19] if r[6] else None,
                }
                for r in agent_rows
                if r[0] not in ("default", "test", "smartsupport-test")
            ],
        }

        return json.dumps(result, default=str)

    except Exception as e:
        log.error(f"memory_metrics error: {e}\n{traceback.format_exc()}")
        return json.dumps({"error": str(e)})



@mcp.tool()
async def memory_working_set_refresh(
    topic_summary: str,
    agent_id: str = "default",
    session_id: str = "default",
    dataset: str = "",
    preserve_anchors: bool = True,
    max_facts: int = MAX_WORKING_SET_SIZE,
) -> str:
    """Refresh the working set with facts relevant to the current topic.

    Queries the agent's private dataset + shared for the most relevant facts.

    Args:
        topic_summary: Description of the current conversation topic/task
        agent_id: Agent identifier
        session_id: Session identifier for working set scoping
        dataset: Override dataset (default: search agent's dataset + shared)
        preserve_anchors: Keep anchor facts across refresh (default: true)
        max_facts: Maximum facts in working set (default: 15)

    Returns:
        JSON with loaded facts, evicted facts, and preserved anchors.
    """
    t0 = time.monotonic()
    datasets = [dataset] if dataset else _search_datasets(agent_id)
    log.info(f"memory_working_set_refresh | agent={agent_id} | session={session_id} | datasets={datasets} | topic={topic_summary!r}")

    try:
        conn = _get_ws_conn()
        now = _now_iso()

        # Get current working set
        current = conn.execute(
            "SELECT fact_id, content, data_class, relevance_score, is_anchor FROM working_set WHERE agent_id=? AND session_id=?",
            (agent_id, session_id),
        ).fetchall()

        # Separate anchors
        anchors = [(r[0], r[1], r[2], r[3]) for r in current if r[4] == 1] if preserve_anchors else []
        anchor_ids = {a[0] for a in anchors}

        # Query cognee for relevant facts
        results = await cognee.search(
            query_text=topic_summary,
            query_type=SearchType.CHUNKS,
            datasets=datasets,
            top_k=max_facts * 2,  # oversample then rank
        )

        formatted = _format_results(results)

        # Score and rank new facts
        new_facts = []
        for i, r in enumerate(formatted[:max_facts - len(anchors)]):
            fact_id = r.get("id", str(uuid.uuid4()))
            content = r.get("text", r.get("content", str(r)))
            relevance = 1.0 - (i / max(len(formatted), 1))  # position-based relevance
            new_facts.append({
                "fact_id": fact_id,
                "content": content,
                "data_class": r.get("data_class", "session.context"),
                "relevance_score": round(relevance, 3),
                "is_anchor": False,
            })

        # Evict non-anchor facts
        evicted = []
        for r in current:
            if r[0] not in anchor_ids:
                evicted.append({"fact_id": r[0], "content": r[1][:200]})

        conn.execute(
            "DELETE FROM working_set WHERE agent_id=? AND session_id=? AND is_anchor=0",
            (agent_id, session_id),
        )

        # Insert new facts
        for f in new_facts:
            conn.execute(
                """INSERT OR REPLACE INTO working_set
                   (agent_id, session_id, fact_id, content, data_class, relevance_score, last_accessed, is_anchor, loaded_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (agent_id, session_id, f["fact_id"], f["content"], f["data_class"],
                 f["relevance_score"], now, 0, now),
            )

        conn.commit()
        conn.close()

        elapsed = time.monotonic() - t0

        # Record compaction event for evicted facts
        if evicted:
            sconn = _get_session_conn()
            sconn.execute(
                "INSERT INTO compaction_events (session_id, agent_id, timestamp, summary, data_class, facts_compacted, dataset) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, agent_id, now,
                 f"Working set refresh for topic: {topic_summary[:200]}",
                 "session.context", len(evicted), datasets[0]),
            )
            sconn.commit()
            sconn.close()

        _audit_log(agent_id, "memory_working_set_refresh", {
            "session_id": session_id, "topic": topic_summary[:200],
            "datasets": datasets,
            "loaded": len(new_facts), "evicted": len(evicted),
            "anchors_preserved": len(anchors), "elapsed_ms": round(elapsed * 1000),
        })

        return json.dumps({
            "loaded": new_facts,
            "evicted": evicted,
            "anchors_preserved": [{"fact_id": a[0], "content": a[1][:200]} for a in anchors],
            "working_set_size": len(new_facts) + len(anchors),
            "elapsed_ms": round(elapsed * 1000),
        }, default=str)

    except Exception as e:
        log.error(f"memory_working_set_refresh error: {e}\n{traceback.format_exc()}")
        _audit_log(agent_id, "memory_working_set_refresh_error", {"error": str(e)})
        return json.dumps({"error": str(e), "loaded": [], "evicted": [], "anchors_preserved": []})


@mcp.tool()
async def memory_session_search(
    query: str,
    session_id: str = "default",
    agent_id: str = "default",
    include_compactions: bool = True,
    limit: int = 20,
) -> str:
    """Search within the current session's history and compaction events.

    This is the reach-back path: find what was in context earlier in the
    same conversation, even if it was evicted from the working set.

    Args:
        query: What to search for within the session
        session_id: Session to search
        agent_id: Agent identifier
        include_compactions: Also search compaction events (default: true)
        limit: Max results (default: 20)

    Returns:
        JSON with matching turns and compaction events.
    """
    t0 = time.monotonic()
    log.info(f"memory_session_search | agent={agent_id} | session={session_id} | query={query!r}")

    try:
        results = {"turns": [], "compactions": [], "working_set": []}

        # Search session turns
        conn = _get_session_conn()
        turns = conn.execute(
            """SELECT id, timestamp, role, content, metadata FROM session_turns
               WHERE session_id=? AND agent_id=? AND content LIKE ?
               ORDER BY id DESC LIMIT ?""",
            (session_id, agent_id, f"%{query}%", limit),
        ).fetchall()

        for t in turns:
            results["turns"].append({
                "id": t[0], "timestamp": t[1], "role": t[2],
                "content": t[3][:500], "source": "session_cache",
            })

        # Search compaction events
        if include_compactions:
            compactions = conn.execute(
                """SELECT id, timestamp, summary, data_class, facts_compacted FROM compaction_events
                   WHERE session_id=? AND agent_id=? AND summary LIKE ?
                   ORDER BY id DESC LIMIT ?""",
                (session_id, agent_id, f"%{query}%", limit),
            ).fetchall()

            for c in compactions:
                results["compactions"].append({
                    "id": c[0], "timestamp": c[1], "summary": c[2][:500],
                    "data_class": c[3], "facts_compacted": c[4], "source": "compaction",
                })

        conn.close()

        # Also include current working set items that match
        ws_conn = _get_ws_conn()
        ws_items = ws_conn.execute(
            """SELECT fact_id, content, data_class, relevance_score, is_anchor FROM working_set
               WHERE agent_id=? AND session_id=? AND content LIKE ?
               LIMIT ?""",
            (agent_id, session_id, f"%{query}%", limit),
        ).fetchall()

        for w in ws_items:
            results["working_set"].append({
                "fact_id": w[0], "content": w[1][:500], "data_class": w[2],
                "relevance_score": w[3], "is_anchor": bool(w[4]), "source": "working_set",
            })
        ws_conn.close()

        elapsed = time.monotonic() - t0
        total = len(results["turns"]) + len(results["compactions"]) + len(results["working_set"])

        _audit_log(agent_id, "memory_session_search", {
            "session_id": session_id, "query": query,
            "results_count": total, "elapsed_ms": round(elapsed * 1000),
        })

        return json.dumps({
            **results, "total_count": total,
            "elapsed_ms": round(elapsed * 1000),
        }, default=str)

    except Exception as e:
        log.error(f"memory_session_search error: {e}\n{traceback.format_exc()}")
        _audit_log(agent_id, "memory_session_search_error", {"query": query, "error": str(e)})
        return json.dumps({"error": str(e), "turns": [], "compactions": [], "working_set": []})


@mcp.tool()
async def memory_session_record(
    content: str,
    role: str = "user",
    session_id: str = "default",
    agent_id: str = "default",
    metadata: str = "{}",
) -> str:
    """Record a conversation turn in session memory (short-term).

    This feeds the session cache that memory_session_search queries.
    Agents should call this to log significant turns for reach-back.

    Args:
        content: The turn content to record
        role: Turn role (user, assistant, system)
        session_id: Session identifier
        agent_id: Agent identifier
        metadata: Optional JSON metadata

    Returns:
        JSON confirmation with turn id.
    """
    log.info(f"memory_session_record | agent={agent_id} | session={session_id} | role={role} | len={len(content)}")

    try:
        conn = _get_session_conn()
        cursor = conn.execute(
            "INSERT INTO session_turns (session_id, agent_id, timestamp, role, content, metadata) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, agent_id, _now_iso(), role, content, metadata),
        )
        turn_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return json.dumps({"status": "recorded", "turn_id": turn_id, "session_id": session_id})

    except Exception as e:
        log.error(f"memory_session_record error: {e}")
        return json.dumps({"error": str(e), "status": "failed"})


@mcp.tool()
async def memory_anchor_set(
    fact_id: str,
    agent_id: str = "default",
    session_id: str = "default",
    anchor: bool = True,
) -> str:
    """Set or clear the anchor flag on a working set fact.

    Anchor facts survive all working set refreshes until explicitly released.
    Use for critical context: client name, task objective, compliance constraints.

    Args:
        fact_id: The fact ID to anchor/unanchor
        agent_id: Agent identifier
        session_id: Session identifier
        anchor: True to anchor, False to release (default: true)

    Returns:
        JSON confirmation.
    """
    log.info(f"memory_anchor_set | agent={agent_id} | fact={fact_id} | anchor={anchor}")

    try:
        conn = _get_ws_conn()
        result = conn.execute(
            "UPDATE working_set SET is_anchor=? WHERE fact_id=? AND agent_id=? AND session_id=?",
            (1 if anchor else 0, fact_id, agent_id, session_id),
        )
        conn.commit()
        updated = result.rowcount
        conn.close()

        if updated == 0:
            return json.dumps({"status": "not_found", "fact_id": fact_id})

        _audit_log(agent_id, "memory_anchor_set", {"fact_id": fact_id, "anchor": anchor})
        return json.dumps({"status": "updated", "fact_id": fact_id, "is_anchor": anchor})

    except Exception as e:
        log.error(f"memory_anchor_set error: {e}")
        return json.dumps({"error": str(e), "status": "failed"})


@mcp.tool()
async def memory_working_set_get(
    agent_id: str = "default",
    session_id: str = "default",
) -> str:
    """Get the current working set for an agent's session.

    Returns all active context facts, their relevance scores, and anchor status.

    Args:
        agent_id: Agent identifier
        session_id: Session identifier

    Returns:
        JSON with facts array and working set size.
    """
    try:
        conn = _get_ws_conn()
        rows = conn.execute(
            """SELECT fact_id, content, data_class, relevance_score, is_anchor, last_accessed, loaded_at
               FROM working_set WHERE agent_id=? AND session_id=?
               ORDER BY is_anchor DESC, relevance_score DESC""",
            (agent_id, session_id),
        ).fetchall()
        conn.close()

        facts = []
        for r in rows:
            facts.append({
                "fact_id": r[0], "content": r[1], "data_class": r[2],
                "relevance_score": r[3], "is_anchor": bool(r[4]),
                "last_accessed": r[5], "loaded_at": r[6],
            })

        return json.dumps({"facts": facts, "size": len(facts)}, default=str)

    except Exception as e:
        log.error(f"memory_working_set_get error: {e}")
        return json.dumps({"error": str(e), "facts": [], "size": 0})


# ========================================================================
# PHASE 2: Feedback Loops + Compaction
# ========================================================================

@mcp.tool()
async def memory_feedback(
    task_summary: str,
    resolution_path: str,
    outcome: str,
    agent_id: str = "default",
    session_id: str = "default",
    dataset: str = "",
) -> str:
    """Submit task outcome feedback to strengthen the knowledge graph.

    Stores a reasoning.pattern into the agent's private dataset by default.
    Pass dataset="shared" to contribute a pattern to all agents.

    Args:
        task_summary: What the task was
        resolution_path: How it was resolved (steps, approach)
        outcome: Result (success, partial, failed)
        agent_id: Agent identifier
        session_id: Session identifier
        dataset: Dataset to store pattern in (default: agent's private dataset)

    Returns:
        JSON with acceptance confirmation.
    """
    t0 = time.monotonic()
    target_dataset = dataset if dataset else _agent_dataset(agent_id)
    log.info(f"memory_feedback | agent={agent_id} | dataset={target_dataset} | outcome={outcome} | task={task_summary[:80]!r}")

    try:
        pattern_content = (
            f"[agent:{agent_id}] [class:reasoning.pattern] "
            f"Task: {task_summary}\n"
            f"Resolution: {resolution_path}\n"
            f"Outcome: {outcome}\n"
            f"Session: {session_id}\n"
            f"Timestamp: {_now_iso()}"
        )

        async with _COGNEE_WRITE_LOCK:
            await cognee.add(pattern_content, dataset_name=target_dataset)
            if not _SKIP_COGNIFY:
                await cognee.cognify(datasets=[target_dataset])

        elapsed = time.monotonic() - t0

        _audit_log(agent_id, "memory_feedback", {
            "task_summary": task_summary[:200], "outcome": outcome,
            "session_id": session_id, "dataset": target_dataset,
            "elapsed_ms": round(elapsed * 1000),
        })

        return json.dumps({
            "status": "accepted",
            "outcome": outcome,
            "dataset": target_dataset,
            "data_class": "reasoning.pattern",
            "elapsed_ms": round(elapsed * 1000),
        }, default=str)

    except Exception as e:
        log.error(f"memory_feedback error: {e}\n{traceback.format_exc()}")
        _audit_log(agent_id, "memory_feedback_error", {"error": str(e)})
        return json.dumps({"error": str(e), "status": "failed"})


@mcp.tool()
async def memory_compact(
    agent_id: str = "default",
    session_id: str = "default",
    data_class: str = "",
    dataset: str = "",
) -> str:
    """Trigger taxonomy-aware compaction for a session or data class.

    Compacts session turns into durable long-term memory entries in the agent's
    private dataset.

    Args:
        agent_id: Agent identifier (empty = all agents)
        session_id: Session to compact (empty = all sessions)
        data_class: Specific data class to compact (empty = all)
        dataset: Target dataset for compacted entries (default: agent's private dataset)

    Returns:
        JSON with compaction stats: items processed, stored, discarded.
    """
    t0 = time.monotonic()
    target_dataset = dataset if dataset else _agent_dataset(agent_id)
    log.info(f"memory_compact | agent={agent_id} | session={session_id} | class={data_class} | dataset={target_dataset}")

    try:
        conn = _get_session_conn()

        # Get turns to compact
        query = "SELECT id, timestamp, role, content, metadata FROM session_turns WHERE 1=1"
        params: list = []
        if agent_id and agent_id != "default":
            query += " AND agent_id=?"
            params.append(agent_id)
        if session_id and session_id != "default":
            query += " AND session_id=?"
            params.append(session_id)

        turns = conn.execute(query + " ORDER BY id", params).fetchall()

        if not turns:
            conn.close()
            return json.dumps({
                "status": "nothing_to_compact",
                "compacted": 0, "archived": 0, "discarded": 0,
            })

        # Taxonomy-based processing
        # session.noise -> discard
        # session.context -> compact to summary, store as session.context
        # everything else -> preserve as-is
        compacted = 0
        discarded = 0
        archived = 0

        # Group turns into a summary
        turn_texts = []
        for t in turns:
            content = t[3]
            metadata = t[4] or "{}"

            # Check for noise markers
            if "[class:session.noise]" in content:
                discarded += 1
                continue

            turn_texts.append(f"[{t[1]}] ({t[2]}): {content}")

        if turn_texts:
            # Create compacted summary
            summary = "\n".join(turn_texts[:50])  # Cap at 50 turns per compaction
            if len(summary) > 100:
                # Store the compacted session to long-term memory
                tagged = f"[agent:{agent_id}] [class:session.context] Session compaction ({session_id}):\n{summary}"
                async with _COGNEE_WRITE_LOCK:
                    await cognee.add(tagged, dataset_name=target_dataset)
                    if not _SKIP_COGNIFY:
                        await cognee.cognify(datasets=[target_dataset])
                compacted = len(turn_texts)

                # Record compaction event
                conn.execute(
                    "INSERT INTO compaction_events (session_id, agent_id, timestamp, summary, data_class, facts_compacted, dataset) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (session_id, agent_id, _now_iso(),
                     f"Session compaction: {len(turn_texts)} turns",
                     "session.context", compacted, target_dataset),
                )
                archived = 1

        # Clean up compacted turns (keep last 5 for continuity)
        if len(turns) > 5:
            keep_after_id = turns[-5][0]
            conn.execute(
                "DELETE FROM session_turns WHERE id < ? AND session_id=? AND agent_id=?",
                (keep_after_id, session_id, agent_id) if session_id != "default" else (keep_after_id,),
            )

        conn.commit()
        conn.close()

        # Clear working set for this session (clean boundary per spec 6.2 Signal 3)
        ws_conn = _get_ws_conn()
        ws_conn.execute(
            "DELETE FROM working_set WHERE agent_id=? AND session_id=? AND is_anchor=0",
            (agent_id, session_id),
        )
        ws_conn.commit()
        ws_conn.close()

        elapsed = time.monotonic() - t0

        _audit_log(agent_id, "memory_compact", {
            "session_id": session_id, "compacted": compacted,
            "archived": archived, "discarded": discarded,
            "elapsed_ms": round(elapsed * 1000),
        })

        return json.dumps({
            "status": "compacted",
            "compacted": compacted,
            "archived": archived,
            "discarded": discarded,
            "flagged": 0,
            "elapsed_ms": round(elapsed * 1000),
        }, default=str)

    except Exception as e:
        log.error(f"memory_compact error: {e}\n{traceback.format_exc()}")
        _audit_log(agent_id, "memory_compact_error", {"error": str(e)})
        return json.dumps({"error": str(e), "status": "failed"})


# ========================================================================
# PHASE 3: Audit
# ========================================================================

@mcp.tool()
async def memory_audit(
    agent_id: str = "",
    operation: str = "",
    limit: int = 50,
    since: str = "",
) -> str:
    """Read the audit log. Compliance and owner roles only (enforced at gateway layer in enterprise).

    Args:
        agent_id: Filter by agent (empty = all agents)
        operation: Filter by operation type (empty = all)
        limit: Max entries to return (default: 50)
        since: ISO timestamp to filter from (empty = no time filter)

    Returns:
        JSON with audit entries array.
    """
    log.info(f"memory_audit | agent_filter={agent_id} | op_filter={operation} | limit={limit}")

    try:
        conn = _get_audit_conn()

        query = "SELECT event_id, timestamp, agent_id, operation, details FROM audit_log WHERE 1=1"
        params: list = []

        if agent_id:
            query += " AND agent_id=?"
            params.append(agent_id)
        if operation:
            query += " AND operation=?"
            params.append(operation)
        if since:
            query += " AND timestamp>=?"
            params.append(since)

        query += " ORDER BY event_id DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        conn.close()

        entries = []
        for r in rows:
            entries.append({
                "event_id": r[0],
                "timestamp": r[1],
                "agent_id": r[2],
                "operation": r[3],
                "details": json.loads(r[4]) if r[4] else None,
            })

        return json.dumps({"entries": entries, "count": len(entries)}, default=str)

    except Exception as e:
        log.error(f"memory_audit error: {e}")
        return json.dumps({"error": str(e), "entries": [], "count": 0})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio").lower()
    if transport == "http":
        host = os.getenv("MCP_HTTP_HOST", "127.0.0.1")
        port = int(os.getenv("MCP_HTTP_PORT", "8002"))
        log.info(f"Memory Gateway MCP server starting (streamable-http) on {host}:{port}")
        try:
            mcp.run(transport="streamable-http", host=host, port=port)
        except TypeError:
            # Older FastMCP: set via settings object
            mcp.settings.host = host
            mcp.settings.port = port
            mcp.run(transport="streamable-http")
    else:
        log.info("Memory Gateway MCP server starting (stdio) — full spec implementation")
        mcp.run(transport="stdio")
