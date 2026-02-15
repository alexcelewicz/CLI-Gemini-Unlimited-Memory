"""
Gemini File Search Memory MCP Server
=====================================
A persistent project memory layer using Google Gemini File Search stores.
Gives CLI coding agents (Claude Code, Gemini CLI, etc.) cross-session memory
by saving session summaries to per-project RAG stores and retrieving relevant
context at the start of each new session.

Architecture:
  - Each project gets its own Gemini File Search store
  - Session logs are uploaded as timestamped documents
  - Context retrieval uses Gemini Flash to semantically search the store
  - Retrieved context is injected into the agent's working context

Author: Built for Aleksander Celewicz
"""

import json
import os
import time as time_module
from datetime import datetime, timezone
from typing import Optional, List

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, ConfigDict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
RETRIEVAL_MODEL = os.environ.get("GEMINI_RETRIEVAL_MODEL", "gemini-3-flash-preview")
DEFAULT_MAX_TOKENS = int(os.environ.get("GEMINI_MEMORY_MAX_TOKENS", "5000"))
STORE_PREFIX = os.environ.get("GEMINI_STORE_PREFIX", "project-memory")

# ---------------------------------------------------------------------------
# Gemini Client (lazy init)
# ---------------------------------------------------------------------------

_client = None


def get_client():
    """Lazy-initialize the Gemini client."""
    global _client
    if _client is None:
        try:
            from google import genai
            _client = genai.Client(api_key=GEMINI_API_KEY if GEMINI_API_KEY else None)
        except ImportError:
            raise RuntimeError(
                "google-genai package not installed. Run: pip install google-genai"
            )
    return _client


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _find_store_by_name(display_name: str):
    """Find a File Search store by its display name."""
    client = get_client()
    pager = client.file_search_stores.list(config={"page_size": 20})
    while True:
        for store in pager.page:
            if store.display_name == display_name:
                return store
        try:
            pager.next_page()
        except (IndexError, StopIteration):
            break
    return None


def _make_store_display_name(project_name: str) -> str:
    """Generate a consistent store display name from a project name."""
    sanitized = project_name.strip().lower().replace(" ", "-")
    return f"{STORE_PREFIX}-{sanitized}"


def _wait_for_operation(operation, timeout: int = 120):
    """Poll an operation until it completes or times out."""
    client = get_client()
    start = time_module.time()
    while not operation.done:
        if time_module.time() - start > timeout:
            raise TimeoutError(
                f"Operation did not complete within {timeout}s. "
                "The document may still be processing — check back later."
            )
        time_module.sleep(2)
        operation = client.operations.get(operation)
    return operation


def _list_documents_in_store(store_name: str) -> list:
    """List all documents in a store."""
    client = get_client()
    docs = []
    pager = client.file_search_stores.documents.list(parent=store_name)
    while True:
        for doc in pager.page:
            docs.append(doc)
        try:
            pager.next_page()
        except (IndexError, StopIteration):
            break
    return docs


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP("gemini_memory_mcp")

# ===== Input Models =====


class CreateStoreInput(BaseModel):
    """Input for creating a new project memory store."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    project_name: str = Field(
        ...,
        description="Human-readable project name (e.g., 'verbscribe-website', 'blighter-proposal-tool'). "
        "Used to create a unique store.",
        min_length=1,
        max_length=100,
    )


class SaveSessionInput(BaseModel):
    """Input for saving a session summary to a project store."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    project_name: str = Field(
        ...,
        description="The project name whose store to save to.",
        min_length=1,
        max_length=100,
    )
    summary: str = Field(
        ...,
        description=(
            "Structured session summary. Should include:\n"
            "- What was built/changed\n"
            "- Key decisions made\n"
            "- File paths involved\n"
            "- APIs/endpoints/hooks/schemas used\n"
            "- Current state/status\n"
            "- Next steps or open issues"
        ),
        min_length=10,
    )
    session_label: Optional[str] = Field(
        default=None,
        description="Optional label for this session (e.g., 'landing-page-v1', 'auth-backend'). "
        "Defaults to a timestamp.",
        max_length=200,
    )


class RetrieveContextInput(BaseModel):
    """Input for retrieving relevant project context."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    project_name: str = Field(
        ...,
        description="The project name whose store to query.",
        min_length=1,
        max_length=100,
    )
    query: str = Field(
        ...,
        description=(
            "What you need context about. Be specific — e.g., "
            "'authentication flow and JWT setup', "
            "'database schema for user profiles', "
            "'webhook endpoints and payload formats'."
        ),
        min_length=3,
    )
    max_output_tokens: Optional[int] = Field(
        default=None,
        description=(
            f"Max tokens for the retrieved context summary. "
            f"Default: {DEFAULT_MAX_TOKENS}. "
            "Lower values (1000-2000) give quick overviews; "
            "higher values (5000-8000) give detailed context."
        ),
        ge=500,
        le=16000,
    )


class UploadFileInput(BaseModel):
    """Input for uploading a specific file to a project store."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    project_name: str = Field(
        ...,
        description="The project name whose store to upload to.",
        min_length=1,
        max_length=100,
    )
    file_path: str = Field(
        ...,
        description="Absolute path to the file to upload.",
        min_length=1,
    )
    document_label: Optional[str] = Field(
        default=None,
        description="Optional display name for the document. Defaults to the filename.",
        max_length=200,
    )


class ProjectNameInput(BaseModel):
    """Input requiring just a project name."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    project_name: str = Field(
        ...,
        description="The project name.",
        min_length=1,
        max_length=100,
    )


class DeleteDocumentInput(BaseModel):
    """Input for deleting a document from a store."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    project_name: str = Field(
        ...,
        description="The project name whose store contains the document.",
        min_length=1,
        max_length=100,
    )
    document_display_name: str = Field(
        ...,
        description="The display name of the document to delete.",
        min_length=1,
    )


class ConsolidateInput(BaseModel):
    """Input for consolidating old sessions into a single summary."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    project_name: str = Field(
        ...,
        description="The project name whose sessions to consolidate.",
        min_length=1,
        max_length=100,
    )
    keep: int = Field(
        default=5,
        description="Number of newest sessions to keep intact (default: 5). "
        "Older sessions will be merged into one consolidated summary.",
        ge=1,
        le=50,
    )


class DeleteStoreInput(BaseModel):
    """Input for deleting a project store."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    project_name: str = Field(
        ...,
        description="The project name whose store to delete.",
        min_length=1,
        max_length=100,
    )
    confirm: bool = Field(
        ...,
        description="Must be true to confirm deletion. This action is irreversible.",
    )


# ===== Tools =====


@mcp.tool(
    name="gemini_memory_create_store",
    annotations={
        "title": "Create Project Memory Store",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def create_store(params: CreateStoreInput) -> str:
    """Create a new Gemini File Search store for a project.

    Each project gets its own persistent RAG store where session summaries
    and files are stored and semantically indexed. If the store already
    exists, returns its details without creating a duplicate.

    Returns:
        str: JSON with store name, display name, and status.
    """
    display_name = _make_store_display_name(params.project_name)

    # Check if store already exists
    existing = _find_store_by_name(display_name)
    if existing:
        return json.dumps({
            "status": "already_exists",
            "store_name": existing.name,
            "display_name": existing.display_name,
            "message": f"Store for project '{params.project_name}' already exists.",
        }, indent=2)

    # Create new store
    client = get_client()
    store = client.file_search_stores.create(
        config={"display_name": display_name}
    )

    return json.dumps({
        "status": "created",
        "store_name": store.name,
        "display_name": store.display_name,
        "message": f"Store created for project '{params.project_name}'. Ready to save sessions.",
    }, indent=2)


@mcp.tool(
    name="gemini_memory_list_stores",
    annotations={
        "title": "List All Project Memory Stores",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def list_stores() -> str:
    """List all Gemini File Search stores (filtered to project-memory stores).

    Returns:
        str: JSON array of stores with their names, display names, and document counts.
    """
    client = get_client()
    stores = []
    pager = client.file_search_stores.list(config={"page_size": 20})
    while True:
        for store in pager.page:
            if store.display_name and store.display_name.startswith(STORE_PREFIX):
                project = store.display_name.replace(f"{STORE_PREFIX}-", "", 1)
                stores.append({
                    "store_name": store.name,
                    "display_name": store.display_name,
                    "project": project,
                    "active_documents": getattr(store, "active_document_count", None),
                    "pending_documents": getattr(store, "pending_document_count", None),
                })
        try:
            pager.next_page()
        except (IndexError, StopIteration):
            break

    return json.dumps({
        "count": len(stores),
        "stores": stores,
    }, indent=2)


@mcp.tool(
    name="gemini_memory_save_session",
    annotations={
        "title": "Save Session to Project Memory",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def save_session(params: SaveSessionInput) -> str:
    """Save a session summary to a project's memory store.

    The summary is uploaded as a timestamped document and indexed for
    semantic retrieval. Future sessions can then query this store to
    understand what was done previously.

    Best practice: Call this at the end of each coding session with a
    structured summary of what was accomplished.

    Returns:
        str: JSON with upload status and document details.
    """
    display_name = _make_store_display_name(params.project_name)
    store = _find_store_by_name(display_name)

    if not store:
        return json.dumps({
            "status": "error",
            "message": (
                f"No store found for project '{params.project_name}'. "
                "Create one first with gemini_memory_create_store."
            ),
        }, indent=2)

    # Build the document content
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    label = params.session_label or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    doc_display_name = f"session-{label}"

    content = (
        f"# Session Log: {label}\n"
        f"**Timestamp:** {timestamp}\n"
        f"**Project:** {params.project_name}\n\n"
        f"{params.summary}"
    )

    # Write to a temp file and upload
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, prefix=f"session-{label}-"
    ) as f:
        f.write(content)
        tmp_path = f.name

    try:
        client = get_client()
        operation = client.file_search_stores.upload_to_file_search_store(
            file=tmp_path,
            file_search_store_name=store.name,
            config={
                "display_name": doc_display_name,
                "custom_metadata": [
                    {"key": "type", "string_value": "session_log"},
                    {"key": "project", "string_value": params.project_name},
                    {"key": "label", "string_value": label},
                    {"key": "timestamp", "string_value": timestamp},
                ],
            },
        )

        operation = _wait_for_operation(operation)

        return json.dumps({
            "status": "saved",
            "document_name": doc_display_name,
            "store": store.display_name,
            "timestamp": timestamp,
            "summary_length": len(params.summary),
            "message": f"Session '{label}' saved to project '{params.project_name}'.",
        }, indent=2)

    finally:
        os.unlink(tmp_path)


@mcp.tool(
    name="gemini_memory_retrieve_context",
    annotations={
        "title": "Retrieve Project Context from Memory",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def retrieve_context(params: RetrieveContextInput) -> str:
    """Retrieve relevant context from a project's memory store.

    Uses Gemini Flash to semantically search the project store and return
    a synthesized summary of relevant past work. This is the primary tool
    for giving agents cross-session awareness.

    Call this at the start of a session or whenever you need context about
    what was previously done in a project.

    Returns:
        str: JSON with the retrieved context and metadata.
    """
    display_name = _make_store_display_name(params.project_name)
    store = _find_store_by_name(display_name)

    if not store:
        return json.dumps({
            "status": "error",
            "message": (
                f"No store found for project '{params.project_name}'. "
                "Create one first with gemini_memory_create_store."
            ),
        }, indent=2)

    max_tokens = params.max_output_tokens or DEFAULT_MAX_TOKENS

    # Use Gemini to query the store with a retrieval-focused prompt
    client = get_client()
    from google.genai import types

    # Count documents to include in response metadata
    doc_count = len(_list_documents_in_store(store.name))

    retrieval_prompt = (
        f"You are a project memory retrieval system for a coding agent. "
        f"Search the project store and return ALL relevant information about the query below.\n\n"
        f"PRIORITIES (follow this order):\n"
        f"1. MOST RECENT sessions first — newer information supersedes older\n"
        f"2. UNRESOLVED issues, known bugs, and open TODOs — these are critical for ongoing work\n"
        f"3. Current architecture and file paths — what exists NOW, not historical states\n"
        f"4. Key decisions and their rationale — why things were built a certain way\n"
        f"5. Function signatures, API endpoints, database schemas — actionable technical detail\n\n"
        f"Query: {params.query}\n\n"
        f"Return a structured response a coding agent can act on immediately. "
        f"When multiple sessions describe the same topic, synthesize into the CURRENT state "
        f"rather than repeating the chronological history. Include timestamps only for the "
        f"most recent relevant session."
    )

    response = client.models.generate_content(
        model=RETRIEVAL_MODEL,
        contents=retrieval_prompt,
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[store.name]
                    )
                )
            ],
            max_output_tokens=max_tokens,
        ),
    )

    # Extract grounding metadata if available
    grounding_info = None
    if response.candidates and response.candidates[0].grounding_metadata:
        gm = response.candidates[0].grounding_metadata
        grounding_info = {
            "grounding_chunks_count": len(gm.grounding_chunks) if gm.grounding_chunks else 0,
            "grounding_supports_count": len(gm.grounding_supports) if gm.grounding_supports else 0,
        }

    return json.dumps({
        "status": "retrieved",
        "project": params.project_name,
        "query": params.query,
        "context": response.text,
        "total_sessions": doc_count,
        "grounding": grounding_info,
        "model_used": RETRIEVAL_MODEL,
        "message": f"Context retrieved for project '{params.project_name}' ({doc_count} documents in store).",
    }, indent=2)


@mcp.tool(
    name="gemini_memory_upload_file",
    annotations={
        "title": "Upload File to Project Memory",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def upload_file(params: UploadFileInput) -> str:
    """Upload a file directly to a project's memory store.

    Useful for indexing key project files (schemas, configs, READMEs, etc.)
    so they're available as context in future sessions.

    Supported formats: .txt, .md, .py, .js, .ts, .json, .yaml, .yml,
    .html, .css, .sql, .csv, .pdf, .docx, and more.

    Returns:
        str: JSON with upload status and document details.
    """
    display_name = _make_store_display_name(params.project_name)
    store = _find_store_by_name(display_name)

    if not store:
        return json.dumps({
            "status": "error",
            "message": (
                f"No store found for project '{params.project_name}'. "
                "Create one first with gemini_memory_create_store."
            ),
        }, indent=2)

    if not os.path.exists(params.file_path):
        return json.dumps({
            "status": "error",
            "message": f"File not found: {params.file_path}",
        }, indent=2)

    doc_label = params.document_label or os.path.basename(params.file_path)

    client = get_client()
    operation = client.file_search_stores.upload_to_file_search_store(
        file=params.file_path,
        file_search_store_name=store.name,
        config={
            "display_name": doc_label,
            "custom_metadata": [
                {"key": "type", "string_value": "project_file"},
                {"key": "project", "string_value": params.project_name},
                {"key": "original_path", "string_value": params.file_path},
                {"key": "uploaded_at", "string_value": datetime.now(timezone.utc).isoformat()},
            ],
        },
    )

    operation = _wait_for_operation(operation)

    return json.dumps({
        "status": "uploaded",
        "document_name": doc_label,
        "store": store.display_name,
        "file_path": params.file_path,
        "message": f"File '{doc_label}' uploaded to project '{params.project_name}'.",
    }, indent=2)


@mcp.tool(
    name="gemini_memory_list_documents",
    annotations={
        "title": "List Documents in Project Memory",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def list_documents(params: ProjectNameInput) -> str:
    """List all documents stored in a project's memory.

    Returns:
        str: JSON array of documents with names and display names.
    """
    display_name = _make_store_display_name(params.project_name)
    store = _find_store_by_name(display_name)

    if not store:
        return json.dumps({
            "status": "error",
            "message": f"No store found for project '{params.project_name}'.",
        }, indent=2)

    docs = _list_documents_in_store(store.name)
    doc_list = []
    for doc in docs:
        doc_list.append({
            "name": doc.name,
            "display_name": doc.display_name,
            "state": getattr(doc, "state", None),
        })

    return json.dumps({
        "status": "ok",
        "project": params.project_name,
        "count": len(doc_list),
        "documents": doc_list,
    }, indent=2)


@mcp.tool(
    name="gemini_memory_delete_document",
    annotations={
        "title": "Delete Document from Project Memory",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def delete_document(params: DeleteDocumentInput) -> str:
    """Delete a specific document from a project's memory store.

    Returns:
        str: JSON with deletion status.
    """
    display_name = _make_store_display_name(params.project_name)
    store = _find_store_by_name(display_name)

    if not store:
        return json.dumps({
            "status": "error",
            "message": f"No store found for project '{params.project_name}'.",
        }, indent=2)

    # Find the document by display name
    docs = _list_documents_in_store(store.name)
    target = None
    for doc in docs:
        if doc.display_name == params.document_display_name:
            target = doc
            break

    if not target:
        return json.dumps({
            "status": "error",
            "message": (
                f"Document '{params.document_display_name}' not found in "
                f"project '{params.project_name}'."
            ),
        }, indent=2)

    client = get_client()
    client.file_search_stores.documents.delete(
        name=target.name,
        config={"force": True},
    )

    return json.dumps({
        "status": "deleted",
        "document": params.document_display_name,
        "project": params.project_name,
        "message": f"Document '{params.document_display_name}' deleted.",
    }, indent=2)


@mcp.tool(
    name="gemini_memory_consolidate",
    annotations={
        "title": "Consolidate Old Sessions",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def consolidate_sessions(params: ConsolidateInput) -> str:
    """Consolidate old session documents into a single summary.

    When a project accumulates many sessions (20+), this merges the oldest
    sessions into one consolidated summary document while keeping the N
    newest sessions intact. Reduces document count and retrieval noise.

    Returns:
        str: JSON with consolidation results.
    """
    display_name = _make_store_display_name(params.project_name)
    store = _find_store_by_name(display_name)

    if not store:
        return json.dumps({
            "status": "error",
            "message": f"No store found for project '{params.project_name}'.",
        }, indent=2)

    docs = _list_documents_in_store(store.name)
    session_docs = [d for d in docs if d.display_name and d.display_name.startswith("session-")]

    if len(session_docs) <= params.keep:
        return json.dumps({
            "status": "skipped",
            "message": f"Only {len(session_docs)} sessions — nothing to consolidate (keep={params.keep}).",
            "session_count": len(session_docs),
        }, indent=2)

    session_docs.sort(key=lambda d: d.display_name)
    to_consolidate = session_docs[:-params.keep]
    to_keep = session_docs[-params.keep:]

    # Use Gemini to synthesize old sessions
    client = get_client()
    from google.genai import types

    consolidation_prompt = (
        f"You are a project memory consolidation system. The project store contains "
        f"{len(to_consolidate)} older session logs that need to be merged into a single "
        f"comprehensive summary document.\n\n"
        f"Create a consolidated summary that preserves ALL important information:\n"
        f"1. Current architecture and file structure\n"
        f"2. ALL unresolved bugs and known issues (do NOT drop any)\n"
        f"3. Key decisions made and their rationale\n"
        f"4. Dead code identified for cleanup\n"
        f"5. Database schemas, API endpoints, function signatures\n"
        f"6. What was built and what changed\n\n"
        f"Format as a structured reference document. Merge overlapping information "
        f"into the CURRENT state. Drop only truly redundant duplicate descriptions. "
        f"Preserve every file path, every bug, every decision."
    )

    response = client.models.generate_content(
        model=RETRIEVAL_MODEL,
        contents=consolidation_prompt,
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[store.name]
                    )
                )
            ],
            max_output_tokens=16000,
        ),
    )

    consolidated_text = response.text
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    label = f"consolidated-{datetime.now(timezone.utc).strftime('%Y%m%d')}"

    content = (
        f"# Consolidated Project Summary: {params.project_name}\n"
        f"**Consolidated:** {timestamp}\n"
        f"**Sessions merged:** {len(to_consolidate)}\n"
        f"**Sessions kept separately:** {len(to_keep)}\n\n"
        f"{consolidated_text}"
    )

    # Upload consolidated document
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, prefix="consolidated-"
    ) as f:
        f.write(content)
        tmp_path = f.name

    try:
        op = client.file_search_stores.upload_to_file_search_store(
            file=tmp_path,
            file_search_store_name=store.name,
            config={
                "display_name": f"session-{label}",
                "custom_metadata": [
                    {"key": "type", "string_value": "consolidated_summary"},
                    {"key": "project", "string_value": params.project_name},
                    {"key": "sessions_merged", "string_value": str(len(to_consolidate))},
                    {"key": "timestamp", "string_value": timestamp},
                ],
            },
        )
        _wait_for_operation(op)
    finally:
        os.unlink(tmp_path)

    # Delete old sessions
    deleted = []
    for doc in to_consolidate:
        client.file_search_stores.documents.delete(name=doc.name, config={"force": True})
        deleted.append(doc.display_name)

    final_count = len(to_keep) + 1

    return json.dumps({
        "status": "consolidated",
        "project": params.project_name,
        "sessions_merged": len(to_consolidate),
        "sessions_kept": len(to_keep),
        "documents_after": final_count,
        "deleted_sessions": deleted,
        "message": f"Consolidated {len(to_consolidate)} old sessions into 1 summary. {final_count} documents remain.",
    }, indent=2)


@mcp.tool(
    name="gemini_memory_delete_store",
    annotations={
        "title": "Delete Project Memory Store",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def delete_store(params: DeleteStoreInput) -> str:
    """Delete an entire project memory store and all its documents.

    ⚠️ This action is IRREVERSIBLE. All session logs and uploaded files
    in the store will be permanently deleted.

    Returns:
        str: JSON with deletion status.
    """
    if not params.confirm:
        return json.dumps({
            "status": "cancelled",
            "message": "Deletion cancelled. Set confirm=true to proceed.",
        }, indent=2)

    display_name = _make_store_display_name(params.project_name)
    store = _find_store_by_name(display_name)

    if not store:
        return json.dumps({
            "status": "error",
            "message": f"No store found for project '{params.project_name}'.",
        }, indent=2)

    client = get_client()
    client.file_search_stores.delete(
        name=store.name,
        config={"force": True},
    )

    return json.dumps({
        "status": "deleted",
        "store": display_name,
        "project": params.project_name,
        "message": f"Store for project '{params.project_name}' permanently deleted.",
    }, indent=2)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
