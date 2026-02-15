#!/usr/bin/env python3
"""
gmem — Gemini Memory CLI
=========================
Standalone CLI tool for persistent project memory using Gemini File Search stores.
Works with ANY coding agent (Claude Code, Gemini CLI, Cursor, Aider, etc.)

Usage:
    gmem init <project>              Create a memory store for a project
    gmem save <project> [--label]    Save a session summary (interactive or piped)
    gmem context <project> <query>   Retrieve relevant context
    gmem upload <project> <file>     Upload a file to project memory
    gmem docs <project>              List documents in a project store
    gmem stores                      List all project stores
    gmem delete-doc <project> <doc>  Delete a document
    gmem delete <project>            Delete an entire project store
    gmem auto-save <project> <dir>   Auto-generate summary from git diff + changed files
    gmem inject <project> <query>    Retrieve context and copy to clipboard / write to file
    gmem consolidate <project>       Merge old sessions into one summary (keeps newest)

Examples:
    gmem init my-website
    gmem save my-website --label "auth-backend"
    gmem context my-website "authentication flow and JWT setup"
    gmem context my-website "all endpoints" --tokens 2000
    gmem upload my-website ./schema.sql
    gmem auto-save my-website . --label "session-3"
    gmem inject my-website "full architecture" --output .context.md
    gmem inject my-website "what was done last" --clipboard

Author: Built for Aleksander Celewicz
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STORE_PREFIX = os.environ.get("GEMINI_STORE_PREFIX", "project-memory")
RETRIEVAL_MODEL = os.environ.get("GEMINI_RETRIEVAL_MODEL", "gemini-3-flash-preview")
DEFAULT_MAX_TOKENS = int(os.environ.get("GEMINI_MEMORY_MAX_TOKENS", "5000"))

# ---------------------------------------------------------------------------
# Gemini Client
# ---------------------------------------------------------------------------

_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        try:
            from google import genai
            _client = genai.Client(api_key=api_key if api_key else None)
        except ImportError:
            print("❌ google-genai not installed. Run: pip install google-genai")
            sys.exit(1)
    return _client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def store_name(project: str) -> str:
    return f"{STORE_PREFIX}-{project.strip().lower().replace(' ', '-')}"


def find_store(display_name: str):
    client = get_client()
    pager = client.file_search_stores.list(config={"page_size": 20})
    while True:
        for s in pager.page:
            if s.display_name == display_name:
                return s
        try:
            pager.next_page()
        except (IndexError, StopIteration):
            break
    return None


def require_store(project: str):
    name = store_name(project)
    store = find_store(name)
    if not store:
        print(f"❌ No store found for project '{project}'.")
        print(f"   Create one first: gmem init {project}")
        sys.exit(1)
    return store


def wait_op(operation, label="operation", timeout=120):
    client = get_client()
    start = time.time()
    spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    i = 0
    while not operation.done:
        if time.time() - start > timeout:
            print(f"\n⚠️  {label} timed out after {timeout}s (may still be processing)")
            return operation
        print(f"\r  {spinner[i % len(spinner)]} {label}...", end="", flush=True)
        i += 1
        time.sleep(1)
        operation = client.operations.get(operation)
    print(f"\r  ✅ {label} complete.   ")
    return operation


def list_store_docs(store_name_str: str) -> list:
    client = get_client()
    docs = []
    pager = client.file_search_stores.documents.list(parent=store_name_str)
    while True:
        for doc in pager.page:
            docs.append(doc)
        try:
            pager.next_page()
        except (IndexError, StopIteration):
            break
    return docs


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_init(args):
    """Create a new project memory store."""
    name = store_name(args.project)
    existing = find_store(name)
    if existing:
        print(f"✅ Store already exists for '{args.project}': {existing.name}")
        return

    client = get_client()
    store = client.file_search_stores.create(config={"display_name": name})
    print(f"✅ Created store for '{args.project}': {store.name}")


def cmd_stores(args):
    """List all project memory stores."""
    client = get_client()
    stores = []
    pager = client.file_search_stores.list(config={"page_size": 20})
    while True:
        for s in pager.page:
            if s.display_name and s.display_name.startswith(STORE_PREFIX):
                proj = s.display_name.replace(f"{STORE_PREFIX}-", "", 1)
                active = getattr(s, "active_document_count", "?")
                stores.append((proj, s.display_name, s.name, active))
        try:
            pager.next_page()
        except (IndexError, StopIteration):
            break

    if not stores:
        print("No project memory stores found.")
        return

    print(f"\n{'Project':<30} {'Documents':<12} {'Store ID'}")
    print("─" * 80)
    for proj, display, sid, active in stores:
        print(f"{proj:<30} {str(active):<12} {sid}")
    print(f"\nTotal: {len(stores)} stores")


def cmd_save(args):
    """Save a session summary to project memory."""
    store = require_store(args.project)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    label = args.label or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    # Get summary from argument, stdin, or interactive
    if args.message:
        summary = args.message
    elif not sys.stdin.isatty():
        summary = sys.stdin.read().strip()
    else:
        print("Enter session summary (Ctrl+D or Ctrl+Z to finish):")
        print("─" * 40)
        lines = []
        try:
            while True:
                lines.append(input())
        except EOFError:
            pass
        summary = "\n".join(lines)

    if not summary or len(summary.strip()) < 10:
        print("❌ Summary too short (min 10 chars). Provide a meaningful summary.")
        sys.exit(1)

    content = (
        f"# Session Log: {label}\n"
        f"**Timestamp:** {timestamp}\n"
        f"**Project:** {args.project}\n\n"
        f"{summary}"
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, prefix=f"session-{label}-"
    ) as f:
        f.write(content)
        tmp_path = f.name

    try:
        client = get_client()
        op = client.file_search_stores.upload_to_file_search_store(
            file=tmp_path,
            file_search_store_name=store.name,
            config={
                "display_name": f"session-{label}",
                "custom_metadata": [
                    {"key": "type", "string_value": "session_log"},
                    {"key": "project", "string_value": args.project},
                    {"key": "label", "string_value": label},
                    {"key": "timestamp", "string_value": timestamp},
                ],
            },
        )
        wait_op(op, f"Indexing session '{label}'")
        print(f"✅ Session '{label}' saved to '{args.project}' ({len(summary)} chars)")
    finally:
        os.unlink(tmp_path)


def cmd_context(args):
    """Retrieve relevant context from project memory."""
    store = require_store(args.project)
    max_tokens = args.tokens or DEFAULT_MAX_TOKENS
    query = " ".join(args.query)

    from google.genai import types

    doc_count = len(list_store_docs(store.name))

    prompt = (
        f"You are a project memory retrieval system for a coding agent. "
        f"Search the project store and return ALL relevant information about the query below.\n\n"
        f"PRIORITIES (follow this order):\n"
        f"1. MOST RECENT sessions first — newer information supersedes older\n"
        f"2. UNRESOLVED issues, known bugs, and open TODOs\n"
        f"3. Current architecture and file paths — what exists NOW\n"
        f"4. Key decisions and their rationale\n"
        f"5. Function signatures, API endpoints, database schemas\n\n"
        f"Query: {query}\n\n"
        f"Return a structured response a coding agent can act on immediately. "
        f"When multiple sessions describe the same topic, synthesize into the CURRENT state "
        f"rather than repeating the chronological history."
    )

    client = get_client()
    print(f"🔍 Searching '{args.project}' memory ({doc_count} documents) for: {query}\n")

    response = client.models.generate_content(
        model=RETRIEVAL_MODEL,
        contents=prompt,
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

    print("─" * 60)
    print(response.text)
    print("─" * 60)

    # Show grounding info
    if response.candidates and response.candidates[0].grounding_metadata:
        gm = response.candidates[0].grounding_metadata
        chunks = len(gm.grounding_chunks) if gm.grounding_chunks else 0
        print(f"\n📎 Grounded from {chunks} document chunks")


def cmd_upload(args):
    """Upload a file to project memory."""
    store = require_store(args.project)
    filepath = args.file

    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        sys.exit(1)

    doc_label = args.label or os.path.basename(filepath)
    client = get_client()

    op = client.file_search_stores.upload_to_file_search_store(
        file=filepath,
        file_search_store_name=store.name,
        config={
            "display_name": doc_label,
            "custom_metadata": [
                {"key": "type", "string_value": "project_file"},
                {"key": "project", "string_value": args.project},
                {"key": "original_path", "string_value": os.path.abspath(filepath)},
                {"key": "uploaded_at", "string_value": datetime.now(timezone.utc).isoformat()},
            ],
        },
    )

    wait_op(op, f"Indexing '{doc_label}'")
    print(f"✅ '{doc_label}' uploaded to '{args.project}'")


def cmd_docs(args):
    """List documents in a project store."""
    store = require_store(args.project)
    docs = list_store_docs(store.name)

    if not docs:
        print(f"No documents in '{args.project}'.")
        return

    print(f"\n{'Display Name':<45} {'State':<15} {'Document ID'}")
    print("─" * 100)
    for doc in docs:
        state = getattr(doc, "state", "unknown")
        print(f"{(doc.display_name or 'unnamed'):<45} {str(state):<15} {doc.name}")
    print(f"\nTotal: {len(docs)} documents")


def cmd_delete_doc(args):
    """Delete a document from a project store."""
    store = require_store(args.project)
    docs = list_store_docs(store.name)

    target = None
    for doc in docs:
        if doc.display_name == args.document:
            target = doc
            break

    if not target:
        print(f"❌ Document '{args.document}' not found in '{args.project}'.")
        print("   Use 'gmem docs' to list documents.")
        sys.exit(1)

    if not args.yes:
        confirm = input(f"Delete '{args.document}' from '{args.project}'? (y/N): ")
        if confirm.lower() != "y":
            print("Cancelled.")
            return

    client = get_client()
    client.file_search_stores.documents.delete(name=target.name, config={"force": True})
    print(f"✅ Deleted '{args.document}' from '{args.project}'")


def cmd_delete(args):
    """Delete an entire project store."""
    store = require_store(args.project)

    if not args.yes:
        confirm = input(
            f"⚠️  DELETE ALL MEMORY for '{args.project}'? This is IRREVERSIBLE. (type 'yes'): "
        )
        if confirm != "yes":
            print("Cancelled.")
            return

    client = get_client()
    client.file_search_stores.delete(name=store.name, config={"force": True})
    print(f"✅ Deleted store for '{args.project}' and all its documents.")


def cmd_auto_save(args):
    """Auto-generate a session summary from git diff and changed files."""
    store = require_store(args.project)
    work_dir = args.dir or "."
    label = args.label or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    parts = []
    parts.append(f"# Auto-generated Session Summary: {label}")
    parts.append(f"**Timestamp:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    parts.append(f"**Project:** {args.project}")
    parts.append(f"**Working Directory:** {os.path.abspath(work_dir)}")
    parts.append("")

    # Git status
    try:
        status = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True, text=True, cwd=work_dir, timeout=10
        )
        if status.returncode == 0 and status.stdout.strip():
            parts.append("## Changed Files")
            parts.append("```")
            parts.append(status.stdout.strip())
            parts.append("```")
            parts.append("")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Git diff (staged + unstaged, limited)
    try:
        diff = subprocess.run(
            ["git", "diff", "--stat", "HEAD~1", "HEAD"],
            capture_output=True, text=True, cwd=work_dir, timeout=10
        )
        if diff.returncode == 0 and diff.stdout.strip():
            parts.append("## Recent Commit Changes")
            parts.append("```")
            parts.append(diff.stdout.strip()[:3000])
            parts.append("```")
            parts.append("")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Git log (last 5 commits)
    try:
        log = subprocess.run(
            ["git", "log", "--oneline", "-5"],
            capture_output=True, text=True, cwd=work_dir, timeout=10
        )
        if log.returncode == 0 and log.stdout.strip():
            parts.append("## Recent Commits")
            parts.append("```")
            parts.append(log.stdout.strip())
            parts.append("```")
            parts.append("")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Key project files snapshot (if they exist)
    key_files = [
        "package.json", "pyproject.toml", "requirements.txt", "Cargo.toml",
        "CLAUDE.md", "README.md", ".env.example"
    ]
    found_files = []
    for kf in key_files:
        kf_path = os.path.join(work_dir, kf)
        if os.path.exists(kf_path):
            found_files.append(kf)

    if found_files:
        parts.append(f"## Key Project Files Present")
        parts.append(", ".join(found_files))
        parts.append("")

    summary = "\n".join(parts)

    if len(summary.strip()) < 50:
        print("⚠️  Not enough git/project data to auto-generate. Use 'gmem save' instead.")
        sys.exit(1)

    # Upload
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, prefix=f"auto-session-{label}-"
    ) as f:
        f.write(summary)
        tmp_path = f.name

    try:
        client = get_client()
        op = client.file_search_stores.upload_to_file_search_store(
            file=tmp_path,
            file_search_store_name=store.name,
            config={
                "display_name": f"auto-session-{label}",
                "custom_metadata": [
                    {"key": "type", "string_value": "auto_session_log"},
                    {"key": "project", "string_value": args.project},
                    {"key": "label", "string_value": label},
                ],
            },
        )
        wait_op(op, f"Indexing auto-session '{label}'")
        print(f"✅ Auto-session '{label}' saved ({len(summary)} chars)")
    finally:
        os.unlink(tmp_path)


def cmd_consolidate(args):
    """Consolidate old sessions into a single summary, reducing document count."""
    store = require_store(args.project)
    docs = list_store_docs(store.name)

    # Filter to session docs only (not uploaded files)
    session_docs = [d for d in docs if d.display_name and d.display_name.startswith("session-")]

    if len(session_docs) <= args.keep:
        print(f"Only {len(session_docs)} sessions — nothing to consolidate (keep={args.keep}).")
        return

    # Sort by name to get chronological order (timestamps in names)
    session_docs.sort(key=lambda d: d.display_name)

    # The ones to consolidate (oldest) vs keep (newest)
    to_consolidate = session_docs[:-args.keep]
    to_keep = session_docs[-args.keep:]

    print(f"Found {len(session_docs)} sessions.")
    print(f"  Consolidating: {len(to_consolidate)} oldest sessions into 1 summary")
    print(f"  Keeping: {len(to_keep)} newest sessions unchanged")

    if not args.yes:
        print(f"\nSessions to consolidate:")
        for d in to_consolidate:
            print(f"  - {d.display_name}")
        print(f"\nSessions to keep:")
        for d in to_keep:
            print(f"  - {d.display_name}")
        confirm = input(f"\nProceed? (y/N): ")
        if confirm.lower() != "y":
            print("Cancelled.")
            return

    # Use Gemini to synthesize the old sessions into a single summary
    from google.genai import types

    print(f"\nGenerating consolidated summary from {len(to_consolidate)} sessions...")

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

    client = get_client()
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
        f"# Consolidated Project Summary: {args.project}\n"
        f"**Consolidated:** {timestamp}\n"
        f"**Sessions merged:** {len(to_consolidate)}\n"
        f"**Sessions kept separately:** {len(to_keep)}\n\n"
        f"{consolidated_text}"
    )

    # Upload the consolidated document
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, prefix=f"consolidated-"
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
                    {"key": "project", "string_value": args.project},
                    {"key": "sessions_merged", "string_value": str(len(to_consolidate))},
                    {"key": "timestamp", "string_value": timestamp},
                ],
            },
        )
        wait_op(op, "Indexing consolidated summary")
    finally:
        os.unlink(tmp_path)

    # Delete the old sessions
    print(f"Deleting {len(to_consolidate)} old sessions...")
    for doc in to_consolidate:
        client.file_search_stores.documents.delete(name=doc.name, config={"force": True})
        print(f"  Deleted: {doc.display_name}")

    final_count = len(to_keep) + 1  # kept sessions + new consolidated doc
    print(f"\nConsolidation complete: {len(session_docs)} sessions -> {final_count} documents")


def cmd_inject(args):
    """Retrieve context and output to file or clipboard for agent injection."""
    store = require_store(args.project)
    max_tokens = args.tokens or DEFAULT_MAX_TOKENS
    query = " ".join(args.query)

    from google.genai import types

    prompt = (
        f"You are a project memory system providing context to a coding agent. "
        f"Return ALL relevant information about: {query}\n\n"
        f"Prioritize: most recent state over historical, unresolved issues, "
        f"current file paths, function signatures, and key decisions. "
        f"Synthesize into current state rather than repeating chronological history. "
        f"Format as a structured briefing the agent can act on immediately."
    )

    client = get_client()
    response = client.models.generate_content(
        model=RETRIEVAL_MODEL,
        contents=prompt,
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

    context_text = response.text

    if args.output:
        # Write to file
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(f"<!-- Project Memory Context for: {args.project} -->\n")
            f.write(f"<!-- Query: {query} -->\n")
            f.write(f"<!-- Retrieved: {datetime.now(timezone.utc).isoformat()} -->\n\n")
            f.write(context_text)
        print(f"✅ Context written to {args.output}")
        print(f"   Inject into agent with: cat {args.output}")
    elif args.clipboard:
        # Copy to clipboard
        try:
            if sys.platform == "win32":
                process = subprocess.Popen(["clip"], stdin=subprocess.PIPE)
                process.communicate(context_text.encode("utf-16le"))
            elif sys.platform == "darwin":
                process = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
                process.communicate(context_text.encode("utf-8"))
            else:
                process = subprocess.Popen(["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE)
                process.communicate(context_text.encode("utf-8"))
            print(f"✅ Context copied to clipboard ({len(context_text)} chars)")
        except FileNotFoundError:
            print("⚠️  Clipboard tool not found. Printing to stdout instead:\n")
            print(context_text)
    else:
        # Print to stdout (can be piped)
        print(context_text)


# ---------------------------------------------------------------------------
# CLI Parser
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="gmem",
        description="Gemini Memory — Persistent project memory for coding agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  gmem init my-website\n"
            "  gmem save my-website --label auth-v1 -m 'Built JWT auth with refresh tokens'\n"
            "  gmem context my-website 'authentication endpoints and JWT flow'\n"
            "  gmem auto-save my-website . --label session-3\n"
            "  gmem inject my-website 'full architecture' --output .context.md\n"
            "  gmem inject my-website 'recent work' --clipboard\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # init
    p = sub.add_parser("init", help="Create a memory store for a project")
    p.add_argument("project", help="Project name")

    # stores
    sub.add_parser("stores", help="List all project memory stores")

    # save
    p = sub.add_parser("save", help="Save a session summary")
    p.add_argument("project", help="Project name")
    p.add_argument("--label", "-l", help="Session label (default: timestamp)")
    p.add_argument("--message", "-m", help="Summary text (or pipe from stdin, or interactive)")

    # context
    p = sub.add_parser("context", help="Retrieve context from project memory")
    p.add_argument("project", help="Project name")
    p.add_argument("query", nargs="+", help="What to search for")
    p.add_argument("--tokens", "-t", type=int, help=f"Max output tokens (default: {DEFAULT_MAX_TOKENS})")

    # upload
    p = sub.add_parser("upload", help="Upload a file to project memory")
    p.add_argument("project", help="Project name")
    p.add_argument("file", help="File path to upload")
    p.add_argument("--label", "-l", help="Document label (default: filename)")

    # docs
    p = sub.add_parser("docs", help="List documents in a project store")
    p.add_argument("project", help="Project name")

    # delete-doc
    p = sub.add_parser("delete-doc", help="Delete a document")
    p.add_argument("project", help="Project name")
    p.add_argument("document", help="Document display name")
    p.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    # delete
    p = sub.add_parser("delete", help="Delete an entire project store")
    p.add_argument("project", help="Project name")
    p.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    # auto-save
    p = sub.add_parser("auto-save", help="Auto-generate summary from git + project state")
    p.add_argument("project", help="Project name")
    p.add_argument("dir", nargs="?", default=".", help="Working directory (default: .)")
    p.add_argument("--label", "-l", help="Session label (default: timestamp)")

    # inject
    p = sub.add_parser("inject", help="Retrieve context for agent injection")
    p.add_argument("project", help="Project name")
    p.add_argument("query", nargs="+", help="What to search for")
    p.add_argument("--tokens", "-t", type=int, help=f"Max output tokens (default: {DEFAULT_MAX_TOKENS})")
    p.add_argument("--output", "-o", help="Write to file (e.g., .context.md)")
    p.add_argument("--clipboard", "-c", action="store_true", help="Copy to clipboard")

    # consolidate
    p = sub.add_parser("consolidate", help="Merge old sessions into one summary")
    p.add_argument("project", help="Project name")
    p.add_argument("--keep", "-k", type=int, default=5, help="Number of newest sessions to keep intact (default: 5)")
    p.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Dispatch
    commands = {
        "init": cmd_init,
        "stores": cmd_stores,
        "save": cmd_save,
        "context": cmd_context,
        "upload": cmd_upload,
        "docs": cmd_docs,
        "delete-doc": cmd_delete_doc,
        "delete": cmd_delete,
        "auto-save": cmd_auto_save,
        "inject": cmd_inject,
        "consolidate": cmd_consolidate,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
