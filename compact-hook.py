#!/usr/bin/env python3
"""
PreCompact Hook — Save session context to Gemini Memory before compaction.

Reads the conversation transcript, summarizes it with Gemini, and uploads
the summary to the project's memory store so detailed context survives
context compaction.

Called by Claude Code's PreCompact hook event.
Input (stdin): JSON with transcript_path, cwd, trigger
Output: None (exit 0 always — never interfere with compaction)
"""

import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STORE_PREFIX = os.environ.get("GEMINI_STORE_PREFIX", "project-memory")
SUMMARY_MODEL = os.environ.get("GEMINI_SUMMARY_MODEL", "gemini-3-flash-preview")
MAX_TRANSCRIPT_CHARS = 80_000  # Focus on recent work


# ---------------------------------------------------------------------------
# Helpers (mirror patterns from gmem.py)
# ---------------------------------------------------------------------------

def get_client():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    from google import genai
    return genai.Client(api_key=api_key if api_key else None)


def store_display_name(project: str) -> str:
    return f"{STORE_PREFIX}-{project.strip().lower().replace(' ', '-')}"


def derive_project_name(cwd: str) -> str:
    """Derive project name from cwd folder (same logic as CLAUDE.md)."""
    return Path(cwd).name.lower().replace(" ", "-")


def find_store(client, display_name: str):
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


def find_or_create_store(client, display_name: str):
    store = find_store(client, display_name)
    if store:
        return store
    return client.file_search_stores.create(config={"display_name": display_name})


def extract_transcript_text(transcript_path: str) -> str:
    """Read JSONL transcript and extract user/assistant text messages."""
    messages = []
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                role = entry.get("role", "")
                if role not in ("user", "assistant"):
                    continue

                content = entry.get("content", "")
                text = ""
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    parts = []
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                parts.append(block.get("text", ""))
                            elif "text" in block:
                                parts.append(block["text"])
                        elif isinstance(block, str):
                            parts.append(block)
                    text = "\n".join(parts)

                if text.strip():
                    messages.append(f"[{role.upper()}]: {text.strip()}")
    except Exception:
        return ""

    full_text = "\n\n".join(messages)

    # Truncate to last MAX_TRANSCRIPT_CHARS (focus on recent work)
    if len(full_text) > MAX_TRANSCRIPT_CHARS:
        full_text = full_text[-MAX_TRANSCRIPT_CHARS:]
        # Find a clean message boundary
        boundary = full_text.find("\n\n[")
        if 0 < boundary < 2000:
            full_text = full_text[boundary + 2:]

    return full_text


def summarize_transcript(client, transcript_text: str, project: str) -> str:
    """Use Gemini to summarize the transcript into a structured session summary."""
    prompt = (
        f'You are a session summarizer for a coding project called "{project}".\n\n'
        "Below is a conversation transcript between a user and an AI coding assistant. "
        "Summarize it into a structured session document that another AI agent can use "
        "to continue the work.\n\n"
        "REQUIRED SECTIONS:\n\n"
        "## What Was Built/Changed\n"
        "- List all files created or modified with their paths\n"
        "- Describe each change concisely\n\n"
        "## Key Decisions\n"
        "- Architecture decisions and their rationale\n"
        "- Technology choices made\n"
        "- Tradeoffs considered\n\n"
        "## Technical Details\n"
        "- API Endpoints: any new/modified endpoints\n"
        "- Database/Schema: any schema changes\n"
        "- Dependencies: new packages installed\n"
        "- Environment Variables: new env vars\n"
        "- Configuration changes\n\n"
        "## Current Status\n"
        "- What works now\n"
        "- What's partially done\n"
        "- Known bugs or issues\n\n"
        "## Next Steps\n"
        "- What should be done next\n"
        "- Open questions or unresolved items\n\n"
        "## Important Code Patterns\n"
        "- Key function signatures, class structures\n"
        "- Recurring patterns or conventions established\n\n"
        "RULES:\n"
        "- Be COMPREHENSIVE — include every file path, every decision, every bug\n"
        "- Focus on CURRENT STATE over historical narrative\n"
        "- Include exact file paths, function names, variable names\n"
        "- Note any workarounds or temporary fixes\n"
        "- Preserve error messages and their resolutions\n"
        "- Maximum detail, minimum fluff\n\n"
        f"TRANSCRIPT:\n{transcript_text}"
    )

    from google.genai import types

    response = client.models.generate_content(
        model=SUMMARY_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=8000,
        ),
    )
    return response.text


def upload_summary(client, store, summary: str, project: str, label: str):
    """Upload summary document to the store and wait for indexing."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    content = (
        f"# Session Log: {label}\n"
        f"**Timestamp:** {timestamp}\n"
        f"**Project:** {project}\n"
        f"**Source:** Pre-compaction auto-save\n\n"
        f"{summary}"
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False,
        prefix=f"compact-session-{label}-", encoding="utf-8"
    ) as f:
        f.write(content)
        tmp_path = f.name

    try:
        op = client.file_search_stores.upload_to_file_search_store(
            file=tmp_path,
            file_search_store_name=store.name,
            config={
                "display_name": f"session-compact-{label}",
                "custom_metadata": [
                    {"key": "type", "string_value": "compact_session_log"},
                    {"key": "project", "string_value": project},
                    {"key": "label", "string_value": label},
                    {"key": "timestamp", "string_value": timestamp},
                ],
            },
        )

        # Wait for indexing (up to 60s) so post-compact retrieval can find it
        start = time.time()
        while not op.done:
            if time.time() - start > 60:
                break
            time.sleep(2)
            op = client.operations.get(op)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    try:
        # Read hook input from stdin
        raw = sys.stdin.read()
        hook_input = json.loads(raw) if raw.strip() else {}

        transcript_path = hook_input.get("transcript_path", "")
        cwd = hook_input.get("cwd", "")

        if not cwd:
            sys.exit(0)

        project = derive_project_name(cwd)

        # Read and extract transcript
        transcript_text = ""
        if transcript_path and os.path.exists(transcript_path):
            transcript_text = extract_transcript_text(transcript_path)

        if not transcript_text or len(transcript_text) < 100:
            sys.exit(0)

        # Initialize Gemini client
        client = get_client()

        # Find or create the project store
        display = store_display_name(project)
        store = find_or_create_store(client, display)

        # Summarize the transcript
        summary = summarize_transcript(client, transcript_text, project)

        if not summary or len(summary) < 50:
            sys.exit(0)

        # Upload to store and wait for indexing
        label = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        upload_summary(client, store, summary, project, label)

    except Exception:
        # Never interfere with compaction
        pass

    sys.exit(0)


if __name__ == "__main__":
    main()
