#!/usr/bin/env python3
"""
Post-Compact Hook — Restore session context from Gemini Memory after compaction.

Retrieves the project's context from Gemini Memory and outputs it as
additionalContext JSON for Claude to see alongside the compacted summary.

Called by Claude Code's SessionStart hook event (matcher: compact).
Input (stdin): JSON with cwd, source
Output (stdout): JSON with hookSpecificOutput.additionalContext
"""

import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STORE_PREFIX = os.environ.get("GEMINI_STORE_PREFIX", "project-memory")
RETRIEVAL_MODEL = os.environ.get("GEMINI_RETRIEVAL_MODEL", "gemini-3-flash-preview")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_client():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    from google import genai
    return genai.Client(api_key=api_key if api_key else None)


def store_display_name(project: str) -> str:
    return f"{STORE_PREFIX}-{project.strip().lower().replace(' ', '-')}"


def derive_project_name(cwd: str) -> str:
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    try:
        raw = sys.stdin.read()
        hook_input = json.loads(raw) if raw.strip() else {}

        cwd = hook_input.get("cwd", "")
        if not cwd:
            sys.exit(0)

        project = derive_project_name(cwd)
        client = get_client()

        display = store_display_name(project)
        store = find_store(client, display)

        if not store:
            sys.exit(0)

        from google.genai import types

        query = (
            "You are a project memory system restoring context after compaction. "
            "Return a comprehensive briefing covering:\n"
            "1. MOST RECENT session work — what was just being done\n"
            "2. Current project architecture and file structure\n"
            "3. All unresolved bugs, issues, and TODOs\n"
            "4. Key decisions and their rationale\n"
            "5. Database schemas, API endpoints, function signatures\n"
            "6. Next steps and open questions\n\n"
            "Prioritize the MOST RECENT sessions heavily — the agent just lost "
            "context due to compaction and needs to continue where it left off. "
            "Format as a structured document the agent can act on immediately."
        )

        response = client.models.generate_content(
            model=RETRIEVAL_MODEL,
            contents=query,
            config=types.GenerateContentConfig(
                tools=[
                    types.Tool(
                        file_search=types.FileSearch(
                            file_search_store_names=[store.name]
                        )
                    )
                ],
                max_output_tokens=10000,
            ),
        )

        context_text = response.text if response.text else ""

        if context_text:
            header = (
                "=== GEMINI MEMORY CONTEXT RESTORED (post-compaction) ===\n"
                f"Project: {project}\n"
                "This context was automatically retrieved from Gemini Memory "
                "to supplement the compacted summary above. Use this detailed "
                "context to continue working seamlessly.\n"
                "=========================================================\n\n"
            )

            output = {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": header + context_text
                }
            }
            print(json.dumps(output))

    except Exception:
        # Never interfere with session start
        pass

    sys.exit(0)


if __name__ == "__main__":
    main()
