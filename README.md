# Gemini Memory MCP Server

**Persistent cross-session project memory for CLI coding agents using Google Gemini File Search stores.**

> "You are the hands; I am the architect." — Now the architect has a memory.

## What This Does

This MCP server gives CLI coding agents (Claude Code, Gemini CLI, etc.) **persistent memory across sessions** by:

1. **Saving** session summaries to per-project Gemini File Search stores (RAG)
2. **Retrieving** relevant context at the start of each new session
3. **Indexing** key project files for future reference

Instead of starting blind every session and grep-ing through files, the agent gets a semantically-searched summary of everything done before — webhooks, schemas, decisions, file paths, architecture choices — in as few as 1-2k tokens.

## How It Works

```
Session 1: Build landing page
    → Agent saves summary to Gemini store: endpoints, components, decisions

Session 2: Build backend API
    → Agent retrieves context: "What frontend endpoints exist?"
    → Gets: landing page routes, component structure, auth decisions
    → Builds backend with full awareness of Session 1
    → Saves its own summary

Session 3: Add webhooks
    → Agent retrieves: full project history
    → Knows both frontend and backend architecture
    → Works 10x faster with zero ramp-up time
```

## Cost

Extremely cheap:
- **Storage**: FREE
- **Embedding at query time**: FREE
- **Initial indexing**: $0.15 per 1M tokens (pennies per project)
- **Retrieved tokens**: Normal Gemini input token pricing

A typical session summary is ~500-2000 tokens. You could save 1000 sessions for about $0.15.

## Prerequisites

- Python 3.10+
- A [Google Gemini API key](https://aistudio.google.com/apikey)
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI installed

## Installation

```powershell
# Clone the repo
git clone https://github.com/Alek-Cel/CLI-Unlimited-Memory.git
cd CLI-Unlimited-Memory

# Run the installer
./install.ps1 -ApiKey "your-gemini-api-key"
```

That's it. The installer handles everything:
- Creates a Python venv and installs dependencies
- Deploys files to `~/tools/gemini-memory-mcp/`
- Registers the MCP server with Claude Code
- Creates the `gmem` CLI command
- Configures compaction hooks (auto-save/restore context)

**Windows (PowerShell 5.1+):** Works out of the box.

**macOS/Linux (PowerShell 7+):** Install PowerShell first: `brew install powershell` or see [Microsoft docs](https://learn.microsoft.com/en-us/powershell/scripting/install/installing-powershell).

### Installer Options

| Flag | Description |
|------|-------------|
| `-ApiKey "key"` | Gemini API key (skips auto-detection / prompt) |
| `-InstallDir "path"` | Custom install location (default: `~/tools/gemini-memory-mcp`) |
| `-SkipHooks` | Don't configure Claude Code compaction hooks |
| `-SkipTest` | Don't run post-install verification |
| `-Uninstall` | Remove everything (see below) |

### API Key Resolution

If you don't pass `-ApiKey`, the installer looks for an existing key in this order:
1. Existing `claude mcp get gemini-memory` registration
2. `GEMINI_API_KEY` environment variable
3. Interactive prompt

This means re-running the installer after `git pull` preserves your API key automatically.

## Update

```powershell
cd CLI-Unlimited-Memory
git pull
./install.ps1
```

The installer detects your existing API key from the previous MCP registration, refreshes all files and dependencies, and re-registers the server.

## Uninstall

```powershell
./install.ps1 -Uninstall
```

Removes:
- MCP server registration from Claude Code
- Compaction hooks from `~/.claude/settings.json`
- Wrapper scripts (`gmem`, `gmem-precompact`, `gmem-postcompact`)
- Install directory (`~/tools/gemini-memory-mcp/`)

Your Gemini memory stores are **not** deleted — they live in your Google account and persist independently.

## Available MCP Tools

| Tool | Description |
|------|-------------|
| `gemini_memory_create_store` | Create a new project memory store |
| `gemini_memory_list_stores` | List all project memory stores |
| `gemini_memory_save_session` | Save a session summary to a project store |
| `gemini_memory_retrieve_context` | Retrieve relevant context from project memory |
| `gemini_memory_upload_file` | Upload a file to a project store for indexing |
| `gemini_memory_list_documents` | List all documents in a project store |
| `gemini_memory_delete_document` | Delete a document from a project store |
| `gemini_memory_consolidate` | Merge old sessions into a single summary |
| `gemini_memory_delete_store` | Delete an entire project store (irreversible) |

## CLI Tool (`gmem`)

The installer also sets up `gmem`, a standalone CLI that works with **any** coding agent (Claude Code, Gemini CLI, Cursor, Aider, etc.).

```bash
gmem init my-website                    # Create a project store
gmem stores                             # List all stores

# Save sessions
gmem save my-website --label auth-v1 -m "Built JWT auth with refresh tokens"
gmem save my-website                    # Interactive editor
echo "summary..." | gmem save my-site   # Piped from script

# Retrieve context
gmem context my-website "auth flow and endpoints"
gmem context my-website "database schema" --tokens 2000

# Auto-save from git state (great for hooks!)
gmem auto-save my-website . --label session-3

# Inject context into agent workflows
gmem inject my-website "full architecture" --output .context.md
gmem inject my-website "recent changes" --clipboard

# Consolidate old sessions
gmem consolidate my-website --keep 5

# File management
gmem upload my-website ./schema.sql
gmem docs my-website
gmem delete-doc my-website "old-session"
gmem delete my-website
```

### Usage with Any Agent

**Gemini CLI** — pipe context as initial prompt:
```bash
gmem inject my-website "what was done last" | gemini "continue from where we left off"
```

**Git hook** — auto-save after each commit:
```bash
# .git/hooks/post-commit
#!/bin/bash
gmem auto-save my-website . --label "commit-$(git rev-parse --short HEAD)"
```

**Any agent** — wrap sessions:
```bash
# Before session
gmem inject my-website "full context" --output .context.md

# ... do your work ...

# After session
gmem save my-website --label "feature-x" -m "Built feature X with endpoints /api/x..."
```

## CLAUDE.md Integration

To make memory fully automatic, copy `CLAUDE-MD-SNIPPET.md` into your global `~/.claude/CLAUDE.md` or project-level `CLAUDE.md`. This tells Claude Code to:
- Retrieve context at the start of every session
- Save session summaries automatically when tasks complete
- Upload important files to memory

## Compaction Hooks

The installer configures Claude Code hooks so memory survives context compaction:

- **PreCompact**: Summarizes the full conversation transcript and saves it to Gemini Memory
- **SessionStart (compact)**: After compaction, retrieves and injects project context back into the session

This means you never lose detailed context when Claude Code compacts your conversation.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | (required) | Your Google Gemini API key |
| `GEMINI_RETRIEVAL_MODEL` | `gemini-3-flash-preview` | Model used for context retrieval |
| `GEMINI_SUMMARY_MODEL` | `gemini-3-flash-preview` | Model used for transcript summarization |
| `GEMINI_MEMORY_MAX_TOKENS` | `5000` | Default max tokens for retrieved context |
| `GEMINI_STORE_PREFIX` | `project-memory` | Prefix for store display names |

## Adjusting Context Size

The `max_output_tokens` parameter on `retrieve_context` controls how much context is returned:

- **1000-2000 tokens**: Quick overview — good for "what was the last thing done?"
- **3000-5000 tokens**: Balanced — good for most sessions, covers key architecture
- **5000-8000 tokens**: Detailed — good for complex projects with many moving parts
- **8000-16000 tokens**: Comprehensive — full project history retrieval

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│ Claude Code  │────▶│  MCP Server      │────▶│ Gemini File Search   │
│ or any CLI   │◀────│  (this tool)     │◀────│ Store (per project)  │
│ agent        │     │                  │     │                      │
│              │     │  - save_session  │     │  - Session logs      │
│              │     │  - retrieve      │     │  - Uploaded files    │
│              │     │  - upload_file   │     │  - Semantic index    │
└─────────────┘     └──────────────────┘     └──────────────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Gemini Flash │
                    │ (retrieval)  │
                    └──────────────┘
```

## Limitations

- Maximum 10 File Search stores per Google AI project (plan store usage)
- Individual stores recommended under 20GB
- Documents are immutable once indexed (delete + re-upload to update)
- File Search store data persists indefinitely (unlike raw Files API which deletes after 48h)

## License

MIT
