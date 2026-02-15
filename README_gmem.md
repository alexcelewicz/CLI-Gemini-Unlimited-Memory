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
- Claude Code or any MCP-compatible client

## Installation

### 1. Clone/copy the server

```bash
# Put it wherever you keep your tools
mkdir -p ~/tools/gemini-memory-mcp
cp server.py requirements.txt ~/tools/gemini-memory-mcp/
```

### 2. Set up Python environment

```bash
cd ~/tools/gemini-memory-mcp
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Set your API key

```bash
# Windows (PowerShell)
$env:GEMINI_API_KEY = "your-key-here"

# Or set it permanently in your environment variables

# macOS/Linux
export GEMINI_API_KEY="your-key-here"
# Or add to ~/.bashrc / ~/.zshrc
```

### 4. Configure Claude Code

Add to your Claude Code MCP config (`~/.claude/claude_desktop_config.json` or project-level `.mcp.json`):

```json
{
  "mcpServers": {
    "gemini-memory": {
      "command": "python",
      "args": ["C:/Users/YOUR_USER/tools/gemini-memory-mcp/server.py"],
      "env": {
        "GEMINI_API_KEY": "your-key-here",
        "GEMINI_RETRIEVAL_MODEL": "gemini-2.5-flash",
        "GEMINI_MEMORY_MAX_TOKENS": "5000"
      }
    }
  }
}
```

**For Windows with venv**, use the full path to the venv Python:

```json
{
  "mcpServers": {
    "gemini-memory": {
      "command": "C:/Users/YOUR_USER/tools/gemini-memory-mcp/.venv/Scripts/python.exe",
      "args": ["C:/Users/YOUR_USER/tools/gemini-memory-mcp/server.py"],
      "env": {
        "GEMINI_API_KEY": "your-key-here"
      }
    }
  }
}
```

## Available Tools

| Tool | Description |
|------|-------------|
| `gemini_memory_create_store` | Create a new project memory store |
| `gemini_memory_list_stores` | List all project memory stores |
| `gemini_memory_save_session` | Save a session summary to a project store |
| `gemini_memory_retrieve_context` | Retrieve relevant context from project memory |
| `gemini_memory_upload_file` | Upload a file to a project store for indexing |
| `gemini_memory_list_documents` | List all documents in a project store |
| `gemini_memory_delete_document` | Delete a document from a project store |
| `gemini_memory_delete_store` | Delete an entire project store (irreversible) |

## Usage Workflow

### Starting a New Project

```
You: "Create a memory store for my verbscribe-website project"
Agent: → calls gemini_memory_create_store(project_name="verbscribe-website")
```

### End of Session — Save Context

```
You: "Save this session to project memory"
Agent: → Generates structured summary of what was done
      → calls gemini_memory_save_session(
            project_name="verbscribe-website",
            session_label="landing-page-v1",
            summary="## What was built\n- Landing page at /src/pages/index.tsx\n- Hero component with CTA..."
         )
```

### Start of Next Session — Retrieve Context

```
You: "Retrieve context about the frontend architecture"
Agent: → calls gemini_memory_retrieve_context(
            project_name="verbscribe-website",
            query="frontend architecture, components, routing, and API endpoints"
         )
      → Gets back: full summary of landing page work, component structure, etc.
      → Continues working with full awareness
```

### Upload Key Files

```
You: "Upload the database schema to project memory"
Agent: → calls gemini_memory_upload_file(
            project_name="verbscribe-website",
            file_path="/path/to/schema.sql"
         )
```

## Tips for CLAUDE.md Integration

Add this to your project's `CLAUDE.md` to make the memory automatic:

```markdown
## Project Memory (Gemini File Search)

This project uses persistent memory via the Gemini Memory MCP server.

### At the START of every session:
1. Call `gemini_memory_retrieve_context` with project_name="YOUR_PROJECT"
   and a query describing what you're about to work on
2. Use the retrieved context to understand prior work

### At the END of every session:
1. Call `gemini_memory_save_session` with a structured summary including:
   - What was built/changed (with file paths)
   - Key decisions and their rationale
   - API endpoints, schemas, hooks used
   - Current status and next steps
   - Any bugs found or issues noted

### When creating key files:
- Upload important files (schemas, configs, READMEs) with `gemini_memory_upload_file`
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | (required) | Your Google Gemini API key |
| `GEMINI_RETRIEVAL_MODEL` | `gemini-2.5-flash` | Model used for context retrieval |
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

---

## Option B: Standalone CLI Tool (`gmem`)

If you prefer a standalone CLI over MCP, or want something that works with **any** agent (Gemini CLI, Cursor, Aider, etc.), use `gmem.py`.

### Setup

```bash
cd ~/tools/gemini-memory-mcp
python -m venv .venv
.venv\Scripts\activate           # Windows
pip install -r requirements.txt

# Windows: run setup-gmem.bat
# Linux/macOS: bash setup-gmem.sh
```

### Commands

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

# File management
gmem upload my-website ./schema.sql
gmem docs my-website
gmem delete-doc my-website "old-session"
gmem delete my-website
```

### Usage with Any Agent

**Claude Code** — inject context before starting:
```bash
gmem inject my-website "full project context" --output .context.md
# Then in Claude Code, the .context.md file is available
```

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

### Option A vs Option B — When to Use Which

| | MCP Server (Option A) | CLI Tool (Option B) |
|---|---|---|
| **Best for** | Claude Code (native integration) | Any agent, scripting, git hooks |
| **Auto-integration** | Agent calls tools directly | Manual or scripted |
| **Save/retrieve** | Agent-initiated | You trigger it |
| **Git hooks** | ❌ | ✅ `gmem auto-save` |
| **Clipboard inject** | ❌ | ✅ `gmem inject --clipboard` |
| **File output** | ❌ | ✅ `gmem inject --output` |
| **Piping** | ❌ | ✅ Full stdin/stdout support |

**You can use both simultaneously** — they share the same Gemini stores.

---

## Limitations

- Maximum 10 File Search stores per Google AI project (plan store usage)
- Individual stores recommended under 20GB
- Documents are immutable once indexed (delete + re-upload to update)
- File Search store data persists indefinitely (unlike raw Files API which deletes after 48h)

## License

MIT
