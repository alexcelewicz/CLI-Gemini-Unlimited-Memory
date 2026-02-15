---

## Gemini Project Memory (Automated)

This system uses the `gemini-memory` MCP server for persistent cross-session project memory.
The store name for each project is derived from the **project folder name** (the last segment of the working directory path, lowercased, spaces replaced with hyphens).

### Determining the Project Name
- Extract the folder name from the current working directory path
- Example: `E:\Vibe_Coding\MaiaChat_New` -> project name is `maiachat_new`
- Example: `E:\Vibe_Coding\VerbScribe_NEW` -> project name is `verbscribe_new`
- Example: `C:\Projects\My Cool App` -> project name is `my-cool-app`

### At the START of every session (MANDATORY - do this BEFORE any other work):
1. Determine the project name from the current working directory
2. Call `gemini_memory_create_store` with the project name (this is idempotent - safe to call every time, returns "already_exists" if it exists)
3. Call `gemini_memory_retrieve_context` with:
   - `project_name`: the derived project name
   - `query`: "full project architecture, recent changes, current status, known bugs, and next steps"
   - `max_output_tokens`: 16000
4. Use the retrieved context to understand all prior work before proceeding
5. Briefly mention to the user what context was retrieved (e.g., "Retrieved memory from X sessions covering Y and Z"), or "No prior memory found - this is a new project store" if empty

### Mid-session targeted retrieval:
When you need specific context during work (not the broad session-start retrieval), use a focused query with lower tokens:
- Call `gemini_memory_retrieve_context` with a specific query and `max_output_tokens`: 5000
- Example queries: "authentication flow and JWT setup", "database schema for contacts table", "what bugs were found in channels"

### At the END of every task/session (AUTOMATIC - do this without being asked):
When the user's task is complete (code written, bug fixed, feature implemented, etc.), automatically save the session:
1. Call `gemini_memory_save_session` with:
   - `project_name`: the derived project name
   - `session_label`: a short descriptive label for what was done (e.g., "auth-backend-v2", "fix-webhook-duplicates")
   - `summary`: a structured summary including ALL of the following:

```
## What Was Built/Changed
- [List all files created or modified with their paths]
- [Describe each change concisely]

## Key Decisions
- [Architecture decisions and their rationale]
- [Technology choices made]

## Technical Details
- API Endpoints: [any new/modified endpoints]
- Database/Schema: [any schema changes]
- Dependencies: [new packages installed]
- Environment Variables: [new env vars]

## Current Status
- [What works now]
- [What's partially done]
- [Known bugs or issues]

## Next Steps
- [What should be done next]
- [Open questions]
```

### Key File Uploads
When creating or significantly modifying important reference files, upload them to memory:
- Database schemas, API docs, config files -> `gemini_memory_upload_file`

### Post-Compaction Context Restoration
When context compaction occurs, hooks automatically:
1. **Before compaction:** Save a detailed session summary to Gemini Memory
2. **After compaction:** Retrieve and inject that context back into the session

If you see "GEMINI MEMORY CONTEXT RESTORED" in your context after compaction, you have
detailed context beyond the compacted summary. Use it to continue working seamlessly —
it contains file paths, decisions, bugs, and next steps from the pre-compaction session.

### Rules
- Do NOT ask the user "should I save to memory?" - just do it automatically
- Do NOT ask the user "should I retrieve context?" - just do it automatically at session start
- If retrieval returns no results (new project), mention it and proceed normally
- Keep session labels short and descriptive (max 3-4 words, kebab-case)
- The 10 store limit per Google AI project means plan store usage across projects

---
