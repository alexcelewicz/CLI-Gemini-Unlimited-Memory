# Gemini File Search — MemoryBench Provider

Adapter for benchmarking Google Gemini File Search stores with [memorybench](https://github.com/supermemoryai/memorybench).

## Setup

### 1. Clone memorybench

```bash
git clone https://github.com/supermemoryai/memorybench.git
cd memorybench
bun install
```

### 2. Install Gemini SDK

```bash
bun add @google/genai
```

### 3. Copy provider files

```bash
mkdir -p src/providers/gemini-filesearch
cp /path/to/this/index.ts src/providers/gemini-filesearch/index.ts
cp /path/to/this/prompts.ts src/providers/gemini-filesearch/prompts.ts
```

### 4. Register the provider

**`src/types/provider.ts`** — Add to the `ProviderName` union:

```diff
-export type ProviderName = "supermemory" | "mem0" | "zep" | "filesystem" | "rag"
+export type ProviderName = "supermemory" | "mem0" | "zep" | "filesystem" | "rag" | "gemini-filesearch"
```

**`src/providers/index.ts`** — Add import and registration:

```diff
+import { GeminiFileSearchProvider } from "./gemini-filesearch"

 const providers: Record<ProviderName, new () => Provider> = {
   supermemory: SupermemoryProvider,
   mem0: Mem0Provider,
   zep: ZepProvider,
   filesystem: FilesystemProvider,
   rag: RAGProvider,
+  "gemini-filesearch": GeminiFileSearchProvider,
 }
```

And add to the exports:

```diff
-export { SupermemoryProvider, Mem0Provider, ZepProvider, FilesystemProvider, RAGProvider }
+export { SupermemoryProvider, Mem0Provider, ZepProvider, FilesystemProvider, RAGProvider, GeminiFileSearchProvider }
```

### 5. Set API key

Add to `.env.local`:

```
GEMINI_API_KEY=your-api-key-here
```

## Usage

### Test a single question

```bash
bun run src/index.ts test -p gemini-filesearch -b locomo
```

### Run full benchmark

```bash
bun run src/index.ts run -p gemini-filesearch -b locomo -j google
```

### Compare providers

```bash
bun run src/index.ts run -p gemini-filesearch,supermemory,mem0 -b locomo -j google
```

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `GEMINI_API_KEY` | — | Google AI API key (required) |

The provider uses `gemini-2.5-flash` for retrieval by default. To override, pass `model` in the provider config.

## How It Works

1. **Ingest**: Creates a Gemini File Search store per `containerTag`, converts each `UnifiedSession` to markdown preserving speaker labels and timestamps, uploads as documents
2. **Indexing**: Polls operations with exponential backoff (1s to 5s, 120s timeout)
3. **Search**: Calls `generateContent` with `fileSearch` tool, returns dual-format results (raw grounding chunks + synthesized text)
4. **Clear**: Deletes the File Search store and all its documents

## Concurrency Defaults

| Phase | Concurrency |
|-------|------------|
| Default | 10 |
| Ingest | 5 |
| Indexing | 1 |
