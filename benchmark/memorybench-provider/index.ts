/**
 * Gemini File Search Provider for MemoryBench
 * =============================================
 * Benchmarks Google Gemini File Search stores as a memory provider.
 * Uses the @google/genai SDK to create stores, upload session documents,
 * and perform grounded retrieval via generateContent + fileSearch tool.
 *
 * STORE DEDUPLICATION: Multiple questions share the same haystack sessions.
 * Instead of creating one store per containerTag (1986 stores!), we create
 * one store per unique session set and map all related containerTags to it.
 */

import { GoogleGenAI } from "@google/genai";
import type {
  Provider,
  ProviderConfig,
  IngestOptions,
  IngestResult,
  SearchOptions,
  IndexingProgressCallback,
} from "../../types/provider";
import type { ConcurrencyConfig } from "../../types/concurrency";
import type { ProviderPrompts } from "../../types/prompts";
import type { UnifiedSession } from "../../types/unified";
import { answerPrompt } from "./prompts";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Convert a UnifiedSession to a markdown document suitable for File Search. */
function sessionToMarkdown(session: UnifiedSession): string {
  const lines: string[] = [];
  lines.push(`# Session: ${session.sessionId}`);

  if (session.metadata) {
    const ts = session.metadata.timestamp || session.metadata.date;
    if (ts) lines.push(`**Date:** ${ts}`);
  }
  lines.push("");

  for (const msg of session.messages) {
    const speaker = msg.speaker || msg.role;
    const ts = msg.timestamp ? ` (${msg.timestamp})` : "";
    lines.push(`**${speaker}${ts}:**`);
    lines.push(msg.content);
    lines.push("");
  }

  return lines.join("\n");
}

/** Sleep for ms milliseconds. */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Derive a stable store key from a containerTag.
 * containerTag format: "{questionId}-{dataSourceRunId}"
 * The conversation prefix (e.g. "conv-0") groups questions that share sessions.
 * We extract the conversation prefix + runId to create a shared store key.
 */
function deriveStoreKey(containerTag: string): string {
  // containerTag = "conv-0-q5-abc123" -> conversation = "conv-0", runSuffix = "abc123"
  // We want all questions from the same conversation to share one store.
  const parts = containerTag.split("-");

  // Find the "-q" part to split conversation prefix from question + runId
  const qIndex = parts.findIndex((p) => p.startsWith("q") && /^q\d+$/.test(p));
  if (qIndex >= 2) {
    // conversation prefix = parts before qIndex, runId = parts after qIndex
    const convPrefix = parts.slice(0, qIndex).join("-");
    const runId = parts.slice(qIndex + 1).join("-");
    return `${convPrefix}-${runId}`;
  }

  // Fallback: use first two segments as conversation key
  // e.g., "conv-0-q5-runid" -> "conv-0-runid"
  return containerTag;
}

/** Retry a function with exponential backoff on 429 errors. */
async function withRetry<T>(
  fn: () => Promise<T>,
  maxRetries = 5,
  baseDelay = 2000
): Promise<T> {
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (e: any) {
      const is429 =
        e?.status === 429 ||
        e?.code === 429 ||
        String(e?.message || e).includes("429") ||
        String(e?.message || e).includes("RESOURCE_EXHAUSTED");

      if (is429 && attempt < maxRetries) {
        const delay = baseDelay * Math.pow(2, attempt);
        await sleep(delay);
        continue;
      }
      throw e;
    }
  }
  throw new Error("Unreachable");
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

export class GeminiFileSearchProvider implements Provider {
  name = "gemini-filesearch";

  prompts: ProviderPrompts = {
    answerPrompt,
  };

  concurrency: ConcurrencyConfig = {
    default: 10,
    ingest: 3, // Conservative to avoid rate limits
    indexing: 1,
  };

  private client!: GoogleGenAI;
  private model = "gemini-2.5-flash";

  // containerTag -> store resource name (many-to-one via dedup)
  private stores = new Map<string, string>();

  // storeKey -> store resource name (one-to-one, the actual stores)
  private sharedStores = new Map<string, string>();

  // storeKey -> Set of already-uploaded sessionIds (avoid re-uploading)
  private uploadedSessions = new Map<string, Set<string>>();

  // storeKey -> Set of containerTags using this store (for cleanup refcounting)
  private storeRefs = new Map<string, Set<string>>();

  // Lock to prevent concurrent store creation for the same storeKey
  private storeCreationLocks = new Map<string, Promise<string>>();

  async initialize(config: ProviderConfig): Promise<void> {
    const apiKey = config.apiKey || process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error(
        "Gemini API key required. Set GEMINI_API_KEY env var or pass apiKey in config."
      );
    }
    this.client = new GoogleGenAI({ apiKey });

    if (config.model && typeof config.model === "string") {
      this.model = config.model;
    }
  }

  /** Get or create a shared store for a given storeKey. Thread-safe via lock. */
  private async getOrCreateStore(storeKey: string): Promise<string> {
    // Return existing store
    const existing = this.sharedStores.get(storeKey);
    if (existing) return existing;

    // Check if another call is already creating this store
    const pendingCreation = this.storeCreationLocks.get(storeKey);
    if (pendingCreation) return pendingCreation;

    // Create the store (with lock to prevent duplicates)
    const creationPromise = (async () => {
      const displayName = `memorybench-${storeKey}`;
      const store = await withRetry(() =>
        this.client.fileSearchStores.create({
          config: { displayName },
        })
      );

      const storeName = store.name!;
      this.sharedStores.set(storeKey, storeName);
      this.uploadedSessions.set(storeKey, new Set());
      this.storeRefs.set(storeKey, new Set());
      this.storeCreationLocks.delete(storeKey);
      return storeName;
    })();

    this.storeCreationLocks.set(storeKey, creationPromise);
    return creationPromise;
  }

  async ingest(
    sessions: UnifiedSession[],
    options: IngestOptions
  ): Promise<IngestResult> {
    const { containerTag } = options;
    const storeKey = deriveStoreKey(containerTag);

    // Get or create the shared store for this conversation
    const storeName = await this.getOrCreateStore(storeKey);

    // Map this containerTag to the shared store
    this.stores.set(containerTag, storeName);
    this.storeRefs.get(storeKey)!.add(containerTag);

    const uploaded = this.uploadedSessions.get(storeKey)!;
    const documentIds: string[] = [];
    const taskIds: string[] = [];

    for (const session of sessions) {
      // Skip sessions already uploaded to this store
      if (uploaded.has(session.sessionId)) {
        documentIds.push(`session-${session.sessionId}`);
        continue;
      }

      const markdown = sessionToMarkdown(session);
      const blob = new Blob([markdown], { type: "text/markdown" });

      const operation = await withRetry(() =>
        this.client.fileSearchStores.uploadToFileSearchStore({
          fileSearchStoreName: storeName,
          file: blob,
          config: {
            displayName: `session-${session.sessionId}`,
          },
        })
      );

      uploaded.add(session.sessionId);

      if (operation.name) {
        taskIds.push(operation.name);
      }
      documentIds.push(`session-${session.sessionId}`);
    }

    return { documentIds, taskIds };
  }

  async awaitIndexing(
    result: IngestResult,
    containerTag: string,
    onProgress?: IndexingProgressCallback
  ): Promise<void> {
    if (!result.taskIds || result.taskIds.length === 0) return;

    const completedIds: string[] = [];
    const failedIds: string[] = [];
    const total = result.taskIds.length;
    const pending = new Set(result.taskIds);

    const timeout = 120_000;
    const start = Date.now();
    let backoff = 1000;

    while (pending.size > 0) {
      if (Date.now() - start > timeout) {
        for (const id of pending) {
          failedIds.push(id);
        }
        break;
      }

      for (const opName of [...pending]) {
        try {
          const op = await this.client.operations.get({
            operation: { name: opName } as any,
          });
          if (op.done) {
            pending.delete(opName);
            completedIds.push(opName);
          }
        } catch {
          pending.delete(opName);
          failedIds.push(opName);
        }
      }

      if (onProgress) {
        onProgress({ completedIds, failedIds, total });
      }

      if (pending.size > 0) {
        await sleep(backoff);
        backoff = Math.min(backoff * 1.5, 5000);
      }
    }

    if (onProgress) {
      onProgress({ completedIds, failedIds, total });
    }
  }

  async search(query: string, options: SearchOptions): Promise<unknown[]> {
    const storeName = this.stores.get(options.containerTag);
    if (!storeName) {
      throw new Error(
        `No store found for containerTag '${options.containerTag}'. Was ingest() called?`
      );
    }

    const response = await withRetry(() =>
      this.client.models.generateContent({
        model: this.model,
        contents: query,
        config: {
          tools: [
            {
              fileSearch: {
                fileSearchStoreNames: [storeName],
              },
            },
          ],
          maxOutputTokens: 4096,
        },
      })
    );

    const results: unknown[] = [];

    const candidate = response.candidates?.[0];
    const gm = candidate?.groundingMetadata;
    if (gm?.groundingChunks) {
      for (const chunk of gm.groundingChunks) {
        const rc = (chunk as Record<string, unknown>).retrievedContext as
          | Record<string, unknown>
          | undefined;
        if (rc) {
          results.push({
            type: "raw_chunk",
            text: rc.text || "",
            uri: rc.uri || "",
            title: rc.title || "",
          });
        }
      }
    }

    if (response.text) {
      results.push({
        type: "synthesis",
        text: response.text,
      });
    }

    return results;
  }

  async clear(containerTag: string): Promise<void> {
    const storeName = this.stores.get(containerTag);
    if (!storeName) return;

    const storeKey = deriveStoreKey(containerTag);
    const refs = this.storeRefs.get(storeKey);

    // Remove this containerTag's reference
    this.stores.delete(containerTag);
    refs?.delete(containerTag);

    // Only delete the actual store when no more containerTags reference it
    if (!refs || refs.size === 0) {
      try {
        await this.client.fileSearchStores.delete({
          name: storeName,
          config: { force: true },
        });
      } catch {
        // Already deleted
      }
      this.sharedStores.delete(storeKey);
      this.uploadedSessions.delete(storeKey);
      this.storeRefs.delete(storeKey);
    }
  }
}
