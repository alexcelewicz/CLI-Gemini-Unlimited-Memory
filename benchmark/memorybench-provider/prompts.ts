/**
 * Custom answer prompt for Gemini File Search provider.
 *
 * Implements two key techniques from competitive analysis:
 * - Dual-format handling: raw chunks (PRIMARY) vs synthesis (SECONDARY)
 * - Temporal reasoning: absolute date resolution from document timestamps
 */

/**
 * Build a context string from search results, separating raw chunks from synthesis.
 */
function formatContext(context: unknown[]): string {
  const rawChunks: string[] = [];
  let synthesis = "";

  for (const item of context) {
    const entry = item as Record<string, unknown>;
    if (entry.type === "raw_chunk" && entry.text) {
      rawChunks.push(String(entry.text));
    } else if (entry.type === "synthesis" && entry.text) {
      synthesis = String(entry.text);
    }
  }

  const parts: string[] = [];

  if (rawChunks.length > 0) {
    parts.push("=== PRIMARY SOURCE: Raw Document Chunks ===");
    parts.push("(These are exact excerpts from stored sessions. Prefer these for factual answers.)\n");
    for (let i = 0; i < rawChunks.length; i++) {
      parts.push(`--- Chunk ${i + 1} ---`);
      parts.push(rawChunks[i]);
      parts.push("");
    }
  }

  if (synthesis) {
    parts.push("=== SECONDARY SOURCE: AI-Synthesized Summary ===");
    parts.push("(Use this for overview/context, but prefer raw chunks when they contain the answer.)\n");
    parts.push(synthesis);
  }

  return parts.join("\n");
}

/**
 * Answer prompt function for memorybench.
 *
 * @param question - The question to answer
 * @param context - Array of search results (raw_chunk and synthesis entries)
 * @param questionDate - Optional date context for temporal reasoning
 */
export function answerPrompt(
  question: string,
  context: unknown[],
  questionDate?: string
): string {
  const formattedContext = formatContext(context);
  const dateInstruction = questionDate
    ? `The question is being asked as of ${questionDate}. Use this to resolve relative time references.`
    : "If relative time references appear (e.g., 'yesterday', 'last week'), resolve them using document timestamps.";

  return `You are answering a question based on stored conversation memory.

CONTEXT FROM MEMORY:
${formattedContext}

INSTRUCTIONS:
1. Answer the question based ONLY on the context above.
2. Raw document chunks (PRIMARY) take precedence over the synthesized summary (SECONDARY) when they conflict.
3. If raw chunks contain the exact answer, use that verbatim rather than the synthesis.
4. If the context does not contain enough information to answer, say so clearly.

TEMPORAL REASONING:
- ${dateInstruction}
- Each document chunk may have a timestamp in its header (e.g., "**Timestamp:** 2025-03-15").
- Convert relative time references to absolute dates based on the document's timestamp.
- When information conflicts across documents, the most recent document is authoritative.

ANSWER FORMAT:
- Be direct and concise.
- Answer the specific question asked — do not add extra information.
- If the answer is a name, date, number, or specific fact, state it plainly.

QUESTION: ${question}`;
}
