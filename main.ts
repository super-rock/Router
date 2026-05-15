// ============================================================
// PERSONAL AI ROUTER  v7.1
//
// v7.1 — surgical fixes on top of v7. What actually changed:
//
//   BUGS FIXED
//   1. Duplicate "[N files attached]" note. Both the frontend persist()
//      and the server handleChatSave() were appending the note. Each save
//      doubled it; each regenerate quadrupled it. Now only the server
//      adds the note, with an idempotency check so it's never doubled.
//   2. Stop button reinstated. Aborts the in-flight request cleanly;
//      partial content is kept and tagged "stopped". Removing it in v7
//      was a mistake — you can't wait on a runaway response.
//   3. PDF + Llama Vision fallback was silent garbage. Llama Vision can't
//      decode PDFs. The vision chain now skips non-PDF-capable steps when
//      a PDF is attached; if Gemini fails, an explicit error is returned.
//   4. Search/Reason intent now sticks across retry and regenerate. They
//      were resetting to false after every send, so retrying an errored
//      search request silently dropped the search.
//   5. End-of-stream flicker fixed. Old code called renderAll() in the
//      finally block, which wiped and rebuilt all bubbles. Now we just
//      append the action buttons to the live bubble — no flicker.
//   6. Memory name regex was matching too loosely (case-insensitive
//      [A-Z], plus first-person patterns that don't appear in stored
//      third-person memories). Tightened to third-person patterns only.
//   7. healthCache is now refreshed every time Settings opens, so adding
//      a provider key in Deno Deploy shows up immediately.
//   8. Sub-1KB files no longer display "0 KB" — they show "< 1 KB".
//
//   FEATURES
//   9. Manual memory entry in Settings — type a memory, hit +, and it's
//      saved. No longer dependent on the auto-extractor catching things.
//  10. Routing label during early stream. When the status event arrives
//      but tokens haven't flowed yet, the bubble shows "Routing via X…"
//      instead of just three dots. Useful when the first provider is slow.
//
//   POLISH
//  11. Respects prefers-reduced-motion: aurora drift, dot pulse, message
//      fade-in, and dot animations all freeze when the OS setting is on.
//  12. /health version bumped to "v7.1".
//
//   CARRIED OVER FROM v7 (unchanged)
//   - File uploads (images + PDFs + text/code) with client-side extraction
//   - Smart streaming scroll with stick-to-bottom and scroll pill
//   - Skills architecture (core/code/document) with declarative triggers
//   - Aurora background, glass header, gradient buttons
//   - Strong system prompt with adult voice and no padding
//
// Daily capacity:
//   Cerebras  : 1M tokens/day, 30 RPM
//   Gemini    : 1,500 req/day, 15 RPM
//   OpenRouter: 50 req/day (free pool)
//   Groq      : ~100-500K tokens/day per model (optional, if key present)
//   Tavily    : 1,000 searches/month
//
// Env vars required (Deno Deploy → Settings → Environment Variables):
//   OPENROUTER_KEY, GOOGLE_AI_KEY, CEREBRAS_KEY, TAVILY_KEY
//   GROQ_KEY (optional)
//
// Fallback chains (auto-failover on 429/404/5xx, commit once tokens flow):
//   Chat        : Cerebras → Groq → Gemini → OpenRouter Auto-Free
//   Uncensored  : OpenRouter Venice (no fallback by design)
//   Reasoner    : OpenRouter Free → Cerebras Qwen Thinking → Gemini
//   Vision      : Gemini 2.5 Flash → OpenRouter Llama Vision
//
// Critical bug class to avoid (already fixed but documented for posterity):
// failed chat saves used to overwrite successful streamed responses with
// "Error: HTTP 500". Frontend persist() now strips images/PDFs before save
// (text-only goes to Deno KV; 64KB per-value limit). Save failures are
// silent (toast at most). Server /chats/save always returns HTTP 200.
// ============================================================

const OPENROUTER_KEY = Deno.env.get("OPENROUTER_KEY") || "";
const TAVILY_KEY     = Deno.env.get("TAVILY_KEY")     || "";
const GOOGLE_AI_KEY  = Deno.env.get("GOOGLE_AI_KEY")  || "";
const CEREBRAS_KEY   = Deno.env.get("CEREBRAS_KEY")   || "";
const GROQ_KEY       = Deno.env.get("GROQ_KEY")       || "";

const kv = await Deno.openKv();

// ------------------------------------------------------------
// Providers
// ------------------------------------------------------------
type Provider = "openrouter" | "cerebras" | "groq" | "gemini";

const PROVIDER_CONFIG: Record<Exclude<Provider, "gemini">, {
  endpoint: string;
  apiKey: () => string;
  headers: Record<string, string>;
}> = {
  openrouter: {
    endpoint: "https://openrouter.ai/api/v1/chat/completions",
    apiKey: () => OPENROUTER_KEY,
    headers: {
      "HTTP-Referer": "https://router.super-rock.deno.net",
      "X-Title": "Personal Router",
    },
  },
  cerebras: {
    endpoint: "https://api.cerebras.ai/v1/chat/completions",
    apiKey: () => CEREBRAS_KEY,
    headers: {},
  },
  groq: {
    endpoint: "https://api.groq.com/openai/v1/chat/completions",
    apiKey: () => GROQ_KEY,
    headers: {},
  },
};

function providerKey(p: Provider): string {
  if (p === "openrouter") return OPENROUTER_KEY;
  if (p === "cerebras")   return CEREBRAS_KEY;
  if (p === "groq")       return GROQ_KEY;
  if (p === "gemini")     return GOOGLE_AI_KEY;
  return "";
}

// ------------------------------------------------------------
// Model registry — chains
// ------------------------------------------------------------
type ProviderOption = { provider: Provider; id: string; label: string; pdfCapable?: boolean };
type ModelEntry = { label: string; blurb: string; vision: boolean; chain: ProviderOption[] };

const MODELS: Record<string, ModelEntry> = {
  chat: {
    label: "Chat (auto)",
    blurb: "Auto-routes Cerebras → Groq → Gemini → OpenRouter. Falls over instantly on rate limit.",
    vision: false,
    chain: [
      { provider: "cerebras",   id: "gpt-oss-120b",                  label: "GPT-OSS 120B [Cerebras]" },
      { provider: "groq",       id: "llama-3.3-70b-versatile",       label: "Llama 3.3 70B [Groq]" },
      { provider: "gemini",     id: "gemini-2.5-flash",              label: "Gemini 2.5 Flash" },
      { provider: "openrouter", id: "openrouter/free",               label: "OpenRouter Auto-Free" },
    ],
  },
  uncensored: {
    label: "Uncensored (Venice)",
    blurb: "Venice via OpenRouter. No content filter. ~50 req/day pool — use deliberately.",
    vision: false,
    chain: [
      { provider: "openrouter", id: "cognitivecomputations/dolphin-mistral-24b-venice-edition:free", label: "Venice (Uncensored)" },
    ],
  },
  reasoner: {
    label: "Deep Reasoner",
    blurb: "For hard multi-step problems. Auto-picks best available free reasoning model.",
    vision: false,
    chain: [
      { provider: "openrouter", id: "openrouter/free",                  label: "OR Auto-Free Reasoner" },
      { provider: "cerebras",   id: "qwen-3-235b-a22b-thinking-2507",   label: "Qwen 3 Thinking [Cerebras]" },
      { provider: "gemini",     id: "gemini-2.5-flash",                 label: "Gemini Fallback" },
    ],
  },
  vision: {
    label: "Vision / Documents",
    blurb: "Auto-engaged when you attach an image or PDF.",
    vision: true,
    chain: [
      // Gemini is the only PDF-capable option here; Llama Vision can't decode PDFs.
      { provider: "gemini",     id: "gemini-2.5-flash",                                      label: "Gemini 2.5 Vision",  pdfCapable: true },
      { provider: "openrouter", id: "meta-llama/llama-3.2-11b-vision-instruct:free",         label: "Llama Vision",       pdfCapable: false },
    ],
  },
};

const DEFAULT_LEAD = "chat";
const REASONER     = "reasoner";
const VISION       = "vision";

// ------------------------------------------------------------
// Skills — declarative capability injection
// To add a new skill: append a Skill object to SKILLS. It auto-composes
// into the system prompt when its trigger fires. No core code changes.
// ------------------------------------------------------------
type SkillTrigger =
  | { type: "always" }
  | { type: "keyword"; keywords: string[] }
  | { type: "filetype"; mimePrefixes: string[] };

type Skill = {
  id: string;
  name: string;
  triggers: SkillTrigger[];
  promptAddition: string;
};

const SKILLS: Skill[] = [
  {
    id: "core",
    name: "Core voice",
    triggers: [{ type: "always" }],
    promptAddition: `You are this person's personal AI assistant. They built you for themselves. Speak directly.

# Voice
Sharp friend, not help desk. Real opinions. Warm without being eager. Confident without being smug. Dry humor welcome. Brief by default; longer when the question earns it. Disagree when you mean it — push back firmly but kindly. Real friends don't capitulate just to keep someone happy.

Never open with "Sure!", "Certainly!", "I'd be happy to", "Great question!", or "As an AI...". Don't ask permission to answer — just answer. Adults talking to adults: no moralizing, no caveats, no hedging unless you genuinely don't know.

# Honesty
If you don't know, say so. If you're guessing, label it. If the person is factually wrong about something, tell them and explain. Don't pad. Don't restate their question back to them. Don't summarize at the end unless it adds real value.

# Format
Markdown when it helps clarity (headers, lists, code blocks, bold for emphasis). Plain prose when the answer is short. Code ALWAYS goes in fenced blocks with language tags.`,
  },
  {
    id: "code",
    name: "Code mode",
    triggers: [
      { type: "keyword", keywords: ["code", "function", "bug", "debug", "error", "compile", "syntax", "regex", "stack trace", "exception", "typescript", "python", "javascript", "rust", "deploy", "docker", "git", "api", "library", "framework", "import ", "from ", "def ", "class ", "const ", "let ", "function(", "=>", "npm ", "pip ", "cargo "] },
      { type: "filetype", mimePrefixes: ["text/x-", "application/javascript", "application/typescript", "application/json", "application/xml", "application/x-sh", "application/x-python"] },
    ],
    promptAddition: `# Code mode is active
- Always use fenced blocks with language tags: \\\`\\\`\\\`python, \\\`\\\`\\\`typescript, \\\`\\\`\\\`bash, etc. Never bare indented code.
- Before the code: one line on what it does (skip if obvious).
- After the code: only explain non-obvious parts. Do NOT narrate line-by-line.
- Fixing a bug: name the bug first, show the fix, briefly say why. Don't restate the whole file unless asked.
- Prefer readable over clever. No unnecessary abstractions. No premature optimization.
- Ambiguous requirements → ask one clarifying question first. Don't guess silently.
- Error messages → translate to plain English, then show the fix.
- CLI/config tasks → command + one-line note on what it does.
- If they're stuck on something, identify the root cause, not just the symptom.`,
  },
  {
    id: "document",
    name: "Document reader",
    triggers: [
      { type: "filetype", mimePrefixes: ["application/pdf", "text/plain", "text/markdown", "text/csv", "text/html", "text/"] },
    ],
    promptAddition: `# Document mode is active
The user attached a document. Read it carefully and engage with the actual content.
- Answer specifically. Quote relevant passages.
- If their question isn't in the doc, say so.
- Don't summarize unless asked.
- If the doc has structure (sections, tables), respect it.`,
  },
];

function pickSkills(lastUserText: string, mimeTypes: string[]): Skill[] {
  const lc = (lastUserText || "").toLowerCase();
  return SKILLS.filter(skill =>
    skill.triggers.some(t => {
      if (t.type === "always") return true;
      if (t.type === "keyword") return t.keywords.some(k => lc.includes(k));
      if (t.type === "filetype") return mimeTypes.some(m => t.mimePrefixes.some(p => m.startsWith(p)));
      return false;
    })
  );
}

// ------------------------------------------------------------
// KV storage
// ------------------------------------------------------------
async function hashKey(syncKey: string): Promise<string> {
  const enc = new TextEncoder().encode(syncKey.trim().toLowerCase());
  const buf = await crypto.subtle.digest("SHA-256", enc);
  return Array.from(new Uint8Array(buf))
    .map(b => b.toString(16).padStart(2, "0"))
    .join("")
    .slice(0, 32);
}

async function listChats(uh: string) {
  const out: any[] = [];
  for await (const e of kv.list({ prefix: ["u", uh, "chats"] })) out.push(e.value);
  return out.sort((a, b) => {
    if (!!b.pinned !== !!a.pinned) return (b.pinned ? 1 : 0) - (a.pinned ? 1 : 0);
    return (b.updated || 0) - (a.updated || 0);
  });
}
const getChat = (uh: string, id: string) => kv.get(["u", uh, "chats", id]).then(r => r.value);
const putChat = (uh: string, c: any)     => kv.set(["u", uh, "chats", c.id], c);
const delChat = (uh: string, id: string) => kv.delete(["u", uh, "chats", id]);

const getMem = async (uh: string): Promise<string[]> =>
  (await kv.get<string[]>(["u", uh, "mem"])).value || [];
const putMem = (uh: string, m: string[]) =>
  kv.set(["u", uh, "mem"], Array.from(new Set(m.map(x => x.trim()).filter(Boolean))).slice(-150));

// ------------------------------------------------------------
// System prompt
// ------------------------------------------------------------
function timeGreeting(): string {
  const h = new Date().getUTCHours();
  return h < 11 ? "morning" : h < 17 ? "afternoon" : "evening";
}

function buildSystemPrompt(
  memories: string[],
  hasSearchContext: boolean,
  lastUserText: string,
  attachedMimeTypes: string[],
): string {
  const activeSkills = pickSkills(lastUserText, attachedMimeTypes);
  const skillBlocks = activeSkills.map(s => s.promptAddition).join("\n\n---\n\n");

  const memBlock = memories.length
    ? `\n\n# What you know about this person\n${memories.map(m => "- " + m).join("\n")}\n\nWeave in naturally only when relevant. Don't recite. Don't announce that you "remember" — just be someone who already knows them.`
    : "";

  const searchNote = hasSearchContext
    ? `\n\n# Note\nThe user's message includes fresh web search results. Use them to answer accurately. Cite sources at the end.`
    : "";

  return `${skillBlocks}${memBlock}${searchNote}\n\nToday is ${new Date().toISOString().slice(0, 10)} (${timeGreeting()} UTC).`;
}

// ------------------------------------------------------------
// Tavily search
// ------------------------------------------------------------
async function tavilySearch(query: string): Promise<string> {
  if (!TAVILY_KEY) return "Search unavailable: TAVILY_KEY not set on server.";
  try {
    const r = await fetch("https://api.tavily.com/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        api_key: TAVILY_KEY,
        query,
        max_results: 5,
        include_answer: true,
        search_depth: "basic",
      }),
    });
    if (!r.ok) return `Search failed (HTTP ${r.status}).`;
    const d = await r.json();
    let out = d.answer ? `Summary: ${d.answer}\n\n` : "";
    if (Array.isArray(d.results)) {
      out += "Sources:\n";
      for (const x of d.results) {
        out += `- ${x.title}\n  ${x.url}\n  ${(x.content || "").slice(0, 220)}\n`;
      }
    }
    return out.slice(0, 3500) || "(no results found)";
  } catch (e) {
    return `Search error: ${e}`;
  }
}

// ------------------------------------------------------------
// OpenAI-compatible (Cerebras, Groq, OpenRouter)
// ------------------------------------------------------------
async function callOAICompatible(
  provider: "openrouter" | "cerebras" | "groq",
  modelId: string,
  messages: any[],
  opts: { stream?: boolean; signal?: AbortSignal } = {},
): Promise<Response> {
  const config = PROVIDER_CONFIG[provider];
  const apiKey = config.apiKey();
  if (!apiKey) {
    return new Response(
      JSON.stringify({ error: { message: `${provider.toUpperCase()}_KEY not set on server` } }),
      { status: 599 },
    );
  }

  const body: any = { model: modelId, messages };
  if (opts.stream) body.stream = true;

  const backoffs = [0, 1500, 4000];
  for (let attempt = 0; attempt < backoffs.length; attempt++) {
    if (backoffs[attempt]) await new Promise(r => setTimeout(r, backoffs[attempt]));
    let r: Response;
    try {
      r = await fetch(config.endpoint, {
        method: "POST",
        headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json", ...config.headers },
        body: JSON.stringify(body),
        signal: opts.signal,
      });
    } catch (e) {
      if (attempt === backoffs.length - 1) {
        return new Response(JSON.stringify({ error: { message: String(e) } }), { status: 599 });
      }
      continue;
    }
    if (r.ok) return r;
    if (r.status === 429) return r;
    if (r.status < 500) return r;
    if (attempt === backoffs.length - 1) return r;
  }
  return new Response("unreachable", { status: 500 });
}

// ------------------------------------------------------------
// Gemini direct (handles images AND PDFs as inline_data)
// ------------------------------------------------------------
async function callGeminiDirect(
  modelId: string,
  messages: any[],
  opts: { stream?: boolean; signal?: AbortSignal } = {},
): Promise<Response> {
  if (!GOOGLE_AI_KEY) {
    return new Response(JSON.stringify({ error: { message: "GOOGLE_AI_KEY not set on server." } }), { status: 599 });
  }

  let systemInstruction = "";
  const contents: any[] = [];

  for (const m of messages) {
    if (m.role === "system") {
      systemInstruction = typeof m.content === "string" ? m.content : "";
      continue;
    }

    const parts: any[] = [];
    if (typeof m.content === "string") {
      parts.push({ text: m.content });
    } else if (Array.isArray(m.content)) {
      for (const part of m.content) {
        if (part.type === "text") {
          parts.push({ text: part.text });
        } else if (part.type === "image_url" && part.image_url?.url) {
          const match = part.image_url.url.match(/^data:([^;]+);base64,(.+)$/);
          if (match) parts.push({ inline_data: { mime_type: match[1], data: match[2] } });
        } else if (part.type === "pdf_url" && part.pdf_url?.url) {
          const match = part.pdf_url.url.match(/^data:([^;]+);base64,(.+)$/);
          if (match) parts.push({ inline_data: { mime_type: match[1], data: match[2] } });
        }
      }
    }

    if (parts.length === 0) continue;
    contents.push({ role: m.role === "assistant" ? "model" : "user", parts });
  }

  const body: any = {
    contents,
    safetySettings: [
      { category: "HARM_CATEGORY_HARASSMENT",        threshold: "BLOCK_ONLY_HIGH" },
      { category: "HARM_CATEGORY_HATE_SPEECH",       threshold: "BLOCK_ONLY_HIGH" },
      { category: "HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold: "BLOCK_ONLY_HIGH" },
      { category: "HARM_CATEGORY_DANGEROUS_CONTENT", threshold: "BLOCK_ONLY_HIGH" },
    ],
  };
  if (systemInstruction) body.systemInstruction = { parts: [{ text: systemInstruction }] };

  const baseUrl = `https://generativelanguage.googleapis.com/v1beta/models/${modelId}`;
  const url = opts.stream
    ? `${baseUrl}:streamGenerateContent?alt=sse&key=${GOOGLE_AI_KEY}`
    : `${baseUrl}:generateContent?key=${GOOGLE_AI_KEY}`;

  try {
    return await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: opts.signal,
    });
  } catch (e) {
    return new Response(JSON.stringify({ error: { message: String(e) } }), { status: 599 });
  }
}

async function callProvider(option: ProviderOption, messages: any[], opts: { stream?: boolean; signal?: AbortSignal } = {}): Promise<Response> {
  if (option.provider === "gemini") return callGeminiDirect(option.id, messages, opts);
  return callOAICompatible(option.provider, option.id, messages, opts);
}

async function callProviderText(option: ProviderOption, messages: any[]): Promise<string> {
  const r = await callProvider(option, messages);
  if (!r.ok) return "";
  const d = await r.json();
  if (option.provider === "gemini") return d.candidates?.[0]?.content?.parts?.[0]?.text || "";
  return d.choices?.[0]?.message?.content || "";
}

// ------------------------------------------------------------
// Message format: handle text inlining + image/PDF parts
// Frontend sends:
//   { role, content, files: [{ kind: "image"|"pdf"|"text", name, mimeType, data?, content? }] }
// ------------------------------------------------------------
function toOAIMessages(messages: any[]): {
  msgs: any[];
  hasImages: boolean;
  hasPDFs: boolean;
  mimeTypes: string[];
} {
  let hasImages = false;
  let hasPDFs = false;
  const mimeTypes: string[] = [];

  const msgs = messages.map(m => {
    const files: any[] = [];
    if (Array.isArray(m.files)) files.push(...m.files);
    // Backward compat: old format used m.images = [{mimeType, data}]
    if (Array.isArray(m.images)) {
      for (const img of m.images) {
        if (img?.data && img?.mimeType) {
          files.push({ kind: "image", name: "image", mimeType: img.mimeType, data: img.data });
        }
      }
    }

    if (m.role !== "user" || files.length === 0) {
      return { role: m.role, content: m.content || "" };
    }

    const parts: any[] = [];
    let inlineText = m.content || "";

    for (const f of files) {
      if (f.mimeType) mimeTypes.push(f.mimeType);

      if (f.kind === "text" && typeof f.content === "string") {
        // Inline text content into the user message
        inlineText += "\n\n--- Attached file: " + (f.name || "file") + " ---\n" + f.content + "\n--- end of file ---";
      } else if (f.kind === "image" && f.data && f.mimeType) {
        hasImages = true;
        parts.push({ type: "image_url", image_url: { url: "data:" + f.mimeType + ";base64," + f.data } });
      } else if (f.kind === "pdf" && f.data && f.mimeType) {
        hasPDFs = true;
        parts.push({ type: "pdf_url", pdf_url: { url: "data:" + f.mimeType + ";base64," + f.data, name: f.name || "doc.pdf" } });
      }
    }

    if (inlineText) parts.unshift({ type: "text", text: inlineText });
    if (parts.length === 0) return { role: "user", content: m.content || "" };
    if (parts.length === 1 && parts[0].type === "text") return { role: "user", content: parts[0].text };
    return { role: "user", content: parts };
  });

  return { msgs, hasImages, hasPDFs, mimeTypes };
}

const sseChunk = (event: string, data: any): Uint8Array =>
  new TextEncoder().encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);

function friendlyStatus(status: number, body: string, option: ProviderOption): string {
  if (status === 401 || status === 403) return `${option.label}: auth issue. Check ${option.provider.toUpperCase()}_KEY in Deno Deploy.`;
  if (status === 404) return `${option.label}: model not found (likely deprecated by provider).`;
  if (status === 429) return `${option.label}: rate-limited.`;
  if (status >= 500) return `${option.label}: server error HTTP ${status}.`;
  try {
    const parsed = JSON.parse(body);
    const m = parsed?.error?.message || parsed?.message;
    if (m) return `${option.label}: ${String(m).slice(0, 200)}`;
  } catch {}
  return `${option.label}: HTTP ${status}.`;
}

async function streamProviderResponse(
  r: Response,
  option: ProviderOption,
  send: (event: string, data: any) => void,
  signal: AbortSignal,
): Promise<{ ok: boolean; reason?: string }> {
  const isGemini = option.provider === "gemini";
  if (!r.body) return { ok: false, reason: "no response body" };

  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let gotAnyToken = false;

  while (true) {
    if (signal.aborted) { try { reader.cancel(); } catch {} return { ok: gotAnyToken, reason: "aborted" }; }
    let chunk;
    try { chunk = await reader.read(); } catch (e) { return { ok: gotAnyToken, reason: "stream error: " + e }; }
    if (chunk.done) break;

    buffer += decoder.decode(chunk.value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const payload = line.slice(6).trim();
      if (payload === "[DONE]") continue;

      let j: any;
      try { j = JSON.parse(payload); } catch { continue; }

      if (j.error) return { ok: gotAnyToken, reason: "provider error: " + (j.error.message || JSON.stringify(j.error).slice(0,100)) };
      if (isGemini && j.promptFeedback?.blockReason) return { ok: gotAnyToken, reason: "blocked by Gemini safety: " + j.promptFeedback.blockReason };

      const token = isGemini
        ? j.candidates?.[0]?.content?.parts?.[0]?.text || ""
        : j.choices?.[0]?.delta?.content || "";

      if (token) { gotAnyToken = true; send("token", { text: token }); }
    }
  }

  return { ok: gotAnyToken, reason: gotAnyToken ? undefined : "empty response" };
}

// ------------------------------------------------------------
// Turn: pick mode → run chain → stream
// ------------------------------------------------------------
async function runTurn(
  controller: ReadableStreamDefaultController,
  messages: any[],
  modelKey: string,
  useSearch: boolean,
  useReasoning: boolean,
  memories: string[],
  signal: AbortSignal,
) {
  const send = (event: string, data: any) => { try { controller.enqueue(sseChunk(event, data)); } catch {} };

  // Convert and analyze attachments
  const { msgs: oaiMsgs, hasImages, hasPDFs, mimeTypes } = toOAIMessages(messages);
  const lastUser = messages[messages.length - 1];
  const lastUserText = typeof lastUser?.content === "string" ? lastUser.content : "";
  const hasMultimodal = hasImages || hasPDFs;

  let actualKey = modelKey;
  let routingNote: string | null = null;
  if (hasMultimodal && !MODELS[modelKey]?.vision) {
    actualKey = VISION;
    routingNote = hasPDFs ? "PDF attached → vision chain" : "Image attached → vision chain";
  } else if (useReasoning) {
    actualKey = REASONER;
    routingNote = "Deep reasoning → reasoner chain";
  }

  const entry = MODELS[actualKey];
  if (!entry) { send("error", { message: "Unknown mode: " + actualKey }); return; }

  // If a PDF is attached, prune the chain to options that can actually decode PDFs.
  // (Llama Vision can't — it would silently produce garbage from the text-only parts.)
  const effectiveChain: ProviderOption[] = hasPDFs
    ? entry.chain.filter(o => o.pdfCapable)
    : entry.chain;

  if (effectiveChain.length === 0) {
    send("error", { message: "No PDF-capable provider available. PDFs need Gemini — check GOOGLE_AI_KEY in Deno Deploy." });
    return;
  }

  send("status", { leader: (effectiveChain[0] || entry.chain[0]).label });
  if (routingNote) send("note", { msg: routingNote });
  if (hasPDFs && effectiveChain.length < entry.chain.length) {
    send("note", { msg: "PDF mode → Gemini only (other vision providers can't decode PDFs)" });
  }

  // Web search if requested (skip if multimodal — Tavily can't see images)
  let finalOaiMsgs = oaiMsgs;
  let didSearch = false;
  if (useSearch && !hasMultimodal && lastUserText) {
    send("tool_start", { tool: "web_search" });
    const searchResult = await tavilySearch(lastUserText);
    send("tool_done", { tool: "web_search", preview: searchResult.slice(0, 200) });
    didSearch = true;
    const lastIdx = finalOaiMsgs.length - 1;
    const orig = finalOaiMsgs[lastIdx];
    const origText = typeof orig.content === "string" ? orig.content : "";
    finalOaiMsgs = [...finalOaiMsgs];
    finalOaiMsgs[lastIdx] = { ...orig, content: "Web search results (use these to answer):\n\n" + searchResult + "\n\n---\n\nMy question: " + origText };
  }

  // System prompt (skills compose based on context)
  const sysContent = buildSystemPrompt(memories, didSearch, lastUserText, mimeTypes);
  const finalMsgs = [{ role: "system", content: sysContent }, ...finalOaiMsgs];

  // Iterate the (possibly filtered) chain
  const triedErrors: string[] = [];
  for (let i = 0; i < effectiveChain.length; i++) {
    if (signal.aborted) { send("aborted", {}); return; }
    const option = effectiveChain[i];

    if (!providerKey(option.provider)) {
      triedErrors.push(option.label + ": " + option.provider.toUpperCase() + "_KEY not set");
      continue;
    }

    if (i > 0) send("note", { msg: "Trying " + option.label + "…" });

    const r = await callProvider(option, finalMsgs, { stream: true, signal });

    if (!r.ok) {
      let errText = "";
      try { errText = await r.text(); } catch {}
      const reason = friendlyStatus(r.status, errText, option);
      triedErrors.push(reason);
      if (r.status === 401 || r.status === 403) {
        send("error", { message: "Stopped: " + reason + ". Fix the key in Deno Deploy env vars." });
        return;
      }
      continue;
    }

    const result = await streamProviderResponse(r, option, send, signal);
    if (result.ok) {
      send("done", {
        responder: option.label,
        tool_used: didSearch ? "web_search" : (actualKey === REASONER ? "ask_reasoner" : null),
        fallback_used: i > 0,
        skills_active: pickSkills(lastUserText, mimeTypes).filter(s => s.id !== "core").map(s => s.id),
      });
      return;
    }
    triedErrors.push(option.label + ": " + (result.reason || "no tokens"));
  }

  const summary = triedErrors.map(e => "• " + e).join("\n");
  send("error", { message: "All providers in the **" + entry.label + "** chain failed:\n\n" + summary + "\n\nTry another mode or wait a minute." });
}

// ------------------------------------------------------------
// Memory extraction (every 3rd user message)
// ------------------------------------------------------------
const MEM_EVERY = 3;

async function maybeExtractMemories(uh: string, chat: any) {
  const msgs = chat.messages || [];
  const userCount = msgs.filter((m: any) => m.role === "user").length;
  if (userCount === 0 || userCount % MEM_EVERY !== 0) return;

  const window = msgs.slice(-6)
    .map((m: any) => m.role.toUpperCase() + ": " + (m.content || "").slice(0, 800))
    .join("\n");
  const existing = await getMem(uh);

  const prompt = `Extract durable facts about the user from this exchange. Only facts useful in future conversations: name, ongoing projects, preferences, recurring interests, location, work, relationships, skills, opinions they hold. Skip anything transient. Skip anything already in the existing list.

Existing memories:
${existing.length ? existing.map(m => "- " + m).join("\n") : "(none yet)"}

Recent exchange:
${window}

Output ONLY a JSON array of new short memory strings, e.g. ["User's name is Sam", "User is learning Spanish"]. Empty array [] if nothing memorable. No other text, no markdown fences.`;

  for (const option of MODELS.chat.chain) {
    if (!providerKey(option.provider)) continue;
    try {
      const text = await callProviderText(option, [{ role: "user", content: prompt }]);
      if (!text) continue;
      const cleaned = text.replace(/^```(?:json)?\s*/i, "").replace(/```\s*$/i, "").trim();
      let parsed: any;
      try { parsed = JSON.parse(cleaned); }
      catch {
        const m = cleaned.match(/\[[\s\S]*\]/);
        if (!m) continue;
        try { parsed = JSON.parse(m[0]); } catch { continue; }
      }
      if (!Array.isArray(parsed) || parsed.length === 0) return;
      const valid = parsed.filter((x: any) => typeof x === "string" && x.length > 3 && x.length < 200);
      if (valid.length === 0) return;
      await putMem(uh, [...existing, ...valid]);
      return;
    } catch { continue; }
  }
}

// ============================================================
// HTTP HANDLERS
// ============================================================
async function handleChat(req: Request): Promise<Response> {
  let body: any;
  try { body = await req.json(); }
  catch { return new Response("Invalid JSON", { status: 400 }); }

  const { messages, sync_key, model, use_search, use_reasoning } = body;
  if (!sync_key) return new Response("missing sync_key", { status: 400 });
  if (!Array.isArray(messages)) return new Response("missing messages", { status: 400 });

  const uh = await hashKey(sync_key);
  const mems = await getMem(uh);
  const modelKey = model && MODELS[model] ? model : DEFAULT_LEAD;

  const ac = new AbortController();
  req.signal.addEventListener("abort", () => ac.abort());

  const stream = new ReadableStream({
    async start(controller) {
      try {
        await runTurn(controller, messages, modelKey, !!use_search, !!use_reasoning, mems, ac.signal);
      } catch (e) {
        try { controller.enqueue(sseChunk("error", { message: String(e) })); } catch {}
      } finally {
        try { controller.close(); } catch {}
      }
    },
    cancel() { ac.abort(); },
  });

  return new Response(stream, {
    headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", "X-Accel-Buffering": "no" },
  });
}

async function handleChatsList(req: Request): Promise<Response> {
  const sk = new URL(req.url).searchParams.get("sk") || "";
  if (!sk) return new Response("missing sk", { status: 400 });
  const uh = await hashKey(sk);
  const chats = await listChats(uh);
  return Response.json(chats.map((c: any) => ({
    id: c.id, title: c.title, updated: c.updated, pinned: !!c.pinned,
    preview: (c.messages?.[c.messages.length - 1]?.content || "").slice(0, 100),
  })));
}

async function handleChatGet(req: Request, id: string): Promise<Response> {
  const sk = new URL(req.url).searchParams.get("sk") || "";
  if (!sk) return new Response("missing sk", { status: 400 });
  const uh = await hashKey(sk);
  const chat = await getChat(uh, id);
  if (!chat) return new Response("not found", { status: 404 });
  return Response.json(chat);
}

async function handleChatSave(req: Request): Promise<Response> {
  let body: any;
  try { body = await req.json(); }
  catch { return Response.json({ ok: false, error: "Invalid JSON" }); }
  const { sync_key, chat } = body;
  if (!sync_key || !chat?.id) return Response.json({ ok: false, error: "bad payload" });
  const uh = await hashKey(sync_key);

  // Strip binary attachment data — keeps KV under 64KB
  if (Array.isArray(chat.messages)) {
    chat.messages = chat.messages.map((m: any) => {
      if (Array.isArray(m.files) && m.files.length > 0) {
        const stripped = m.files.map((f: any) => {
          if (f.kind === "text") return f; // keep small text inline
          return { kind: f.kind, name: f.name, mimeType: f.mimeType }; // drop the data
        });
        const counts = m.files.length;
        const note = counts === 1 ? "[1 file attached]" : "[" + counts + " files attached]";
        // Idempotent: only append the note if it's not already there (or any older note variant).
        const content = typeof m.content === "string" ? m.content : "";
        const alreadyTagged = /\[\d+ files? attached\]\s*$/.test(content.trimEnd());
        const newContent = alreadyTagged || !content
          ? (content || note)
          : content + "\n\n" + note;
        return { ...m, files: stripped, content: newContent };
      }
      // Legacy
      if (Array.isArray(m.images) && m.images.length > 0) {
        const note = m.images.length === 1 ? "[image attached]" : "[" + m.images.length + " images attached]";
        const content = typeof m.content === "string" ? m.content : "";
        const alreadyTagged = /\[\d+ images? attached\]\s*$|\[image attached\]\s*$/.test(content.trimEnd());
        const newContent = alreadyTagged || !content
          ? (content || note)
          : content + "\n\n" + note;
        return { ...m, images: [], content: newContent };
      }
      return m;
    });
  }

  const SIZE_LIMIT = 60000;
  let serialized = JSON.stringify(chat);
  if (serialized.length > SIZE_LIMIT && Array.isArray(chat.messages)) {
    while (chat.messages.length > 4 && serialized.length > SIZE_LIMIT) {
      chat.messages.shift();
      serialized = JSON.stringify(chat);
    }
    if (serialized.length > SIZE_LIMIT) {
      chat.messages = chat.messages.map((m: any) => ({
        ...m,
        content: typeof m.content === "string" && m.content.length > 4000
          ? m.content.slice(0, 4000) + "…[truncated]"
          : m.content,
      }));
      serialized = JSON.stringify(chat);
    }
    if (serialized.length > SIZE_LIMIT) return Response.json({ ok: false, error: "Chat too large even after truncation" });
  }

  try {
    await putChat(uh, chat);
    maybeExtractMemories(uh, chat);
    return Response.json({ ok: true });
  } catch (e) {
    return Response.json({ ok: false, error: String(e) });
  }
}

async function handleChatDelete(req: Request, id: string): Promise<Response> {
  const sk = new URL(req.url).searchParams.get("sk") || "";
  if (!sk) return new Response("missing sk", { status: 400 });
  await delChat(await hashKey(sk), id);
  return Response.json({ ok: true });
}

async function handleChatPin(req: Request, id: string): Promise<Response> {
  let body: any;
  try { body = await req.json(); }
  catch { return new Response("Invalid JSON", { status: 400 }); }
  const { sync_key, pinned } = body;
  if (!sync_key) return new Response("missing sync_key", { status: 400 });
  const uh = await hashKey(sync_key);
  const c: any = await getChat(uh, id);
  if (!c) return new Response("not found", { status: 404 });
  c.pinned = !!pinned;
  await putChat(uh, c);
  return Response.json({ ok: true });
}

async function handleMemGet(req: Request): Promise<Response> {
  const sk = new URL(req.url).searchParams.get("sk") || "";
  if (!sk) return new Response("missing sk", { status: 400 });
  return Response.json(await getMem(await hashKey(sk)));
}

async function handleMemSave(req: Request): Promise<Response> {
  let body: any;
  try { body = await req.json(); }
  catch { return new Response("Invalid JSON", { status: 400 }); }
  const { sync_key, memories } = body;
  if (!sync_key || !Array.isArray(memories)) return new Response("bad payload", { status: 400 });
  await putMem(await hashKey(sync_key), memories);
  return Response.json({ ok: true });
}


// ============================================================
// FRONTEND
// ============================================================
const HTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover,maximum-scale=1">
<meta name="theme-color" content="#0a0a0d">
<title>Router</title>
<link rel="manifest" href="/manifest.json">
<link rel="icon" href="/icon.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="/icon.svg">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify@3.1.6/dist/purify.min.js"></script>
<style>
:root{
  --bg:#0a0a0d; --bg-2:#0e0e12; --panel:#131318; --surface:#1a1a20; --surface-2:#21212a; --surface-3:#2a2a35;
  --border:rgba(255,255,255,.06); --border-strong:rgba(255,255,255,.1); --border-bright:rgba(255,255,255,.18);
  --text:#f0f0f4; --text-dim:#a8a8b2; --text-muted:#71717a; --text-faint:#52525b;
  --indigo:#7c7af8; --violet:#a855f7; --pink:#ec4899; --cyan:#22d3ee;
  --accent:#7c7af8; --accent-soft:rgba(124,122,248,.16);
  --search:#fbbf24; --search-soft:rgba(251,191,36,.14);
  --reason:#c084fc; --reason-soft:rgba(192,132,252,.16);
  --danger:#ef4444; --danger-soft:rgba(239,68,68,.13);
  --success:#10b981; --success-soft:rgba(16,185,129,.13);
  --warn:#f59e0b;
  --grad:linear-gradient(135deg,#7c7af8 0%,#a855f7 55%,#ec4899 100%);
  --grad-soft:linear-gradient(135deg,rgba(124,122,248,.18) 0%,rgba(168,85,247,.14) 100%);
  --shadow-lg:0 20px 60px rgba(0,0,0,.5);
  --shadow:0 10px 32px rgba(0,0,0,.4);
  --shadow-glow:0 0 0 1px rgba(124,122,248,.3),0 8px 32px rgba(124,122,248,.28);
  --radius:14px; --radius-lg:18px; --radius-sm:10px;
  --ease:cubic-bezier(.32,.72,.32,1);
  --ease-out:cubic-bezier(.16,1,.3,1);
}
*,*::before,*::after{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
html,body{margin:0;padding:0;height:100%;background:var(--bg);color:var(--text);font-family:"Inter",-apple-system,BlinkMacSystemFont,system-ui,sans-serif;-webkit-font-smoothing:antialiased;overflow:hidden;font-size:15px;line-height:1.55;letter-spacing:-.005em}
body{display:flex;flex-direction:column;position:relative;isolation:isolate}
button,input,textarea,select{font-family:inherit;letter-spacing:inherit}
button{user-select:none;-webkit-user-select:none;cursor:pointer}
::selection{background:rgba(124,122,248,.4);color:#fff}

/* ============ AURORA BACKGROUND ============ */
.aurora{position:fixed;inset:-50px;overflow:hidden;z-index:-1;pointer-events:none}
.aurora-blob{position:absolute;border-radius:50%;filter:blur(80px);will-change:transform,opacity}
.aurora-blob:nth-child(1){width:520px;height:520px;background:radial-gradient(circle,var(--indigo),transparent 70%);top:-180px;left:-160px;opacity:.4;animation:drift1 28s ease-in-out infinite}
.aurora-blob:nth-child(2){width:620px;height:620px;background:radial-gradient(circle,var(--violet),transparent 70%);bottom:-220px;right:-220px;opacity:.35;animation:drift2 32s ease-in-out infinite}
.aurora-blob:nth-child(3){width:380px;height:380px;background:radial-gradient(circle,var(--pink),transparent 70%);top:45%;left:40%;opacity:.18;animation:drift3 24s ease-in-out infinite}
@keyframes drift1{0%,100%{transform:translate(0,0) scale(1)}33%{transform:translate(80px,-40px) scale(1.1)}66%{transform:translate(-30px,60px) scale(.9)}}
@keyframes drift2{0%,100%{transform:translate(0,0) scale(1)}33%{transform:translate(-60px,30px) scale(1.05)}66%{transform:translate(40px,-50px) scale(.95)}}
@keyframes drift3{0%,100%{transform:translate(0,0) scale(1)}50%{transform:translate(-100px,-80px) scale(1.2)}}

/* ============ HEADER (glass) ============ */
.header{display:flex;align-items:center;padding:11px 14px;border-bottom:1px solid var(--border);flex-shrink:0;gap:4px;padding-top:max(11px,env(safe-area-inset-top));background:rgba(10,10,13,.7);position:relative;z-index:5;backdrop-filter:blur(20px) saturate(150%);-webkit-backdrop-filter:blur(20px) saturate(150%)}
.icon-btn{background:transparent;border:none;color:var(--text-dim);padding:9px;border-radius:11px;display:flex;align-items:center;justify-content:center;width:40px;height:40px;transition:background .14s var(--ease),color .12s,transform .08s}
.icon-btn:hover{color:var(--text);background:var(--surface)}
.icon-btn:active{transform:scale(.92);background:var(--surface-2)}
.icon-btn svg{width:21px;height:21px;stroke-width:2}
.title-wrap{flex:1;text-align:center;padding:0 6px;overflow:hidden;cursor:pointer}
.title{font-size:15px;color:var(--text);font-weight:700;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;letter-spacing:-.02em}
.model-badge{font-size:10.5px;color:var(--text-muted);margin-top:1px;display:flex;align-items:center;justify-content:center;gap:5px;font-weight:600;letter-spacing:.01em}
.model-badge .dot{width:6px;height:6px;border-radius:50%;background:var(--success);box-shadow:0 0 0 3px rgba(16,185,129,.18);animation:dot-pulse 2.4s ease-in-out infinite}
@keyframes dot-pulse{0%,100%{box-shadow:0 0 0 3px rgba(16,185,129,.18)}50%{box-shadow:0 0 0 6px rgba(16,185,129,.04)}}

/* ============ SIDEBAR ============ */
.sidebar{position:fixed;top:0;left:0;bottom:0;width:316px;max-width:86vw;background:rgba(19,19,24,.95);backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);transform:translateX(-100%);transition:transform .32s var(--ease);z-index:100;display:flex;flex-direction:column;border-right:1px solid var(--border);padding-top:env(safe-area-inset-top);padding-bottom:env(safe-area-inset-bottom);box-shadow:var(--shadow-lg)}
.sidebar.open{transform:translateX(0)}
.sb-head{padding:14px;border-bottom:1px solid var(--border);display:flex;flex-direction:column;gap:10px}
.new-chat{width:100%;background:var(--grad);color:#fff;border:none;padding:13px 16px;border-radius:13px;font-size:14.5px;font-weight:600;display:flex;align-items:center;justify-content:center;gap:8px;letter-spacing:-.01em;box-shadow:inset 0 1px 0 rgba(255,255,255,.18),0 6px 20px rgba(124,122,248,.36),0 1px 2px rgba(0,0,0,.1);transition:transform .12s,filter .14s,box-shadow .18s}
.new-chat:hover{filter:brightness(1.1);box-shadow:inset 0 1px 0 rgba(255,255,255,.22),0 8px 26px rgba(124,122,248,.44),0 1px 2px rgba(0,0,0,.1)}
.new-chat:active{transform:translateY(1px) scale(.99);filter:brightness(.9)}
.search-wrap{position:relative}
.search-wrap svg{position:absolute;left:12px;top:50%;transform:translateY(-50%);width:14px;height:14px;color:var(--text-muted);pointer-events:none}
.search-box{width:100%;background:var(--surface);border:1px solid var(--border);color:var(--text);padding:10px 12px 10px 34px;border-radius:11px;font-size:13.5px;outline:none;transition:border-color .14s,background .14s}
.search-box:focus{border-color:var(--accent);background:var(--surface-2)}
.chats-list{flex:1;overflow-y:auto;padding:8px;-webkit-overflow-scrolling:touch}
.chat-item{padding:11px 13px;border-radius:12px;font-size:13.5px;color:var(--text-dim);margin-bottom:3px;display:flex;align-items:center;gap:8px;transition:background .14s,color .12s;position:relative}
.chat-item:hover{background:var(--surface);color:var(--text)}
.chat-item.active{background:var(--accent-soft);color:var(--text);box-shadow:inset 0 0 0 1px rgba(124,122,248,.22)}
.chat-item .row{flex:1;overflow:hidden;display:flex;flex-direction:column;gap:2px;min-width:0;cursor:pointer}
.chat-title-row{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:500;display:flex;align-items:center;gap:6px;font-size:13.5px}
.chat-prev{font-size:11.5px;color:var(--text-muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pin-icon{font-size:10px;opacity:.7}
.chat-actions{display:flex;gap:2px;flex-shrink:0}
.chat-actions button{background:transparent;border:none;color:var(--text-faint);font-size:13px;padding:6px;border-radius:7px;transition:color .12s,background .12s;display:flex;align-items:center;justify-content:center;width:30px;height:30px}
.chat-actions button:hover{color:var(--text);background:var(--surface-2)}
.chat-actions button:active{transform:scale(.88)}
.chat-actions .del:hover{color:var(--danger);background:var(--danger-soft)}
.chat-actions svg{width:14px;height:14px}
.list-empty{text-align:center;color:var(--text-muted);font-size:13px;padding:40px 20px;font-style:italic;line-height:1.65}
.backdrop{position:fixed;inset:0;background:rgba(0,0,0,.7);backdrop-filter:blur(4px);-webkit-backdrop-filter:blur(4px);z-index:99;opacity:0;pointer-events:none;transition:opacity .24s}
.backdrop.open{opacity:1;pointer-events:auto}

/* ============ MESSAGES ============ */
.messages-container{flex:1;position:relative;overflow:hidden;display:flex;flex-direction:column}
.messages{flex:1;overflow-y:auto;padding:20px 14px 6px;-webkit-overflow-scrolling:touch;overscroll-behavior-y:contain}
.messages-inner{max-width:780px;margin:0 auto;display:flex;flex-direction:column;gap:18px}
.msg{display:flex;flex-direction:column;gap:5px;max-width:100%;animation:fadeUp .3s var(--ease-out)}
@keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.msg.user{align-items:flex-end}
.msg.assistant{align-items:flex-start}
.bubble{padding:11px 15px;border-radius:18px;line-height:1.6;word-wrap:break-word;overflow-wrap:break-word;font-size:15px;max-width:92%;position:relative}
.msg.user .bubble{background:var(--grad);color:#fff;border-bottom-right-radius:6px;box-shadow:inset 0 1px 0 rgba(255,255,255,.15),0 4px 16px rgba(124,122,248,.28),0 1px 2px rgba(0,0,0,.1)}
.msg.assistant .bubble{background:rgba(26,26,32,.85);color:var(--text);border-bottom-left-radius:6px;border:1px solid var(--border);backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px)}
.msg.assistant .bubble.error-bubble{background:var(--danger-soft);border-color:rgba(239,68,68,.4);color:#fecaca}
.bubble.empty-streaming{min-height:42px;min-width:60px}
.routing-pill{display:inline-flex;align-items:center;gap:8px;color:var(--text-dim);font-size:13px;font-style:italic;font-weight:500}
.routing-pill .rp-dot{width:7px;height:7px;border-radius:50%;background:var(--accent);box-shadow:0 0 0 3px rgba(124,122,248,.2);animation:rp-pulse 1.4s ease-in-out infinite}
@keyframes rp-pulse{0%,100%{opacity:.5;transform:scale(.85)}50%{opacity:1;transform:scale(1.1)}}
.msg-img{max-width:240px;max-height:240px;border-radius:12px;margin:4px 0;display:block;border:1px solid var(--border);object-fit:cover;cursor:zoom-in;transition:transform .14s}
.msg-img:active{transform:scale(.98)}
.msg-file{display:flex;align-items:center;gap:10px;background:rgba(255,255,255,.06);border:1px solid var(--border);padding:9px 12px;border-radius:11px;margin:4px 0;font-size:13px;max-width:280px}
.msg-file .file-icon{width:28px;height:28px;border-radius:7px;background:var(--surface-2);display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:13px;font-weight:700;letter-spacing:.02em;color:var(--text-dim)}
.msg-file .file-info{flex:1;min-width:0}
.msg-file .file-name{font-weight:500;color:var(--text);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:12.5px}
.msg-file .file-meta{font-size:10.5px;color:var(--text-muted);margin-top:1px}
.bubble p{margin:0 0 .6em}
.bubble p:last-child{margin:0}
.bubble pre{background:#08080b;border:1px solid var(--border);border-radius:12px;padding:14px 14px 12px;overflow-x:auto;margin:10px 0;font-family:"JetBrains Mono",ui-monospace,monospace;font-size:13px;line-height:1.55;position:relative}
.bubble code{font-family:"JetBrains Mono",ui-monospace,monospace;font-size:.88em;background:rgba(255,255,255,.08);padding:2.5px 6px;border-radius:5px;font-weight:500}
.bubble pre code{background:transparent;padding:0;border-radius:0;font-size:13px;font-weight:400}
.msg.user .bubble code{background:rgba(255,255,255,.22)}
.msg.user .bubble pre{background:rgba(0,0,0,.32);border-color:rgba(255,255,255,.18)}
.bubble ul,.bubble ol{margin:5px 0 8px 22px;padding:0}
.bubble li{margin:4px 0}
.bubble h1,.bubble h2,.bubble h3,.bubble h4{margin:14px 0 6px;font-weight:700;letter-spacing:-.02em}
.bubble h1{font-size:1.28em}.bubble h2{font-size:1.16em}.bubble h3{font-size:1.06em}.bubble h4{font-size:1em}
.bubble h1:first-child,.bubble h2:first-child,.bubble h3:first-child{margin-top:0}
.bubble a{color:#c4b5fd;text-decoration:underline;text-underline-offset:2px;text-decoration-color:rgba(196,181,253,.4)}
.bubble a:hover{text-decoration-color:#c4b5fd}
.bubble blockquote{border-left:3px solid var(--accent);padding:2px 0 2px 14px;color:var(--text-dim);margin:10px 0;font-style:italic}
.bubble strong{font-weight:700;color:#fff}
.bubble hr{border:none;border-top:1px solid var(--border-strong);margin:14px 0}
.bubble table{border-collapse:collapse;margin:10px 0;font-size:.92em}
.bubble th,.bubble td{border:1px solid var(--border-strong);padding:7px 11px}
.bubble th{background:var(--surface-2);font-weight:600}
.copy-btn{position:absolute;top:7px;right:7px;background:rgba(255,255,255,.08);border:1px solid var(--border);color:var(--text-dim);padding:4px 9px;border-radius:7px;font-size:11px;opacity:.6;transition:opacity .14s,color .14s,background .14s;font-family:"Inter",sans-serif;font-weight:500;letter-spacing:.01em}
.copy-btn:hover{opacity:1;color:var(--text);background:rgba(255,255,255,.16)}
.bubble pre:hover .copy-btn,.bubble pre:active .copy-btn{opacity:1}
.code-lang{position:absolute;top:7px;left:14px;font-size:10.5px;color:var(--text-muted);font-weight:600;text-transform:uppercase;letter-spacing:.08em;font-family:"Inter",sans-serif;pointer-events:none}
.bubble pre:has(.code-lang){padding-top:30px}
.msg-actions{display:flex;gap:6px;margin-top:5px;padding:0 6px;opacity:.62;transition:opacity .16s}
.msg-actions:hover{opacity:1}
.msg-actions button{background:rgba(26,26,32,.7);border:1px solid var(--border);color:var(--text-muted);font-size:11.5px;padding:6px 11px;border-radius:9px;display:flex;align-items:center;gap:6px;transition:all .14s;font-weight:500}
.msg-actions button:hover{color:var(--text);background:var(--surface);border-color:var(--border-strong)}
.msg-actions button:active{transform:scale(.94)}
.msg-actions button svg{width:11px;height:11px;stroke-width:2.4}
.msg-actions .retry{color:#fca5a5;border-color:rgba(239,68,68,.32)}
.msg-actions .retry:hover{color:#fff;background:rgba(239,68,68,.2)}
.caption{font-size:11px;color:var(--text-muted);padding:0 8px;display:flex;gap:6px;align-items:center;flex-wrap:wrap;font-weight:500}
.badge{font-size:9.5px;padding:2.5px 7px;border-radius:5px;font-weight:700;letter-spacing:.04em;text-transform:uppercase}
.b-search{background:var(--search-soft);color:var(--search)}
.b-reason{background:var(--reason-soft);color:var(--reason)}
.b-fallback{background:rgba(192,132,252,.13);color:#c4b5fd}
.b-skill{background:rgba(34,211,238,.12);color:var(--cyan)}
.b-stopped{background:var(--danger-soft);color:#fca5a5}

/* ============ THINKING / TOOL STATES ============ */
.thinking{display:inline-flex;gap:5px;padding:8px 2px;align-items:center}
.thinking span{width:7px;height:7px;border-radius:50%;background:linear-gradient(135deg,var(--indigo),var(--violet));animation:pulse 1.3s ease-in-out infinite}
.thinking span:nth-child(2){animation-delay:.16s}
.thinking span:nth-child(3){animation-delay:.32s}
@keyframes pulse{0%,80%,100%{opacity:.35;transform:scale(.75)}40%{opacity:1;transform:scale(1.1)}}
.tool-running{display:inline-flex;align-items:center;gap:8px;font-size:12px;color:var(--text-dim);padding:7px 13px;background:rgba(26,26,32,.85);border:1px solid var(--border);border-radius:11px;margin-bottom:5px;width:fit-content;animation:fadeUp .24s;font-weight:500;backdrop-filter:blur(8px)}
.tool-running.search{color:var(--search);border-color:rgba(251,191,36,.34)}
.tool-running.reason{color:var(--reason);border-color:rgba(192,132,252,.34)}
.tool-running.note{color:var(--success);border-color:rgba(16,185,129,.34);background:rgba(16,185,129,.06)}

/* ============ SCROLL-TO-BOTTOM PILL ============ */
.scroll-pill{position:absolute;bottom:16px;left:50%;transform:translateX(-50%) translateY(8px);background:var(--grad);color:#fff;border:none;padding:7px 14px 7px 11px;border-radius:20px;font-size:12px;font-weight:600;display:none;align-items:center;gap:6px;box-shadow:inset 0 1px 0 rgba(255,255,255,.2),0 6px 20px rgba(124,122,248,.4);opacity:0;transition:opacity .2s,transform .2s var(--ease-out);z-index:10;letter-spacing:-.005em}
.scroll-pill.visible{display:flex;opacity:1;transform:translateX(-50%) translateY(0)}
.scroll-pill svg{width:14px;height:14px;stroke-width:2.6}

/* ============ EMPTY STATE ============ */
.empty{text-align:center;padding:50px 22px 80px;max-width:600px;margin:30px auto 0;animation:fadeUp .45s var(--ease-out)}
.empty .greet{font-size:32px;font-weight:800;letter-spacing:-.03em;background:var(--grad);-webkit-background-clip:text;background-clip:text;color:transparent;margin-bottom:10px;line-height:1.15;background-size:200% 200%;animation:gradient-shift 8s ease infinite}
@keyframes gradient-shift{0%,100%{background-position:0% 50%}50%{background-position:100% 50%}}
.empty .sub{color:var(--text-dim);font-size:14px;margin-bottom:32px;line-height:1.6}
.cap-row{display:flex;gap:8px;justify-content:center;flex-wrap:wrap;margin-top:8px}
.cap-chip{background:rgba(26,26,32,.6);border:1px solid var(--border);color:var(--text-dim);padding:7px 14px;border-radius:20px;font-size:12.5px;font-weight:500;display:inline-flex;align-items:center;gap:6px;backdrop-filter:blur(8px);transition:all .15s}
.cap-chip:hover{background:var(--surface);color:var(--text);border-color:var(--border-strong)}

/* ============ INPUT BAR (the centerpiece) ============ */
.input-bar{flex-shrink:0;padding:0 12px max(10px,env(safe-area-inset-bottom));background:linear-gradient(180deg,transparent 0%,rgba(10,10,13,.8) 30%,rgba(10,10,13,.95) 100%);position:relative;z-index:4}
.toggles{display:flex;gap:8px;padding:10px 0 0;max-width:780px;margin:0 auto;flex-wrap:wrap}
.toggle{background:rgba(26,26,32,.75);backdrop-filter:blur(8px);border:1px solid var(--border);color:var(--text-muted);padding:6px 13px;border-radius:18px;font-size:12.5px;display:flex;align-items:center;gap:6px;transition:all .15s;font-weight:600;letter-spacing:-.005em}
.toggle:hover{color:var(--text);border-color:var(--border-strong);background:var(--surface)}
.toggle:active{transform:scale(.96)}
.toggle.active.search{background:var(--search-soft);color:var(--search);border-color:rgba(251,191,36,.55);box-shadow:0 0 0 1px rgba(251,191,36,.25),0 4px 14px rgba(251,191,36,.2)}
.toggle.active.reason{background:var(--reason-soft);color:var(--reason);border-color:rgba(192,132,252,.55);box-shadow:0 0 0 1px rgba(192,132,252,.25),0 4px 14px rgba(192,132,252,.2)}
.toggle .lbl{display:none}
@media (min-width:380px){.toggle .lbl{display:inline}}

.composer{margin:9px auto 0;max-width:780px;background:rgba(26,26,32,.7);backdrop-filter:blur(16px) saturate(150%);-webkit-backdrop-filter:blur(16px) saturate(150%);border:1.5px solid var(--border);border-radius:24px;transition:border-color .18s,box-shadow .18s,background .18s;overflow:hidden}
.composer:focus-within{border-color:rgba(124,122,248,.5);background:rgba(26,26,32,.85);box-shadow:0 0 0 4px rgba(124,122,248,.12),0 8px 32px rgba(124,122,248,.18)}
.attached-chips{display:flex;gap:8px;padding:10px 12px 0;flex-wrap:wrap}
.attached-chips:empty{display:none}
.chip{position:relative;display:flex;align-items:center;gap:8px;background:var(--surface-2);border:1px solid var(--border);padding:7px 10px 7px 7px;border-radius:11px;font-size:12px;font-weight:500;animation:fadeUp .2s var(--ease-out);max-width:200px}
.chip .thumb{width:36px;height:36px;border-radius:7px;overflow:hidden;flex-shrink:0;background:var(--bg);display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:var(--text-dim);letter-spacing:.05em}
.chip .thumb img{width:100%;height:100%;object-fit:cover}
.chip .chip-info{min-width:0}
.chip .chip-name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--text);font-size:12px}
.chip .chip-size{font-size:10.5px;color:var(--text-muted);margin-top:1px}
.chip .rm{background:rgba(0,0,0,.5);color:#fff;border:none;border-radius:50%;width:18px;height:18px;font-size:12px;display:flex;align-items:center;justify-content:center;padding:0;line-height:1;font-weight:600;position:absolute;top:-5px;right:-5px}
.chip .rm:hover{background:var(--danger)}
.chip .rm:active{transform:scale(.85)}

.input{background:transparent;color:var(--text);border:none;padding:13px 16px 4px;font-size:15.5px;resize:none;max-height:200px;font-family:inherit;outline:none;line-height:1.55;width:100%;display:block}
.input::placeholder{color:var(--text-muted);transition:color .15s}

.composer-controls{display:flex;align-items:center;padding:6px 8px 8px 10px;gap:4px}
.tool-btn{background:transparent;border:none;color:var(--text-muted);padding:8px;border-radius:10px;display:flex;align-items:center;justify-content:center;width:38px;height:38px;transition:all .14s;position:relative}
.tool-btn:hover{color:var(--text);background:rgba(255,255,255,.06)}
.tool-btn:active{transform:scale(.93)}
.tool-btn svg{width:19px;height:19px;stroke-width:2}
.tool-btn .badge-count{position:absolute;top:1px;right:1px;background:var(--accent);color:#fff;font-size:9.5px;border-radius:50%;width:16px;height:16px;display:flex;align-items:center;justify-content:center;font-weight:700;border:2px solid var(--bg)}
.tool-btn.recording{color:#fff;background:var(--danger);animation:rec-pulse 1.4s ease-in-out infinite}
@keyframes rec-pulse{0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,.5)}50%{box-shadow:0 0 0 9px rgba(239,68,68,0)}}
.composer-spacer{flex:1}

.send{background:var(--grad);color:#fff;border:none;border-radius:50%;width:40px;height:40px;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:all .16s var(--ease);box-shadow:inset 0 1px 0 rgba(255,255,255,.22),0 6px 18px rgba(124,122,248,.4),0 1px 2px rgba(0,0,0,.1)}
.send:disabled{background:#26262b;color:#52525b;box-shadow:none;cursor:not-allowed}
.send:hover:not(:disabled){filter:brightness(1.1);box-shadow:inset 0 1px 0 rgba(255,255,255,.28),0 8px 24px rgba(124,122,248,.5),0 1px 2px rgba(0,0,0,.1)}
.send:active:not(:disabled){transform:translateY(1px) scale(.96);filter:brightness(.9)}
.stop{background:#2a2a35;color:#fff;border:1px solid var(--border-bright);border-radius:50%;width:40px;height:40px;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:all .14s var(--ease);box-shadow:0 6px 18px rgba(0,0,0,.35);position:relative}
.stop svg{width:14px;height:14px}
.stop::before{content:"";position:absolute;inset:-3px;border-radius:50%;border:1px solid rgba(239,68,68,.55);animation:stop-pulse 1.6s ease-out infinite}
.stop:hover{background:#33333f;border-color:rgba(239,68,68,.6)}
.stop:active{transform:scale(.94)}
@keyframes stop-pulse{0%{transform:scale(1);opacity:.7}100%{transform:scale(1.25);opacity:0}}
.send svg{width:19px;height:19px;stroke-width:2.6;transform:translateY(-1px)}

.shortcut-hint{font-size:10.5px;color:var(--text-faint);text-align:center;padding:4px 0 2px;font-weight:500;letter-spacing:.02em}

/* ============ MODALS ============ */
.modal-back{position:fixed;inset:0;background:rgba(0,0,0,.78);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);z-index:200;display:none;align-items:center;justify-content:center;padding:20px;animation:fadeIn .22s}
.modal-back.open{display:flex}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
.modal{background:rgba(19,19,24,.95);backdrop-filter:blur(20px) saturate(150%);-webkit-backdrop-filter:blur(20px) saturate(150%);border:1px solid var(--border-strong);border-radius:22px;max-width:480px;width:100%;padding:26px;box-shadow:var(--shadow-lg);max-height:88vh;overflow-y:auto;animation:modalIn .28s var(--ease-out)}
@keyframes modalIn{from{opacity:0;transform:translateY(16px) scale(.96)}to{opacity:1;transform:translateY(0) scale(1)}}
.modal h2{margin:0 0 5px;font-size:23px;font-weight:800;letter-spacing:-.025em;background:var(--grad);-webkit-background-clip:text;background-clip:text;color:transparent}
.modal .sub{color:var(--text-dim);font-size:13.5px;margin-bottom:18px;line-height:1.6}
.section-label{display:block;font-size:11px;font-weight:700;color:var(--text-muted);margin:18px 0 9px;text-transform:uppercase;letter-spacing:.08em}
.section-label:first-of-type{margin-top:8px}
.modal input,.modal textarea,.modal select{width:100%;background:var(--surface);color:var(--text);border:1px solid var(--border);border-radius:12px;padding:11px 14px;font-size:14px;outline:none;transition:border-color .14s,background .14s}
.modal input:focus,.modal textarea:focus{border-color:var(--accent);background:var(--surface-2)}
.modal .actions{display:flex;gap:8px;margin-top:18px}
.btn{padding:12px 18px;border-radius:12px;font-size:14px;font-weight:600;border:none;transition:all .15s;flex:1;letter-spacing:-.005em;display:flex;align-items:center;justify-content:center;gap:6px}
.btn-primary{background:var(--grad);color:#fff;box-shadow:inset 0 1px 0 rgba(255,255,255,.18),0 5px 18px rgba(124,122,248,.36),0 1px 2px rgba(0,0,0,.1)}
.btn-primary:hover{filter:brightness(1.08)}
.btn-primary:active{transform:translateY(1px) scale(.99);filter:brightness(.92)}
.btn-ghost{background:transparent;color:var(--text-dim);border:1px solid var(--border)}
.btn-ghost:hover{background:var(--surface);color:var(--text);border-color:var(--border-strong)}
.btn-ghost:active{transform:scale(.98)}
.btn-danger{background:var(--danger);color:#fff;box-shadow:inset 0 1px 0 rgba(255,255,255,.18),0 5px 16px rgba(239,68,68,.34)}
.btn-danger:hover{filter:brightness(1.1)}
.btn-danger:active{transform:translateY(1px) scale(.99);filter:brightness(.9)}
.divider{height:1px;background:var(--border);margin:22px -26px}
.key-display{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px;font-family:"JetBrains Mono",monospace;font-size:15px;text-align:center;letter-spacing:.04em;margin:8px 0;color:#c4b5fd;word-break:break-all;font-weight:500}
.tabs{display:flex;gap:4px;background:var(--surface);padding:4px;border-radius:12px;margin-bottom:18px}
.tab{flex:1;background:transparent;border:none;color:var(--text-dim);padding:10px;border-radius:9px;font-weight:600;font-size:13px;transition:all .14s;letter-spacing:-.005em}
.tab.active{background:var(--panel);color:var(--text);box-shadow:0 1px 4px rgba(0,0,0,.25)}
.mem-list{display:flex;flex-direction:column;gap:7px;max-height:280px;overflow-y:auto;padding-right:4px}
.mem-item{display:flex;align-items:center;gap:8px;background:var(--surface);border:1px solid var(--border);border-radius:11px;padding:10px 13px;font-size:13px;animation:fadeUp .2s}
.mem-item .text{flex:1;color:var(--text-dim);line-height:1.45}
.mem-item .rm{background:transparent;border:none;color:var(--text-muted);font-size:17px;padding:3px 7px;border-radius:7px;transition:color .12s,background .12s;line-height:1;font-weight:500}
.mem-item .rm:hover{color:var(--danger);background:var(--danger-soft)}
.mem-item .rm:active{transform:scale(.85)}
.mem-empty{text-align:center;color:var(--text-muted);font-size:13px;padding:24px;font-style:italic;line-height:1.55}
.mem-add-row{display:flex;gap:8px;margin-top:10px}
.mem-input{flex:1;background:var(--surface);border:1px solid var(--border);color:var(--text);padding:10px 13px;border-radius:11px;font-size:13.5px;outline:none;transition:border-color .14s,background .14s}
.mem-input:focus{border-color:var(--accent);background:var(--surface-2)}
.mem-add-btn{flex-shrink:0;width:44px;padding:0;font-size:22px;font-weight:400;line-height:1;display:flex;align-items:center;justify-content:center}
.model-grid{display:grid;gap:10px}
.model-card{background:var(--surface);border:1.5px solid var(--border);border-radius:13px;padding:13px 15px;text-align:left;transition:all .15s;display:flex;align-items:flex-start;gap:10px}
.model-card:hover{background:var(--surface-2);border-color:var(--border-strong)}
.model-card:active{transform:scale(.99)}
.model-card.selected{border-color:var(--accent);background:var(--accent-soft);box-shadow:inset 0 0 0 1px rgba(124,122,248,.3)}
.model-card .info{flex:1;min-width:0}
.model-card .name{font-weight:700;color:var(--text);font-size:14.5px;margin-bottom:3px;letter-spacing:-.01em}
.model-card .blurb{font-size:12px;color:var(--text-dim);line-height:1.5}
.model-card .chain-list{font-size:11px;color:var(--text-muted);margin-top:6px;font-family:"JetBrains Mono",monospace;letter-spacing:-.02em}
.model-card .check{color:var(--accent);font-size:20px;line-height:1;opacity:0;transition:opacity .14s;font-weight:600}
.model-card.selected .check{opacity:1}
.providers-status{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:13px;margin-top:8px;font-size:12.5px;display:flex;flex-direction:column;gap:8px}
.providers-status .row{display:flex;justify-content:space-between;color:var(--text-dim);align-items:center;gap:8px}
.providers-status .label{flex:1;min-width:0}
.providers-status .note{color:var(--text-muted);font-weight:400;font-size:11.5px}
.providers-status .ok{color:var(--success);font-weight:700;flex-shrink:0}
.providers-status .miss{color:var(--danger);font-weight:700;flex-shrink:0}
.skill-list{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:13px;font-size:12.5px;display:flex;flex-direction:column;gap:8px}
.skill-list .skill-row{display:flex;justify-content:space-between;color:var(--text-dim);align-items:flex-start;gap:8px}
.skill-list .skill-row .name{font-weight:600;color:var(--text);font-size:13px}
.skill-list .skill-row .when{color:var(--text-muted);font-size:11.5px;margin-top:2px}
.about-block{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:13px;font-size:12px;color:var(--text-muted);line-height:1.65}
.about-block strong{color:var(--text);font-weight:700}

/* ============ TOAST ============ */
.toast{position:fixed;top:max(16px,env(safe-area-inset-top));left:50%;transform:translateX(-50%) translateY(-30px);background:rgba(19,19,24,.95);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);border:1px solid var(--border-strong);color:var(--text);padding:11px 18px;border-radius:13px;font-size:13px;z-index:300;opacity:0;pointer-events:none;transition:opacity .22s,transform .22s var(--ease-out);box-shadow:var(--shadow);max-width:90vw;font-weight:500;display:flex;align-items:center;gap:8px}
.toast.show{opacity:1;transform:translateX(-50%) translateY(0)}
.toast.success{border-color:rgba(16,185,129,.5)}
.toast.success .toast-icon{color:var(--success)}
.toast.error{border-color:rgba(239,68,68,.5)}
.toast.error .toast-icon{color:var(--danger)}
.toast.warn{border-color:rgba(245,158,11,.5)}
.toast.warn .toast-icon{color:var(--warn)}
.toast-icon{font-size:14px;line-height:1;font-weight:700}

/* ============ LIGHTBOX ============ */
.lightbox{position:fixed;inset:0;background:rgba(0,0,0,.96);z-index:400;display:none;align-items:center;justify-content:center;padding:24px;cursor:zoom-out;animation:fadeIn .2s}
.lightbox.open{display:flex}
.lightbox img{max-width:100%;max-height:100%;border-radius:10px;box-shadow:var(--shadow-lg)}

/* ============ SCROLLBAR ============ */
::-webkit-scrollbar{width:9px;height:9px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--surface-3);border-radius:5px}
::-webkit-scrollbar-thumb:hover{background:var(--border-bright)}

button:focus-visible,input:focus-visible,textarea:focus-visible{outline:2px solid var(--accent);outline-offset:2px}

@media (prefers-reduced-motion: reduce){
  .aurora-blob,.model-badge .dot,.routing-pill .rp-dot,.thinking span,.stop::before,.voice-btn.recording,.rec-dot,.gradient-bg{animation:none !important}
  .msg,.modal,.toast,.mem-item{animation-duration:.001s !important}
  *,*::before,*::after{transition-duration:.001s !important;scroll-behavior:auto !important}
}
</style>
</head>
<body>
<div class="aurora"><div class="aurora-blob"></div><div class="aurora-blob"></div><div class="aurora-blob"></div></div>

<div class="header">
  <button class="icon-btn" id="menuBtn" aria-label="menu">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
  </button>
  <div class="title-wrap" id="titleWrap">
    <div class="title" id="chatTitle">Router</div>
    <div class="model-badge"><span class="dot"></span><span id="modelLabel">Chat (auto)</span></div>
  </div>
  <button class="icon-btn" id="settingsBtn" aria-label="settings">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
  </button>
</div>

<div class="backdrop" id="backdrop"></div>

<aside class="sidebar" id="sidebar">
  <div class="sb-head">
    <button class="new-chat" id="newBtn">
      <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
      New chat
    </button>
    <div class="search-wrap">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
      <input class="search-box" id="searchBox" placeholder="Search chats…">
    </div>
  </div>
  <div class="chats-list" id="chatsList"></div>
</aside>

<div class="messages-container">
  <div class="messages" id="messages"><div class="messages-inner" id="messagesInner"></div></div>
  <button class="scroll-pill" id="scrollPill">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"/><polyline points="19 12 12 19 5 12"/></svg>
    New message
  </button>
</div>

<div class="input-bar">
  <div class="toggles">
    <button class="toggle search" id="searchToggle" title="Web search">🔍 <span class="lbl">Search</span></button>
    <button class="toggle reason" id="reasonToggle" title="Deep reasoning">🧠 <span class="lbl">Deep</span></button>
  </div>
  <div class="composer">
    <div class="attached-chips" id="attachedChips"></div>
    <textarea class="input" id="input" placeholder="Ask anything…" rows="1"></textarea>
    <div class="composer-controls">
      <button class="tool-btn" id="attachBtn" aria-label="attach file" title="Attach files (image, PDF, text, code)">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>
        <span class="badge-count" id="attachCount" style="display:none">0</span>
      </button>
      <input type="file" id="fileInput" accept="image/*,.pdf,application/pdf,text/*,.txt,.md,.csv,.json,.html,.htm,.xml,.yaml,.yml,.py,.js,.ts,.tsx,.jsx,.css,.scss,.go,.rs,.java,.c,.cpp,.h,.hpp,.sh,.rb,.php,.sql,.kt,.swift,.toml,.ini,.env" multiple style="display:none">
      <button class="tool-btn" id="voiceBtn" aria-label="voice input" title="Voice input">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
      </button>
      <div class="composer-spacer"></div>
      <button class="send" id="sendBtn" aria-label="send" disabled>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/></svg>
      </button>
      <button class="stop" id="stopBtn" aria-label="stop generation" title="Stop generating" style="display:none">
        <svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>
      </button>
    </div>
  </div>
  <div class="shortcut-hint">Tap send. Ctrl/Cmd+Enter also sends.</div>
</div>

<div class="modal-back" id="welcomeModal">
  <div class="modal">
    <h2>Welcome.</h2>
    <p class="sub">Your chats live in the cloud, tied to a personal sync key. Use the same key on any device.</p>
    <div class="tabs">
      <button class="tab active" data-tab="new">First time</button>
      <button class="tab" data-tab="have">I have a key</button>
    </div>
    <div id="tab-new">
      <div class="section-label">Your new sync key</div>
      <div class="key-display" id="newKey"></div>
      <p class="sub" style="font-size:12px;margin-top:10px">Save this somewhere — password manager, a note, anywhere. It's the only way back on another device.</p>
      <div class="actions">
        <button class="btn btn-ghost" id="regenKey">Regenerate</button>
        <button class="btn btn-primary" id="useNewKey">Saved it, continue</button>
      </div>
    </div>
    <div id="tab-have" style="display:none">
      <div class="section-label">Enter your sync key</div>
      <input type="text" id="existingKey" placeholder="word-word-word-word" autocapitalize="off" autocorrect="off" spellcheck="false">
      <div class="actions">
        <button class="btn btn-primary" id="useExistingKey">Continue</button>
      </div>
    </div>
  </div>
</div>

<div class="modal-back" id="settingsModal">
  <div class="modal">
    <h2>Settings</h2>
    <p class="sub">Modes, providers, skills, memory, sync.</p>
    <div class="section-label">Default mode</div>
    <div class="model-grid" id="modelGrid"></div>
    <div class="section-label">Active skills</div>
    <div class="skill-list" id="skillList"></div>
    <div class="section-label">Provider keys</div>
    <div class="providers-status" id="providersStatus"></div>
    <div class="divider"></div>
    <div class="section-label">Sync key (use on other devices)</div>
    <div class="key-display" id="myKey"></div>
    <div class="actions" style="margin-top:8px">
      <button class="btn btn-ghost" id="copyKey">Copy key</button>
    </div>
    <div class="divider"></div>
    <div class="section-label">What I remember about you</div>
    <div class="mem-list" id="memList"></div>
    <div class="mem-add-row">
      <input type="text" id="memInput" class="mem-input" placeholder="Add a memory: e.g. &quot;User prefers Python over JS&quot;" maxlength="190">
      <button class="btn btn-primary mem-add-btn" id="memAddBtn" aria-label="add memory">+</button>
    </div>
    <div class="actions" style="margin-top:12px">
      <button class="btn btn-ghost" id="clearMem">Clear all</button>
      <button class="btn btn-ghost" id="exportAll">Export chats</button>
    </div>
    <div class="divider"></div>
    <div class="section-label">About</div>
    <div class="about-block">
      <strong>Router v7.1</strong> — multi-provider fallback with skills.<br>
      Drops files, reads documents, chains across free tiers.<br>
      <span id="aboutVersion" style="display:block;margin-top:6px;color:var(--text-faint)"></span>
    </div>
    <div class="actions" style="margin-top:18px">
      <button class="btn btn-primary" id="closeSettings">Done</button>
    </div>
  </div>
</div>

<div class="modal-back" id="confirmModal">
  <div class="modal" style="max-width:380px">
    <h2 id="confirmTitle">Sure?</h2>
    <p class="sub" id="confirmMessage">This can't be undone.</p>
    <div class="actions">
      <button class="btn btn-ghost" id="confirmCancel">Cancel</button>
      <button class="btn btn-danger" id="confirmOK">Yes</button>
    </div>
  </div>
</div>

<div class="toast" id="toast"><span class="toast-icon" id="toastIcon"></span><span id="toastText"></span></div>
<div class="lightbox" id="lightbox"><img id="lightboxImg" alt=""></div>

<script>
const $ = (id) => document.getElementById(id);

// ===== Toast =====
let toastTimer;
function toast(msg, type) {
  type = type || "info";
  const t = $("toast");
  const icons = { success: "✓", error: "✕", warn: "!", info: "•" };
  $("toastIcon").textContent = icons[type] || "•";
  $("toastText").textContent = msg;
  t.className = "toast show " + type;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(function(){ t.classList.remove("show"); }, 2600);
}

// ===== Custom confirm =====
let confirmResolver = null;
function customConfirm(title, message, btnText, danger) {
  return new Promise(function(resolve){
    $("confirmTitle").textContent = title;
    $("confirmMessage").textContent = message;
    const okBtn = $("confirmOK");
    okBtn.textContent = btnText || "Confirm";
    okBtn.className = "btn " + (danger !== false ? "btn-danger" : "btn-primary");
    $("confirmModal").classList.add("open");
    confirmResolver = resolve;
  });
}
function closeConfirm(result) {
  $("confirmModal").classList.remove("open");
  if (confirmResolver) { confirmResolver(result); confirmResolver = null; }
}
$("confirmCancel").onclick = function(){ closeConfirm(false); };
$("confirmOK").onclick = function(){ closeConfirm(true); };

// ===== Sync key =====
const SK_STORE = "router_sync_key_v1";
const PREFS_STORE = "router_prefs_v1";
const WORDS = ["amber","arrow","aspen","autumn","azure","basil","birch","blade","blaze","bloom","brave","brick","bridge","brook","calm","canyon","cedar","cherry","clay","clear","cloud","clover","coast","comet","copper","coral","cosmos","crane","creek","crimson","crystal","cypress","dawn","deep","desert","drift","dune","dusk","ember","emerald","fable","falcon","feather","fern","flame","flint","forest","frost","glade","golden","granite","grove","harbor","harmony","haven","hazel","heron","horizon","indigo","ivory","jade","jasper","juniper","keen","lake","lantern","laurel","leaf","light","linen","lupine","maple","marble","meadow","midnight","mist","moon","moss","mountain","north","oak","ocean","olive","onyx","opal","orchid","otter","pearl","pebble","petal","pine","plum","poppy","prairie","quartz","quiet","quill","rain","rapid","raven","reed","ridge","ripple","river","robin","rose","rust","saffron","sage","sapphire","scarlet","shade","shadow","shore","silver","sky","slate","snow","sparrow","spring","spruce","star","stone","storm","stream","summer","sunset","swift","teal","thicket","thorn","thunder","tide","tiger","topaz","trail","tundra","valley","velvet","violet","vista","walnut","wave","wheat","whisper","willow","wind","winter","wren"];
function genKey() {
  const p = function(){ return WORDS[Math.floor(Math.random() * WORDS.length)]; };
  return p() + "-" + p() + "-" + p() + "-" + p();
}

// ===== Models for picker =====
const MODELS_CLIENT = {
  chat:       { label: "Chat (auto)",         blurb: "Auto-routes Cerebras → Groq → Gemini → OpenRouter. Falls over instantly on rate limit.", chain: ["Cerebras", "Groq", "Gemini", "OpenRouter"] },
  uncensored: { label: "Uncensored (Venice)", blurb: "Venice via OpenRouter. No filter. Limited daily quota — use deliberately.",                chain: ["OpenRouter"] },
  reasoner:   { label: "Deep Reasoner",       blurb: "For hard multi-step problems. OpenRouter free → Cerebras Qwen Thinking → Gemini.",         chain: ["OpenRouter", "Cerebras", "Gemini"] },
};

// ===== Markdown =====
marked.setOptions({ breaks: true, gfm: true });
function renderMarkdown(t) { return DOMPurify.sanitize(marked.parse(t || ""), { ADD_ATTR: ["target"] }); }

function decorateCodeBlocks(container) {
  const pres = container.querySelectorAll("pre");
  for (let i = 0; i < pres.length; i++) {
    const pre = pres[i];
    if (pre.querySelector(".copy-btn")) continue;
    const codeEl = pre.querySelector("code");
    if (codeEl) {
      const cls = codeEl.className || "";
      const m = cls.match(/language-(\\S+)/);
      if (m) {
        const lang = document.createElement("span");
        lang.className = "code-lang";
        lang.textContent = m[1];
        pre.appendChild(lang);
      }
    }
    const b = document.createElement("button");
    b.className = "copy-btn";
    b.textContent = "copy";
    b.onclick = function(e){
      e.stopPropagation();
      const code = (pre.querySelector("code") || pre).innerText;
      navigator.clipboard.writeText(code).then(function(){
        b.textContent = "copied";
        setTimeout(function(){ b.textContent = "copy"; }, 1400);
      });
    };
    pre.appendChild(b);
  }
}

// ===== State =====
let syncKey = localStorage.getItem(SK_STORE) || "";
let prefs = JSON.parse(localStorage.getItem(PREFS_STORE) || "{}");
if (prefs.model === "cerebras_llama" || prefs.model === "cerebras_gpt" || prefs.model === "cerebras_scout") prefs.model = "chat";
if (prefs.model === "venice") prefs.model = "uncensored";
if (prefs.model === "r1") prefs.model = "reasoner";
if (!prefs.model || !MODELS_CLIENT[prefs.model]) prefs.model = "chat";

function savePrefs() {
  localStorage.setItem(PREFS_STORE, JSON.stringify(prefs));
  $("modelLabel").textContent = (MODELS_CLIENT[prefs.model] || {}).label || prefs.model;
}

let currentChatId = null;
let currentMessages = [];
let isStreaming = false;
let pendingFiles = []; // { kind: "image"|"pdf"|"text", name, mimeType, data?, content?, size }
let currentAbort = null;
let userMemoryName = null;
let useSearchNext = false;
let useReasonNext = false;
let healthCache = null;

function uuid() { return Date.now().toString(36) + Math.random().toString(36).slice(2, 8); }
function formatBytes(n) {
  if (!n) return "";
  if (n < 1024) return "< 1 KB";
  if (n < 1024 * 1024) return Math.round(n / 1024) + " KB";
  return (n / (1024 * 1024)).toFixed(1) + " MB";
}

// ===== API =====
async function api(path, opts) {
  const r = await fetch(path, opts || {});
  if (!r.ok) throw new Error("HTTP " + r.status);
  return r.json();
}
function apiList()       { return api("/chats?sk=" + encodeURIComponent(syncKey)); }
function apiGet(id)      { return api("/chats/" + id + "?sk=" + encodeURIComponent(syncKey)); }
function apiSave(c)      { return api("/chats/save", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ sync_key: syncKey, chat: c }) }); }
function apiDel(id)      { return api("/chats/" + id + "?sk=" + encodeURIComponent(syncKey), { method: "DELETE" }); }
function apiPin(id, p)   { return api("/chats/" + id + "/pin", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ sync_key: syncKey, pinned: p }) }); }
function apiMemGet()     { return api("/memories?sk=" + encodeURIComponent(syncKey)); }
function apiMemSave(m)   { return api("/memories/save", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ sync_key: syncKey, memories: m }) }); }
function apiHealth()     { return api("/health"); }

// ===== Greetings =====
function smartGreeting() {
  const h = new Date().getHours();
  const name = userMemoryName ? ", " + userMemoryName : "";
  let pool;
  if (h < 5) pool = ["Still up" + name + "?", "Late one tonight" + name + ".", "Burning the midnight oil" + name + "?"];
  else if (h < 9) pool = ["Morning" + name + ".", "Up early" + name + ".", "Hello" + name + "."];
  else if (h < 12) pool = ["Hey" + name + ".", "Morning" + name + ".", "What's the move" + name + "?"];
  else if (h < 17) pool = ["Hey" + name + ".", "Afternoon" + name + ".", "Hello" + name + "."];
  else if (h < 22) pool = ["Evening" + name + ".", "Welcome back" + name + ".", "Long day" + name + "?"];
  else pool = ["Evening" + name + ".", "Hey" + name + ".", "Still going" + name + "?"];
  return pool[Math.floor(Math.random() * pool.length)];
}

async function extractNameFromMem() {
  try {
    const mems = await apiMemGet();
    for (let i = 0; i < mems.length; i++) {
      const m = mems[i];
      // Memories are stored in third-person ("User's name is Sandy", "Name: Sandy", "Called Sandy"),
      // so we only match those forms — no first-person "I'm X" patterns slip in.
      const match = m.match(/(?:user(?:'s)?\\s+name\\s+is|user\\s+is\\s+called|name\\s*:)\\s+([A-Z][a-zA-Z]+)/);
      if (match) { userMemoryName = match[1]; return; }
    }
  } catch (e) {}
}

// ===== Cycling placeholder =====
const PLACEHOLDERS = [
  "Ask anything…",
  "What's on your mind?",
  "Throw me a problem…",
  "What do you want to know?",
  "Drop a file, ask a question…",
  "I'm listening…",
  "Hit me with something…",
  "What are we figuring out?",
];
let placeholderTimer = null;
function startCyclingPlaceholder() {
  const input = $("input");
  let idx = Math.floor(Math.random() * PLACEHOLDERS.length);
  input.placeholder = PLACEHOLDERS[idx];
  clearInterval(placeholderTimer);
  placeholderTimer = setInterval(function(){
    if (input.value || document.activeElement === input) return;
    idx = (idx + 1) % PLACEHOLDERS.length;
    input.style.transition = "opacity .3s";
    input.style.opacity = "0";
    setTimeout(function(){
      input.placeholder = PLACEHOLDERS[idx];
      input.style.opacity = "1";
    }, 300);
  }, 4500);
}

// ===== Empty state =====
function renderEmpty() {
  const root = $("messagesInner");
  root.innerHTML = "";
  const e = document.createElement("div");
  e.className = "empty";
  const greet = document.createElement("div");
  greet.className = "greet";
  greet.textContent = smartGreeting();
  const sub = document.createElement("div");
  sub.className = "sub";
  sub.textContent = "Drop files, ask questions, search the web, think deep. Whatever.";
  const caps = document.createElement("div");
  caps.className = "cap-row";
  caps.innerHTML = '<span class="cap-chip">🖼 Images</span><span class="cap-chip">📄 PDFs</span><span class="cap-chip">📝 Text & code</span><span class="cap-chip">🔍 Web search</span><span class="cap-chip">🧠 Deep reasoning</span>';
  e.appendChild(greet);
  e.appendChild(sub);
  e.appendChild(caps);
  root.appendChild(e);
}

// ===== Smart scroll =====
const messagesEl = function(){ return $("messages"); };
let stickToBottom = true;
let pendingScrollFrame = null;

function isAtBottom() {
  const m = messagesEl();
  return m.scrollHeight - m.scrollTop - m.clientHeight < 80;
}

function setupScrollWatcher() {
  const m = messagesEl();
  m.addEventListener("scroll", function(){
    stickToBottom = isAtBottom();
    $("scrollPill").classList.toggle("visible", !stickToBottom && isStreaming);
  }, { passive: true });
}

function maybeScroll() {
  if (!stickToBottom) return;
  if (pendingScrollFrame) return;
  pendingScrollFrame = requestAnimationFrame(function(){
    pendingScrollFrame = null;
    const m = messagesEl();
    m.scrollTop = m.scrollHeight;
  });
}

function forceScrollBottom() {
  stickToBottom = true;
  const m = messagesEl();
  m.scrollTop = m.scrollHeight;
  $("scrollPill").classList.remove("visible");
}

$("scrollPill").onclick = forceScrollBottom;

// ===== Message bubbles =====
function makeBubble(role, content, files, meta, isStreamingFlag) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + role;
  const b = document.createElement("div");
  const isError = role === "assistant" && content && /^(Error:|⚠)/.test(content);
  b.className = "bubble" + (isStreamingFlag ? " empty-streaming" : "") + (isError ? " error-bubble" : "");

  // Render attached files
  if (Array.isArray(files)) {
    for (let i = 0; i < files.length; i++) {
      const f = files[i];
      if (f.kind === "image" && f.data) {
        const img = document.createElement("img");
        img.src = "data:" + f.mimeType + ";base64," + f.data;
        img.className = "msg-img";
        img.alt = "uploaded";
        img.onclick = function(){ $("lightboxImg").src = img.src; $("lightbox").classList.add("open"); };
        b.appendChild(img);
      } else if (f.kind === "pdf" || f.kind === "text") {
        const card = document.createElement("div");
        card.className = "msg-file";
        const ext = (f.name || "").split(".").pop().toUpperCase().slice(0, 4);
        const sizeKB = formatBytes(f.size);
        card.innerHTML = '<div class="file-icon">' + (f.kind === "pdf" ? "PDF" : (ext || "TXT")) + '</div><div class="file-info"><div class="file-name"></div><div class="file-meta">' + (f.kind === "pdf" ? "Document" : "Text") + (sizeKB ? " · " + sizeKB : "") + '</div></div>';
        card.querySelector(".file-name").textContent = f.name || "file";
        b.appendChild(card);
      }
    }
  }

  if (role === "assistant") {
    if (isStreamingFlag && !content) {
      const t = document.createElement("div");
      t.className = "thinking";
      t.innerHTML = "<span></span><span></span><span></span>";
      b.appendChild(t);
    } else {
      const div = document.createElement("div");
      div.innerHTML = renderMarkdown(content);
      decorateCodeBlocks(div);
      b.appendChild(div);
    }
  } else if (content) {
    const t = document.createElement("div");
    t.textContent = content;
    t.style.whiteSpace = "pre-wrap";
    b.appendChild(t);
  }

  wrap.appendChild(b);
  if (role === "assistant" && meta && (meta.responder || meta.tool_used || meta.stopped)) {
    const cap = document.createElement("div");
    cap.className = "caption";
    cap.innerHTML = captionHTML(meta);
    wrap.appendChild(cap);
  }
  return { wrap: wrap, bubble: b };
}

function captionHTML(meta) {
  let s = "via " + (meta.responder || "lead");
  if (meta.stopped) s += ' <span class="badge b-stopped">stopped</span>';
  if (meta.fallback_used) s += ' <span class="badge b-fallback">fallback</span>';
  if (meta.tool_used === "web_search") s += ' <span class="badge b-search">searched</span>';
  if (meta.tool_used === "ask_reasoner") s += ' <span class="badge b-reason">reasoner</span>';
  if (Array.isArray(meta.skills_active)) {
    for (let i = 0; i < meta.skills_active.length; i++) {
      s += ' <span class="badge b-skill">' + meta.skills_active[i] + '</span>';
    }
  }
  return s;
}

function renderAll() {
  const root = $("messagesInner");
  if (currentMessages.length === 0) { renderEmpty(); return; }
  root.innerHTML = "";
  for (let i = 0; i < currentMessages.length; i++) {
    const m = currentMessages[i];
    const built = makeBubble(m.role, m.content, m.files, m.meta, false);
    const isLast = i === currentMessages.length - 1;
    const isError = m.role === "assistant" && m.content && /^(Error:|⚠)/.test(m.content);
    if (isLast && !isStreaming && m.role === "assistant") {
      const actions = document.createElement("div");
      actions.className = "msg-actions";
      if (isError) {
        const retry = document.createElement("button");
        retry.className = "retry";
        retry.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg> Retry';
        retry.onclick = retryLast;
        actions.appendChild(retry);
      } else {
        const regen = document.createElement("button");
        regen.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg> Regenerate';
        regen.onclick = regenerate;
        const copy = document.createElement("button");
        copy.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy';
        copy.onclick = function(){ navigator.clipboard.writeText(m.content || ""); toast("Copied", "success"); };
        actions.appendChild(regen);
        actions.appendChild(copy);
      }
      built.wrap.appendChild(actions);
    }
    root.appendChild(built.wrap);
  }
  forceScrollBottom();
}

// ===== Chat list =====
let allChatsCache = [];
async function renderList(filter) {
  filter = filter || "";
  if (!syncKey) return;
  try {
    if (!filter) allChatsCache = await apiList();
    const f = filter.toLowerCase().trim();
    const chats = f
      ? allChatsCache.filter(function(c){ return (c.title || "").toLowerCase().includes(f) || (c.preview || "").toLowerCase().includes(f); })
      : allChatsCache;
    const root = $("chatsList");
    root.innerHTML = "";
    if (chats.length === 0) {
      const e = document.createElement("div");
      e.className = "list-empty";
      e.textContent = f ? "No matches." : "No chats yet.";
      root.appendChild(e);
      return;
    }
    for (let i = 0; i < chats.length; i++) {
      const c = chats[i];
      const it = document.createElement("div");
      it.className = "chat-item" + (c.id === currentChatId ? " active" : "");
      const row = document.createElement("div");
      row.className = "row";
      const tt = document.createElement("div");
      tt.className = "chat-title-row";
      if (c.pinned) { const p = document.createElement("span"); p.className = "pin-icon"; p.textContent = "📌"; tt.appendChild(p); }
      const ts = document.createElement("span");
      ts.textContent = c.title || "New chat";
      ts.style.cssText = "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1";
      tt.appendChild(ts);
      const p2 = document.createElement("div");
      p2.className = "chat-prev";
      p2.textContent = c.preview || "";
      row.appendChild(tt);
      if (c.preview) row.appendChild(p2);
      row.onclick = function(){ loadChat(c.id); };
      const actions = document.createElement("div");
      actions.className = "chat-actions";
      const pinBtn = document.createElement("button");
      pinBtn.title = c.pinned ? "Unpin" : "Pin";
      pinBtn.innerHTML = c.pinned
        ? '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2a3 3 0 0 0-3 3v6l-3 3v2h12v-2l-3-3V5a3 3 0 0 0-3-3zm-1 17v3l1 1 1-1v-3h-2z"/></svg>'
        : '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 0-3 3v6l-3 3v2h12v-2l-3-3V5a3 3 0 0 0-3-3zm-1 17v3l1 1 1-1v-3"/></svg>';
      pinBtn.onclick = async function(e){
        e.stopPropagation();
        try { await apiPin(c.id, !c.pinned); renderList(filter); } catch (err) { toast("Couldn't pin", "error"); }
      };
      const delBtn = document.createElement("button");
      delBtn.className = "del";
      delBtn.title = "Delete";
      delBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-2 14a2 2 0 0 1-2 2H9a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6M14 11v6"/><path d="M9 6V4a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2"/></svg>';
      delBtn.onclick = async function(e){
        e.stopPropagation();
        const ts2 = (c.title || "this chat").slice(0, 30);
        if (await customConfirm("Delete chat?", '"' + ts2 + '" will be gone for good.', "Delete")) {
          try {
            await apiDel(c.id);
            if (c.id === currentChatId) newChat();
            renderList(filter);
            toast("Deleted", "success");
          } catch (err) { toast("Couldn't delete", "error"); }
        }
      };
      actions.appendChild(pinBtn);
      actions.appendChild(delBtn);
      it.appendChild(row);
      it.appendChild(actions);
      root.appendChild(it);
    }
  } catch (e) { console.error(e); }
}

function newChat() {
  currentChatId = null;
  currentMessages = [];
  pendingFiles = [];
  useSearchNext = false;
  useReasonNext = false;
  updateToggles();
  updateAttachCount();
  renderAttachedChips();
  $("chatTitle").textContent = "Router";
  $("input").value = "";
  autoresize();
  updateSendEnabled();
  renderAll();
  renderList();
  closeSB();
}

async function loadChat(id) {
  try {
    const c = await apiGet(id);
    if (!c) return;
    currentChatId = id;
    currentMessages = c.messages || [];
    $("chatTitle").textContent = c.title || "Chat";
    renderAll();
    renderList();
    closeSB();
  } catch (e) { toast("Couldn't load chat", "error"); }
}

async function persist() {
  if (!currentChatId) currentChatId = uuid();
  const firstUser = currentMessages.find(function(m){ return m.role === "user"; });
  const title = (firstUser && firstUser.content) ? firstUser.content.slice(0, 50) : "New chat";
  // Strip binary data from attachments; keep names/kinds so we can render placeholders.
  // We do NOT append a "[N files attached]" note here — the server does that exactly once,
  // with an idempotency check, so the note never doubles up across saves.
  const storage = currentMessages.map(function(m){
    if (Array.isArray(m.files) && m.files.length) {
      const stripped = m.files.map(function(f){
        if (f.kind === "text") return f;
        return { kind: f.kind, name: f.name, mimeType: f.mimeType, size: f.size };
      });
      const out = { role: m.role, content: m.content || "", files: stripped };
      if (m.role === "assistant" && m.meta) out.meta = m.meta;
      if (m.searchIntent) out.searchIntent = true;
      if (m.reasonIntent) out.reasonIntent = true;
      return out;
    }
    return m;
  });
  try {
    await apiSave({ id: currentChatId, title: title, messages: storage, updated: Date.now(), pinned: false });
    $("chatTitle").textContent = title;
    renderList();
  } catch (e) {
    console.warn("persist failed:", e);
    toast("Couldn't save chat", "warn");
  }
}

// ===== Sidebar =====
function openSB() { $("sidebar").classList.add("open"); $("backdrop").classList.add("open"); }
function closeSB() { $("sidebar").classList.remove("open"); $("backdrop").classList.remove("open"); }

// ===== File handling =====
const TEXT_EXTS = ["txt","md","markdown","csv","json","html","htm","xml","yaml","yml","py","js","ts","tsx","jsx","css","scss","sass","go","rs","java","c","cpp","h","hpp","sh","bash","rb","php","sql","kt","swift","toml","ini","env","log","conf","cfg","dockerfile","makefile"];
function detectKind(file) {
  const t = (file.type || "").toLowerCase();
  const name = (file.name || "").toLowerCase();
  const ext = name.split(".").pop();
  if (t.startsWith("image/")) return "image";
  if (t === "application/pdf" || ext === "pdf") return "pdf";
  if (t.startsWith("text/") || TEXT_EXTS.indexOf(ext) !== -1) return "text";
  if (t === "application/json") return "text";
  return null;
}
function fileToBase64(file) {
  return new Promise(function(res, rej){
    const r = new FileReader();
    r.onload = function(){ res(r.result.split(",")[1]); };
    r.onerror = rej;
    r.readAsDataURL(file);
  });
}
function fileToText(file) {
  return new Promise(function(res, rej){
    const r = new FileReader();
    r.onload = function(){ res(r.result); };
    r.onerror = rej;
    r.readAsText(file);
  });
}

async function addFiles(fileList) {
  for (let i = 0; i < fileList.length; i++) {
    const f = fileList[i];
    const kind = detectKind(f);
    if (!kind) { toast("Skipping " + f.name + " (unsupported)", "warn"); continue; }
    if (kind === "image" && f.size > 4 * 1024 * 1024) { toast("Image " + f.name + " too big (4MB max)", "warn"); continue; }
    if (kind === "pdf" && f.size > 8 * 1024 * 1024) { toast("PDF " + f.name + " too big (8MB max)", "warn"); continue; }
    if (kind === "text" && f.size > 1 * 1024 * 1024) { toast("Text file " + f.name + " too big (1MB max)", "warn"); continue; }
    try {
      if (kind === "text") {
        const text = await fileToText(f);
        pendingFiles.push({ kind: "text", name: f.name, mimeType: f.type || "text/plain", content: text, size: f.size });
      } else {
        const data = await fileToBase64(f);
        pendingFiles.push({ kind: kind, name: f.name, mimeType: f.type || (kind === "pdf" ? "application/pdf" : "image/jpeg"), data: data, size: f.size });
      }
    } catch (e) { toast("Couldn't read " + f.name, "error"); }
  }
  renderAttachedChips();
  updateAttachCount();
  updateSendEnabled();
}

function renderAttachedChips() {
  const root = $("attachedChips");
  root.innerHTML = "";
  for (let i = 0; i < pendingFiles.length; i++) {
    (function(idx){
      const f = pendingFiles[idx];
      const chip = document.createElement("div");
      chip.className = "chip";
      const thumb = document.createElement("div");
      thumb.className = "thumb";
      if (f.kind === "image") {
        const im = document.createElement("img");
        im.src = "data:" + f.mimeType + ";base64," + f.data;
        thumb.appendChild(im);
      } else if (f.kind === "pdf") {
        thumb.textContent = "PDF";
      } else {
        const ext = (f.name || "").split(".").pop().toUpperCase().slice(0, 3);
        thumb.textContent = ext || "TXT";
      }
      const info = document.createElement("div");
      info.className = "chip-info";
      const nm = document.createElement("div");
      nm.className = "chip-name";
      nm.textContent = f.name;
      const sz = document.createElement("div");
      sz.className = "chip-size";
      sz.textContent = (formatBytes(f.size) || "") + " · " + f.kind;
      info.appendChild(nm);
      info.appendChild(sz);
      const rm = document.createElement("button");
      rm.className = "rm";
      rm.textContent = "×";
      rm.onclick = function(){
        pendingFiles.splice(idx, 1);
        renderAttachedChips();
        updateAttachCount();
        updateSendEnabled();
      };
      chip.appendChild(thumb);
      chip.appendChild(info);
      chip.appendChild(rm);
      root.appendChild(chip);
    })(i);
  }
}

function updateAttachCount() {
  const c = $("attachCount");
  if (pendingFiles.length === 0) c.style.display = "none";
  else { c.style.display = "flex"; c.textContent = String(pendingFiles.length); }
}

$("attachBtn").onclick = function(){ $("fileInput").click(); };
$("fileInput").onchange = async function(e){ await addFiles(e.target.files); e.target.value = ""; };

document.addEventListener("dragover", function(e){ e.preventDefault(); });
document.addEventListener("drop", async function(e){ e.preventDefault(); if (e.dataTransfer && e.dataTransfer.files) await addFiles(e.dataTransfer.files); });
$("input").addEventListener("paste", async function(e){
  if (!e.clipboardData) return;
  const items = e.clipboardData.items;
  const files = [];
  for (let i = 0; i < items.length; i++) {
    if (items[i].type.startsWith("image/")) { const f = items[i].getAsFile(); if (f) files.push(f); }
  }
  if (files.length) { e.preventDefault(); await addFiles(files); }
});

// ===== Toggles =====
function updateToggles() {
  $("searchToggle").classList.toggle("active", useSearchNext);
  $("reasonToggle").classList.toggle("active", useReasonNext);
}
$("searchToggle").onclick = function(){ useSearchNext = !useSearchNext; updateToggles(); };
$("reasonToggle").onclick = function(){ useReasonNext = !useReasonNext; updateToggles(); };

// ===== Voice =====
let recog = null;
function setupVoice() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { $("voiceBtn").style.display = "none"; return; }
  recog = new SR();
  recog.continuous = false;
  recog.interimResults = true;
  recog.lang = "en-US";
  let base = "";
  recog.onstart = function(){ base = $("input").value; $("voiceBtn").classList.add("recording"); };
  recog.onresult = function(ev){
    let interim = "", finalT = "";
    for (let i = ev.resultIndex; i < ev.results.length; i++) {
      const t = ev.results[i][0].transcript;
      if (ev.results[i].isFinal) finalT += t; else interim += t;
    }
    $("input").value = (base ? base + " " : "") + finalT + interim;
    autoresize();
    updateSendEnabled();
  };
  recog.onerror = function(){ $("voiceBtn").classList.remove("recording"); };
  recog.onend = function(){ $("voiceBtn").classList.remove("recording"); };
}
$("voiceBtn").onclick = function(){
  if (!recog) { toast("Voice not supported here", "warn"); return; }
  if ($("voiceBtn").classList.contains("recording")) recog.stop();
  else { try { recog.start(); } catch (e) { toast("Already listening", "info"); } }
};

// ===== Input =====
function autoresize() {
  const e = $("input");
  e.style.height = "auto";
  e.style.height = Math.min(e.scrollHeight, 200) + "px";
}
function updateSendEnabled() {
  const has = $("input").value.trim().length > 0 || pendingFiles.length > 0;
  $("sendBtn").disabled = !has || isStreaming;
  // While streaming, hide the send button and show stop in its place.
  $("sendBtn").style.display = isStreaming ? "none" : "";
  $("stopBtn").style.display = isStreaming ? "flex" : "none";
}

$("stopBtn").onclick = function(){
  if (currentAbort && isStreaming) {
    try { currentAbort.abort(); } catch (e) {}
    toast("Stopped", "info");
  }
};

$("input").addEventListener("keydown", function(e){
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); send(); }
});
$("input").addEventListener("input", function(){ autoresize(); updateSendEnabled(); });

// ===== Send / regenerate / retry =====
function reuseLastUserIntent() {
  // Merge current toggle state into the stored intent on the last user message.
  // This means: original intent persists across retries, AND if user toggles
  // search/reason ON before hitting regenerate, that adds to it.
  for (let i = currentMessages.length - 1; i >= 0; i--) {
    if (currentMessages[i].role === "user") {
      if (useSearchNext) currentMessages[i].searchIntent = true;
      if (useReasonNext) currentMessages[i].reasonIntent = true;
      return;
    }
  }
}
async function regenerate() {
  if (isStreaming) return;
  if (currentMessages[currentMessages.length - 1] && currentMessages[currentMessages.length - 1].role === "assistant") {
    currentMessages.pop();
  }
  reuseLastUserIntent();
  renderAll();
  await streamResponse();
}
async function retryLast() {
  if (isStreaming) return;
  if (currentMessages[currentMessages.length - 1] && currentMessages[currentMessages.length - 1].role === "assistant") {
    currentMessages.pop();
  }
  reuseLastUserIntent();
  renderAll();
  await streamResponse();
}

async function send() {
  if (isStreaming) return;
  const input = $("input");
  const text = input.value.trim();
  if (!text && pendingFiles.length === 0) return;
  // Capture search/reason intent on the user message itself so retry/regenerate
  // can reuse it instead of falling through to whatever the globals happen to be.
  currentMessages.push({
    role: "user",
    content: text,
    files: pendingFiles.slice(),
    searchIntent: useSearchNext,
    reasonIntent: useReasonNext,
  });
  pendingFiles = [];
  renderAttachedChips();
  updateAttachCount();
  input.value = "";
  autoresize();
  updateSendEnabled();
  // Reset toggles now that the intent is locked onto the message.
  useSearchNext = false;
  useReasonNext = false;
  updateToggles();
  await streamResponse();
}

async function streamResponse() {
  isStreaming = true;
  updateSendEnabled();
  $("voiceBtn").disabled = true;
  // Find the most recent user message and read intent off it
  let lastUserMsg = null;
  for (let i = currentMessages.length - 1; i >= 0; i--) {
    if (currentMessages[i].role === "user") { lastUserMsg = currentMessages[i]; break; }
  }
  const requestedSearch = !!(lastUserMsg && lastUserMsg.searchIntent);
  const requestedReason = !!(lastUserMsg && lastUserMsg.reasonIntent);
  const aMsg = { role: "assistant", content: "", meta: {} };
  currentMessages.push(aMsg);

  if ($("messagesInner").querySelector(".empty")) $("messagesInner").innerHTML = "";
  $("messagesInner").innerHTML = "";
  for (let i = 0; i < currentMessages.length - 1; i++) {
    const m = currentMessages[i];
    const built = makeBubble(m.role, m.content, m.files, m.meta, false);
    $("messagesInner").appendChild(built.wrap);
  }
  const built = makeBubble("assistant", "", null, null, true);
  const aWrap = built.wrap, aBub = built.bubble;
  $("messagesInner").appendChild(aWrap);
  stickToBottom = true;
  forceScrollBottom();

  currentAbort = new AbortController();

  // Throttled render — coalesces tokens into rAF batches
  let pendingRender = null;
  let renderQueue = "";
  function scheduleRender() {
    if (pendingRender) return;
    pendingRender = requestAnimationFrame(function(){
      pendingRender = null;
      if (renderQueue) {
        aMsg.content += renderQueue;
        renderQueue = "";
      }
      aBub.classList.remove("empty-streaming");
      aBub.innerHTML = renderMarkdown(aMsg.content);
      decorateCodeBlocks(aBub);
      maybeScroll();
    });
  }

  try {
    const sendMessages = currentMessages.slice(0, -1).map(function(m){
      return { role: m.role, content: m.content, files: m.files || [] };
    });
    const resp = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sync_key: syncKey,
        model: prefs.model,
        use_search: requestedSearch,
        use_reasoning: requestedReason,
        messages: sendMessages,
      }),
      signal: currentAbort.signal,
    });
    if (!resp.ok || !resp.body) {
      aMsg.content = "⚠ Server error (HTTP " + resp.status + "). Try again.";
      aBub.classList.add("error-bubble");
      aBub.innerHTML = renderMarkdown(aMsg.content);
      return;
    }
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    let toolEl = null;
    while (true) {
      const chunk = await reader.read();
      if (chunk.done) break;
      buf += dec.decode(chunk.value, { stream: true });
      const parts = buf.split("\\n\\n");
      buf = parts.pop() || "";
      for (let i = 0; i < parts.length; i++) {
        const evt = parts[i];
        const lines = evt.split("\\n");
        let et = "", data = "";
        for (let j = 0; j < lines.length; j++) {
          const ln = lines[j];
          if (ln.startsWith("event: ")) et = ln.slice(7);
          else if (ln.startsWith("data: ")) data = ln.slice(6);
        }
        if (!data) continue;
        let p;
        try { p = JSON.parse(data); } catch (e) { continue; }
        if (et === "status") {
          aMsg.meta.responder = p.leader;
          // Replace the three-dot thinking indicator with a routing label
          // until the first token arrives. Keeps the user informed when
          // the first provider is slow to respond.
          if (aBub.classList.contains("empty-streaming") && !aMsg.content) {
            aBub.innerHTML = '<div class="routing-pill"><span class="rp-dot"></span>Routing via ' + (p.leader || "lead") + '…</div>';
          }
        } else if (et === "note") {
          if (toolEl) toolEl.remove();
          toolEl = document.createElement("div");
          toolEl.className = "tool-running note";
          toolEl.textContent = "→ " + p.msg;
          aWrap.insertBefore(toolEl, aBub);
          setTimeout(function(){ if (toolEl) { toolEl.remove(); toolEl = null; } }, 3800);
        } else if (et === "token") {
          renderQueue += p.text;
          scheduleRender();
        } else if (et === "tool_start") {
          aMsg.meta.tool_used = p.tool;
          if (toolEl) toolEl.remove();
          toolEl = document.createElement("div");
          toolEl.className = "tool-running " + (p.tool === "web_search" ? "search" : "reason");
          toolEl.innerHTML = (p.tool === "web_search" ? "🔍 searching the web…" : "🧠 reasoning…");
          aWrap.insertBefore(toolEl, aBub);
        } else if (et === "tool_done") {
          if (toolEl) { toolEl.remove(); toolEl = null; }
        } else if (et === "done") {
          Object.assign(aMsg.meta, p);
        } else if (et === "aborted") {
          // Server saw the abort and stopped cleanly. The catch below will also handle
          // the AbortError thrown by fetch, but we set a flag here so partial content
          // is tagged correctly even if the stream ended via this event first.
          aMsg.meta.stopped = true;
        } else if (et === "error") {
          aMsg.content += (aMsg.content ? "\\n\\n" : "") + "⚠ " + p.message;
          aBub.classList.add("error-bubble");
          aBub.innerHTML = renderMarkdown(aMsg.content);
        }
      }
    }
    // Final flush
    if (pendingRender) { cancelAnimationFrame(pendingRender); pendingRender = null; }
    if (renderQueue) { aMsg.content += renderQueue; renderQueue = ""; }
    if (toolEl) toolEl.remove();
    if (!aMsg.content) aMsg.content = aMsg.meta.stopped ? "_(stopped before any response)_" : "_(empty response — try regenerate)_";
    aBub.classList.remove("empty-streaming");
    aBub.innerHTML = renderMarkdown(aMsg.content);
    decorateCodeBlocks(aBub);
    if (aMsg.meta.responder || aMsg.meta.tool_used || aMsg.meta.stopped) {
      const cap = document.createElement("div");
      cap.className = "caption";
      cap.innerHTML = captionHTML(aMsg.meta);
      aWrap.appendChild(cap);
    }
    await persist();
  } catch (e) {
    if (e.name === "AbortError") {
      // User hit Stop. Keep any partial content; mark the message as stopped.
      aMsg.meta.stopped = true;
      if (pendingRender) { cancelAnimationFrame(pendingRender); pendingRender = null; }
      if (renderQueue) { aMsg.content += renderQueue; renderQueue = ""; }
      if (!aMsg.content) aMsg.content = "_(stopped)_";
      aBub.classList.remove("empty-streaming");
      aBub.innerHTML = renderMarkdown(aMsg.content);
      decorateCodeBlocks(aBub);
      if (!aWrap.querySelector(".caption")) {
        const cap = document.createElement("div");
        cap.className = "caption";
        cap.innerHTML = captionHTML(aMsg.meta);
        aWrap.appendChild(cap);
      }
      try { await persist(); } catch (_) {}
    } else if (e.message && !e.message.startsWith("HTTP")) {
      aMsg.content = "⚠ Network glitch: " + e.message + ". Try again.";
      aBub.classList.add("error-bubble");
      aBub.innerHTML = renderMarkdown(aMsg.content);
    }
  } finally {
    isStreaming = false;
    currentAbort = null;
    $("voiceBtn").disabled = false;
    useSearchNext = false;
    useReasonNext = false;
    updateToggles();
    updateSendEnabled();
    $("scrollPill").classList.remove("visible");
    // No full renderAll() — we keep the live bubble and just append actions to it.
    // (Calling renderAll() here wiped the DOM and rebuilt it, causing a visible flicker.)
    appendActionsForLastMessage(aWrap, aMsg);
  }
}

// Append Copy/Regenerate (or Retry on error) buttons to the live bubble wrap,
// so the end of streaming doesn't require a full DOM rebuild.
function appendActionsForLastMessage(aWrap, aMsg) {
  if (!aWrap || aWrap.querySelector(".msg-actions")) return;
  const isError = aMsg && aMsg.content && /^(Error:|⚠)/.test(aMsg.content);
  const actions = document.createElement("div");
  actions.className = "msg-actions";
  if (isError) {
    const retry = document.createElement("button");
    retry.className = "retry";
    retry.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg> Retry';
    retry.onclick = retryLast;
    actions.appendChild(retry);
  } else {
    const regen = document.createElement("button");
    regen.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg> Regenerate';
    regen.onclick = regenerate;
    const copy = document.createElement("button");
    copy.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy';
    copy.onclick = function(){ navigator.clipboard.writeText(aMsg.content || ""); toast("Copied", "success"); };
    actions.appendChild(regen);
    actions.appendChild(copy);
  }
  aWrap.appendChild(actions);
}

$("sendBtn").onclick = send;

// ===== Welcome =====
function showWelcome() { $("welcomeModal").classList.add("open"); refreshNewKey(); }
function refreshNewKey() { $("newKey").textContent = genKey(); }
function pickTab(name) {
  const tabs = document.querySelectorAll(".tab");
  for (let i = 0; i < tabs.length; i++) tabs[i].classList.toggle("active", tabs[i].dataset.tab === name);
  $("tab-new").style.display = name === "new" ? "" : "none";
  $("tab-have").style.display = name === "have" ? "" : "none";
}
const tabs = document.querySelectorAll(".tab");
for (let i = 0; i < tabs.length; i++) tabs[i].onclick = function(){ pickTab(this.dataset.tab); };
$("regenKey").onclick = refreshNewKey;
$("useNewKey").onclick = async function(){
  syncKey = $("newKey").textContent.trim();
  localStorage.setItem(SK_STORE, syncKey);
  $("welcomeModal").classList.remove("open");
  toast("Welcome.", "success");
  await renderList();
  renderAll();
};
$("useExistingKey").onclick = async function(){
  const v = $("existingKey").value.trim();
  if (v.length < 4) { toast("That doesn't look right", "warn"); return; }
  syncKey = v;
  localStorage.setItem(SK_STORE, syncKey);
  $("welcomeModal").classList.remove("open");
  toast("Synced.", "success");
  await renderList();
  await extractNameFromMem();
  renderAll();
};

// ===== Settings =====
async function buildModelGrid() {
  if (!healthCache) { try { healthCache = await apiHealth(); } catch (e) { healthCache = {}; } }
  const grid = $("modelGrid");
  grid.innerHTML = "";
  const keys = Object.keys(MODELS_CLIENT);
  for (let i = 0; i < keys.length; i++) {
    (function(key){
      const info = MODELS_CLIENT[key];
      const card = document.createElement("button");
      card.className = "model-card" + (prefs.model === key ? " selected" : "");
      const chainStr = info.chain.join(" → ");
      card.innerHTML = '<div class="info"><div class="name"></div><div class="blurb"></div><div class="chain-list"></div></div><div class="check">✓</div>';
      card.querySelector(".name").textContent = info.label;
      card.querySelector(".blurb").textContent = info.blurb;
      card.querySelector(".chain-list").textContent = chainStr;
      card.onclick = function(){
        prefs.model = key;
        savePrefs();
        buildModelGrid();
        toast("Mode: " + info.label, "success");
      };
      grid.appendChild(card);
    })(keys[i]);
  }
}

function buildSkillList() {
  const skills = [
    { name: "Core voice", when: "always — sets tone and style" },
    { name: "Code mode", when: "when code keywords or code files detected" },
    { name: "Document reader", when: "when PDF or text/markdown attached" },
  ];
  const root = $("skillList");
  root.innerHTML = "";
  for (let i = 0; i < skills.length; i++) {
    const s = skills[i];
    const row = document.createElement("div");
    row.className = "skill-row";
    row.innerHTML = '<div><div class="name"></div><div class="when"></div></div>';
    row.querySelector(".name").textContent = s.name;
    row.querySelector(".when").textContent = s.when;
    root.appendChild(row);
  }
}

async function buildProvidersStatus() {
  if (!healthCache) { try { healthCache = await apiHealth(); } catch (e) { healthCache = {}; } }
  const root = $("providersStatus");
  root.innerHTML = "";
  const providers = (healthCache && healthCache.providers) || {};
  const items = [
    ["cerebras",   "Cerebras",   "1M tokens/day · primary chat"],
    ["gemini",     "Google AI",  "1500 req/day · chat fallback + vision/PDF"],
    ["openrouter", "OpenRouter", "~50 req/day · Venice + auto-router"],
    ["groq",       "Groq",       "optional · chat fallback"],
    ["tavily",     "Tavily",     "1000/month · web search"],
  ];
  for (let i = 0; i < items.length; i++) {
    const k = items[i][0], label = items[i][1], note = items[i][2];
    const row = document.createElement("div");
    row.className = "row";
    const has = !!providers[k];
    row.innerHTML = '<span class="label"><span class="lbl-text"></span> <span class="note"></span></span><span></span>';
    row.querySelector(".lbl-text").textContent = label;
    row.querySelector(".note").textContent = note;
    const tag = row.querySelector("span:last-child");
    tag.className = has ? "ok" : "miss";
    tag.textContent = has ? "✓ set" : "✗ not set";
    root.appendChild(row);
  }
  $("aboutVersion").textContent = "Running: " + ((healthCache && healthCache.version) || "?");
}

async function openSettings() {
  $("myKey").textContent = syncKey;
  // Force-refresh health each time settings opens. Otherwise adding a provider
  // key in Deno Deploy wouldn't show up here until a full page reload.
  healthCache = null;
  await buildModelGrid();
  buildSkillList();
  await buildProvidersStatus();
  let mems = [];
  try { mems = await apiMemGet(); } catch (e) {}
  const list = $("memList");
  list.innerHTML = "";
  if (mems.length === 0) {
    const e = document.createElement("div");
    e.className = "mem-empty";
    e.textContent = "Nothing learned yet. Have a few conversations and check back.";
    list.appendChild(e);
  } else {
    for (let i = 0; i < mems.length; i++) {
      (function(m){
        const it = document.createElement("div");
        it.className = "mem-item";
        const t = document.createElement("div");
        t.className = "text";
        t.textContent = m;
        const rm = document.createElement("button");
        rm.className = "rm";
        rm.textContent = "×";
        rm.title = "Forget this";
        rm.onclick = async function(){
          const f = mems.filter(function(x){ return x !== m; });
          try { await apiMemSave(f); openSettings(); await extractNameFromMem(); }
          catch (e) { toast("Couldn't forget that", "error"); }
        };
        it.appendChild(t);
        it.appendChild(rm);
        list.appendChild(it);
      })(mems[i]);
    }
  }
  $("settingsModal").classList.add("open");
}

$("copyKey").onclick = function(){ navigator.clipboard.writeText(syncKey); toast("Key copied", "success"); };
async function addManualMemory() {
  const inp = $("memInput");
  const v = inp.value.trim();
  if (!v) return;
  if (v.length < 4) { toast("Too short", "warn"); return; }
  try {
    const existing = await apiMemGet();
    if (existing.indexOf(v) !== -1) { toast("Already there", "warn"); return; }
    await apiMemSave([...existing, v]);
    inp.value = "";
    await extractNameFromMem();
    openSettings();
    toast("Remembered", "success");
  } catch (e) { toast("Couldn't save", "error"); }
}
$("memAddBtn").onclick = addManualMemory;
$("memInput").addEventListener("keydown", function(e){
  if (e.key === "Enter") { e.preventDefault(); addManualMemory(); }
});
$("clearMem").onclick = async function(){
  if (await customConfirm("Wipe memory?", "I'll forget everything I've learned about you.", "Wipe")) {
    try { await apiMemSave([]); userMemoryName = null; openSettings(); toast("Cleared", "success"); }
    catch (e) { toast("Couldn't clear", "error"); }
  }
};
$("exportAll").onclick = async function(){
  try {
    const chats = await apiList();
    const full = [];
    for (let i = 0; i < chats.length; i++) full.push(await apiGet(chats[i].id));
    const blob = new Blob([JSON.stringify({ exported: new Date().toISOString(), chats: full }, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "router-chats-" + Date.now() + ".json";
    a.click();
    toast("Exported", "success");
  } catch (e) { toast("Export failed", "error"); }
};
$("closeSettings").onclick = function(){ $("settingsModal").classList.remove("open"); };
$("settingsBtn").onclick = openSettings;

// ===== Nav =====
$("menuBtn").onclick = openSB;
$("backdrop").onclick = function(){
  closeSB();
  $("welcomeModal").classList.remove("open");
  $("settingsModal").classList.remove("open");
  $("confirmModal").classList.remove("open");
  if (confirmResolver) { confirmResolver(false); confirmResolver = null; }
};
$("newBtn").onclick = newChat;

let searchTimer;
$("searchBox").oninput = function(e){
  clearTimeout(searchTimer);
  const v = e.target.value;
  searchTimer = setTimeout(function(){ renderList(v); }, 150);
};
$("lightbox").onclick = function(){ $("lightbox").classList.remove("open"); };

// Triple-tap title easter egg
let titleTaps = 0, titleTimer = null;
$("titleWrap").onclick = async function(){
  titleTaps++;
  clearTimeout(titleTimer);
  titleTimer = setTimeout(function(){ titleTaps = 0; }, 600);
  if (titleTaps >= 3) {
    titleTaps = 0;
    try {
      const h = await apiHealth();
      const active = Object.entries(h.providers || {}).filter(function(e){ return e[1]; }).map(function(e){ return e[0]; }).join(", ");
      toast("v" + h.version + " · " + (active || "none"), "info");
    } catch (e) {}
  }
};

// ESC closes things
document.addEventListener("keydown", function(e){
  if (e.key === "Escape") {
    if ($("confirmModal").classList.contains("open")) closeConfirm(false);
    else if ($("settingsModal").classList.contains("open")) $("settingsModal").classList.remove("open");
    else if ($("welcomeModal").classList.contains("open") && syncKey) $("welcomeModal").classList.remove("open");
    else if ($("sidebar").classList.contains("open")) closeSB();
    else if ($("lightbox").classList.contains("open")) $("lightbox").classList.remove("open");
  }
});

// ===== Init =====
setupVoice();
savePrefs();
setupScrollWatcher();
startCyclingPlaceholder();
(async function(){
  if (!syncKey) { showWelcome(); return; }
  await extractNameFromMem();
  await renderList();
  renderAll();
})();
</script>
</body>
</html>`;


// ============================================================
// STATIC ASSETS
// ============================================================
const MANIFEST = JSON.stringify({
  name: "Router",
  short_name: "Router",
  start_url: "/",
  display: "standalone",
  background_color: "#07070a",
  theme_color: "#07070a",
  icons: [{ src: "/icon.svg", sizes: "any", type: "image/svg+xml", purpose: "any" }],
});

const ICON_SVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 192 192">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#7c7af8"/>
      <stop offset="1" stop-color="#a855f7"/>
    </linearGradient>
  </defs>
  <rect width="192" height="192" fill="url(#g)" rx="42"/>
  <text x="96" y="132" font-size="115" text-anchor="middle" fill="white"
        font-family="Inter,sans-serif" font-weight="700">R</text>
</svg>`;

// ============================================================
// SERVER ROUTER
// ============================================================
Deno.serve(async (req) => {
  const url = new URL(req.url);
  const p = url.pathname;

  // Static
  if (p === "/" && req.method === "GET")
    return new Response(HTML, { headers: { "Content-Type": "text/html; charset=utf-8" } });
  if (p === "/manifest.json")
    return new Response(MANIFEST, { headers: { "Content-Type": "application/json" } });
  if (p === "/icon.svg")
    return new Response(ICON_SVG, { headers: { "Content-Type": "image/svg+xml" } });

  // Chat
  if (p === "/chat" && req.method === "POST") return handleChat(req);

  // Chats CRUD
  if (p === "/chats" && req.method === "GET")       return handleChatsList(req);
  if (p === "/chats/save" && req.method === "POST") return handleChatSave(req);

  const pinMatch = p.match(/^\/chats\/([^\/]+)\/pin$/);
  if (pinMatch && req.method === "POST") return handleChatPin(req, pinMatch[1]);

  const chatMatch = p.match(/^\/chats\/([^\/]+)$/);
  if (chatMatch) {
    if (req.method === "GET")    return handleChatGet(req, chatMatch[1]);
    if (req.method === "DELETE") return handleChatDelete(req, chatMatch[1]);
  }

  // Memories
  if (p === "/memories" && req.method === "GET")        return handleMemGet(req);
  if (p === "/memories/save" && req.method === "POST")  return handleMemSave(req);

  // Health / diagnostics
  if (p === "/health") {
    try {
      return Response.json({
        ok: true,
        version: "v7.1",
        providers: {
          cerebras:   !!CEREBRAS_KEY,
          gemini:     !!GOOGLE_AI_KEY,
          openrouter: !!OPENROUTER_KEY,
          groq:       !!GROQ_KEY,
          tavily:     !!TAVILY_KEY,
        },
        models: Object.fromEntries(
          Object.entries(MODELS).map(([k, v]) => [
            k,
            { label: v.label, chain: v.chain.map(c => `${c.provider}:${c.id}`) },
          ])
        ),
      });
    } catch (e) {
      return Response.json({ ok: false, error: String(e) }, { status: 500 });
    }
  }

  return new Response("Not found", { status: 404 });
});
