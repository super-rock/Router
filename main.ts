// ============================================================
// PERSONAL AI ROUTER  v6.2
//
// v6.2 — UX polish pass. Enter creates new lines (send is button-only,
// or Ctrl/Cmd+Enter). Custom confirmation modals replace browser confirm().
// Message actions always visible on mobile. Error bubbles styled distinctly
// with inline retry. Smart greetings, paste/drag-drop images, code language
// labels, debounced search, ESC closes modals, triple-tap title shows version.
// All v6.1 backend guarantees preserved.
//
// v6.1 (carried forward) — CRITICAL FIX: persist() was overwriting successful
// responses with "Error: HTTP 500" whenever the chat save failed (which it did
// for any chat with images, because base64 images exceed Deno KV's 64KB
// per-value limit). Now: images are stripped before save, save failures are
// silent (toast only), and the server never returns non-200 from /chats/save
// so the frontend can't throw on it.
//
// Fallback-chain architecture. Each user-facing "model" is now an
// ORDERED CHAIN of provider attempts. On 429/5xx/404 from one provider,
// the next is tried automatically. User sees one continuous response,
// with a small note when a fallback kicks in.
//
// Daily capacity (no Groq):
//   Cerebras  : 1M tokens/day, 30 RPM
//   Gemini    : 1,500 req/day, 15 RPM
//   OpenRouter: 50 req/day (free pool — reduced from 200 in 2026)
//   Tavily    : 1,000 searches/month
//
// Env vars required:
//   OPENROUTER_KEY  — for Venice (uncensored) and openrouter/free auto-router
//   GOOGLE_AI_KEY   — for Gemini (chat fallback + vision)
//   CEREBRAS_KEY    — for primary Chat (GPT-OSS 120B)
//   TAVILY_KEY      — for web search (🔍 toggle)
//   GROQ_KEY        — optional, used if present
//
// Design rules:
//   1. Each "model" in MODELS is a chain of provider attempts.
//   2. Fail-fast on 429 — never retry the same provider, fall through to next.
//   3. Retry on 5xx with backoff (transient server issues).
//   4. Once tokens start streaming, commit to that response (no mid-stream fallback).
//   5. Errors are descriptive and name the chain attempted.
// ============================================================

const OPENROUTER_KEY = Deno.env.get("OPENROUTER_KEY") || "";
const TAVILY_KEY     = Deno.env.get("TAVILY_KEY")     || "";
const GOOGLE_AI_KEY  = Deno.env.get("GOOGLE_AI_KEY")  || "";
const CEREBRAS_KEY   = Deno.env.get("CEREBRAS_KEY")   || "";
const GROQ_KEY       = Deno.env.get("GROQ_KEY")       || "";

const kv = await Deno.openKv();

// ------------------------------------------------------------
// Provider config (OAI-compatible endpoints)
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
// Model registry — each entry is a CHAIN of provider attempts
// ------------------------------------------------------------
type ProviderOption = {
  provider: Provider;
  id: string;
  label: string; // descriptive: "GPT-OSS 120B [Cerebras]"
};

type ModelEntry = {
  label: string;          // user-facing label: "Chat (auto)"
  blurb: string;
  vision: boolean;
  chain: ProviderOption[];
};

const MODELS: Record<string, ModelEntry> = {
  chat: {
    label: "Chat (auto)",
    blurb: "Auto-routes across Cerebras → Gemini → OpenRouter free. Falls over instantly when rate-limited.",
    vision: false,
    chain: [
      { provider: "cerebras",   id: "gpt-oss-120b",                              label: "GPT-OSS 120B [Cerebras]" },
      { provider: "groq",       id: "llama-3.3-70b-versatile",                   label: "Llama 3.3 70B [Groq]" },
      { provider: "gemini",     id: "gemini-2.5-flash",                          label: "Gemini 2.5 Flash" },
      { provider: "openrouter", id: "openrouter/free",                           label: "OpenRouter Auto-Free" },
    ],
  },
  uncensored: {
    label: "Uncensored (Venice)",
    blurb: "Venice via OpenRouter. No content filter. ~50 req/day free pool — use deliberately.",
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
      { provider: "openrouter", id: "openrouter/free",                           label: "OR Auto-Free Reasoner" },
      { provider: "cerebras",   id: "qwen-3-235b-a22b-thinking-2507",            label: "Qwen 3 Thinking [Cerebras]" },
      { provider: "gemini",     id: "gemini-2.5-flash",                          label: "Gemini Fallback" },
    ],
  },
  vision: {
    label: "Vision",
    blurb: "For images. Auto-engaged when you attach a photo.",
    vision: true,
    chain: [
      { provider: "gemini",     id: "gemini-2.5-flash",                                              label: "Gemini 2.5 Vision" },
      { provider: "openrouter", id: "meta-llama/llama-3.2-11b-vision-instruct:free",                 label: "Llama Vision" },
    ],
  },
};

const DEFAULT_LEAD = "chat";
const REASONER     = "reasoner";
const VISION       = "vision";

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

function buildSystemPrompt(memories: string[], hasSearchContext: boolean): string {
  const memBlock = memories.length
    ? `\n\n# What you know about this person (from past conversations)\n${memories.map(m => `- ${m}`).join("\n")}\n\nWeave this in naturally only when relevant. Don't recite. Don't announce that you "remember." Just be a person who already knows them.`
    : "";

  const searchNote = hasSearchContext
    ? `\n\n# Note\nThe user's message includes web search results. Use them to give an accurate, current answer. Cite the sources at the end.`
    : "";

  return `You are the lead of a personal AI assistant built for one specific person. You answer directly.

# Voice
Warm, present, a touch dry. Not overeager, not robotic. Brief by default; longer when the question earns it. Playful when it fits, never performative. Speak like a sharp friend, not a help desk. Disagree when you mean it. Adults talking to adults — no caveats, no moralizing, no "as an AI" framing.

# Style
- Format with markdown when it helps (code blocks with language tags, lists, bold). Plain prose otherwise.
- Never open with "Sure!", "Certainly!", "Of course!", or "I'd be happy to."
- Don't ask permission to answer. Just answer.

Today is ${new Date().toISOString().slice(0, 10)} (${timeGreeting()} UTC).${memBlock}${searchNote}`;
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
// OpenAI-compatible call (Cerebras, Groq, OpenRouter)
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

  // Fail-fast on 429; retry only on 5xx
  const backoffs = [0, 1500, 4000];
  for (let attempt = 0; attempt < backoffs.length; attempt++) {
    if (backoffs[attempt]) await new Promise(r => setTimeout(r, backoffs[attempt]));

    let r: Response;
    try {
      r = await fetch(config.endpoint, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
          ...config.headers,
        },
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
// Gemini direct call (different API format)
// ------------------------------------------------------------
async function callGeminiDirect(
  modelId: string,
  messages: any[],
  opts: { stream?: boolean; signal?: AbortSignal } = {},
): Promise<Response> {
  if (!GOOGLE_AI_KEY) {
    return new Response(
      JSON.stringify({ error: { message: "GOOGLE_AI_KEY not set on server." } }),
      { status: 599 },
    );
  }

  // Convert OpenAI-format messages to Gemini format
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
          if (match) {
            parts.push({ inline_data: { mime_type: match[1], data: match[2] } });
          }
        }
      }
    }

    // Skip messages with no parts (would cause 400)
    if (parts.length === 0) continue;

    contents.push({
      role: m.role === "assistant" ? "model" : "user",
      parts,
    });
  }

  const body: any = {
    contents,
    safetySettings: [
      { category: "HARM_CATEGORY_HARASSMENT", threshold: "BLOCK_ONLY_HIGH" },
      { category: "HARM_CATEGORY_HATE_SPEECH", threshold: "BLOCK_ONLY_HIGH" },
      { category: "HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold: "BLOCK_ONLY_HIGH" },
      { category: "HARM_CATEGORY_DANGEROUS_CONTENT", threshold: "BLOCK_ONLY_HIGH" },
    ],
  };
  if (systemInstruction) {
    body.systemInstruction = { parts: [{ text: systemInstruction }] };
  }

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

// ------------------------------------------------------------
// Provider dispatcher
// ------------------------------------------------------------
async function callProvider(
  option: ProviderOption,
  messages: any[],
  opts: { stream?: boolean; signal?: AbortSignal } = {},
): Promise<Response> {
  if (option.provider === "gemini") {
    return callGeminiDirect(option.id, messages, opts);
  }
  return callOAICompatible(option.provider, option.id, messages, opts);
}

async function callProviderText(option: ProviderOption, messages: any[]): Promise<string> {
  const r = await callProvider(option, messages);
  if (!r.ok) return "";
  const d = await r.json();
  if (option.provider === "gemini") {
    return d.candidates?.[0]?.content?.parts?.[0]?.text || "";
  }
  return d.choices?.[0]?.message?.content || "";
}

// ------------------------------------------------------------
// Convert message format with images to OAI multimodal
// ------------------------------------------------------------
function toOAIMessages(messages: any[]): { msgs: any[]; hasImages: boolean } {
  let hasImages = false;
  const msgs = messages.map(m => {
    if (m.role === "user" && Array.isArray(m.images) && m.images.length) {
      hasImages = true;
      const parts: any[] = [];
      if (m.content) parts.push({ type: "text", text: m.content });
      for (const img of m.images) {
        if (img?.data && img?.mimeType) {
          parts.push({
            type: "image_url",
            image_url: { url: `data:${img.mimeType};base64,${img.data}` },
          });
        }
      }
      return { role: "user", content: parts };
    }
    return { role: m.role, content: m.content || "" };
  });
  return { msgs, hasImages };
}

const sseChunk = (event: string, data: any): Uint8Array =>
  new TextEncoder().encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);

// ------------------------------------------------------------
// Friendly per-status, per-provider error
// ------------------------------------------------------------
function friendlyStatus(status: number, body: string, option: ProviderOption): string {
  if (status === 401 || status === 403) {
    return `${option.label}: auth issue. Check ${option.provider.toUpperCase()}_KEY in Deno Deploy.`;
  }
  if (status === 404) {
    return `${option.label}: model not found (likely deprecated by provider).`;
  }
  if (status === 429) {
    return `${option.label}: rate-limited.`;
  }
  if (status >= 500 && status <= 599) {
    return `${option.label}: server error HTTP ${status}.`;
  }
  try {
    const parsed = JSON.parse(body);
    const m = parsed?.error?.message || parsed?.message;
    if (m) return `${option.label}: ${String(m).slice(0, 200)}`;
  } catch {}
  return `${option.label}: HTTP ${status}.`;
}

// ------------------------------------------------------------
// Stream a single provider's response. Returns true on success
// (at least one token received), false on failure.
// ------------------------------------------------------------
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
    if (signal.aborted) {
      try { reader.cancel(); } catch {}
      return { ok: gotAnyToken, reason: "aborted" };
    }

    let chunk;
    try {
      chunk = await reader.read();
    } catch (e) {
      return { ok: gotAnyToken, reason: `stream error: ${e}` };
    }
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

      // Inline error from provider
      if (j.error) {
        return { ok: gotAnyToken, reason: `provider error: ${j.error.message || JSON.stringify(j.error).slice(0,100)}` };
      }

      // Gemini may return prompt feedback indicating block
      if (isGemini && j.promptFeedback?.blockReason) {
        return { ok: gotAnyToken, reason: `blocked by Gemini safety: ${j.promptFeedback.blockReason}` };
      }

      const token = isGemini
        ? j.candidates?.[0]?.content?.parts?.[0]?.text || ""
        : j.choices?.[0]?.delta?.content || "";

      if (token) {
        gotAnyToken = true;
        send("token", { text: token });
      }
    }
  }

  return { ok: gotAnyToken, reason: gotAnyToken ? undefined : "empty response" };
}

// ------------------------------------------------------------
// Run a turn — iterate the model's fallback chain
// ------------------------------------------------------------
async function runTurn(
  controller: ReadableStreamDefaultController,
  messages: any[],
  modelKey: string,
  useSearch: boolean,
  useReasoning: boolean,
  systemPromptBase: (hasSearch: boolean) => string,
  signal: AbortSignal,
) {
  const send = (event: string, data: any) => {
    try { controller.enqueue(sseChunk(event, data)); } catch {}
  };

  // Decide which mode to use
  const lastUser = messages[messages.length - 1];
  const hasImages = Array.isArray(lastUser?.images) && lastUser.images.length > 0;

  let actualKey = modelKey;
  let routingNote: string | null = null;

  if (hasImages && !MODELS[modelKey]?.vision) {
    actualKey = VISION;
    routingNote = "Image detected → vision chain";
  } else if (useReasoning) {
    actualKey = REASONER;
    routingNote = "Deep reasoning → reasoner chain";
  }

  const entry = MODELS[actualKey];
  if (!entry) {
    send("error", { message: `Unknown mode: ${actualKey}` });
    return;
  }

  send("status", { leader: entry.label });
  if (routingNote) send("note", { msg: routingNote });

  // Run Tavily search if requested
  let augmentedMessages = [...messages];
  let didSearch = false;

  if (useSearch && !hasImages && lastUser?.content) {
    send("tool_start", { tool: "web_search" });
    const searchResult = await tavilySearch(lastUser.content);
    send("tool_done", { tool: "web_search", preview: searchResult.slice(0, 200) });
    didSearch = true;

    const lastIdx = augmentedMessages.length - 1;
    const orig = augmentedMessages[lastIdx];
    augmentedMessages[lastIdx] = {
      ...orig,
      content: `Web search results (use these to answer):\n\n${searchResult}\n\n---\n\nMy question: ${orig.content}`,
    };
  }

  // Build messages in OAI format (Gemini conversion happens inside callGeminiDirect)
  const { msgs: oaiMsgs } = toOAIMessages(augmentedMessages);
  const sysContent = systemPromptBase(didSearch);
  const finalMsgs = [{ role: "system", content: sysContent }, ...oaiMsgs];

  // Iterate the fallback chain
  const triedErrors: string[] = [];
  let succeeded = false;

  for (let i = 0; i < entry.chain.length; i++) {
    if (signal.aborted) { send("aborted", {}); return; }

    const option = entry.chain[i];

    // Skip if provider key missing
    if (!providerKey(option.provider)) {
      triedErrors.push(`${option.label}: ${option.provider.toUpperCase()}_KEY not set`);
      continue;
    }

    if (i > 0) {
      send("note", { msg: `Trying ${option.label}…` });
    }

    // Initial call (not yet streaming)
    const r = await callProvider(option, finalMsgs, { stream: true, signal });

    if (!r.ok) {
      let errText = "";
      try { errText = await r.text(); } catch {}
      const reason = friendlyStatus(r.status, errText, option);
      triedErrors.push(reason);

      // Stop chain entirely on auth issues — won't be fixed by trying other providers
      if (r.status === 401 || r.status === 403) {
        send("error", { message: `Stopped: ${reason}. Fix the key in Deno Deploy env vars.` });
        return;
      }

      // For 429/404/5xx — continue to next provider in chain
      continue;
    }

    // Stream this provider's response
    const result = await streamProviderResponse(r, option, send, signal);

    if (result.ok) {
      // Got at least one token — commit to this response
      send("done", {
        responder: option.label,
        tool_used: didSearch ? "web_search" : (actualKey === REASONER ? "ask_reasoner" : null),
        fallback_used: i > 0,
      });
      succeeded = true;
      return;
    }

    // Stream failed without any tokens — record and try next
    triedErrors.push(`${option.label}: ${result.reason || "no tokens"}`);
  }

  if (!succeeded) {
    const summary = triedErrors.map(e => `• ${e}`).join("\n");
    send("error", {
      message: `All providers in the **${entry.label}** chain failed:\n\n${summary}\n\nTry another mode in Settings, or wait a minute and retry.`,
    });
  }
}

// ------------------------------------------------------------
// Memory extraction — uses chat chain (will fall over too if Cerebras down)
// ------------------------------------------------------------
const MEM_EVERY = 3;

async function maybeExtractMemories(uh: string, chat: any) {
  const msgs = chat.messages || [];
  const userCount = msgs.filter((m: any) => m.role === "user").length;
  if (userCount === 0 || userCount % MEM_EVERY !== 0) return;

  const window = msgs.slice(-6)
    .map((m: any) => `${m.role.toUpperCase()}: ${(m.content || "").slice(0, 800)}`)
    .join("\n");
  const existing = await getMem(uh);

  const prompt = `Extract durable facts about the user from this exchange. Only facts useful in future conversations: name, ongoing projects, preferences, recurring interests, location, work, relationships, skills, opinions they hold. Skip anything transient. Skip anything already in the existing list.

Existing memories:
${existing.length ? existing.map(m => `- ${m}`).join("\n") : "(none yet)"}

Recent exchange:
${window}

Output ONLY a JSON array of new short memory strings, e.g. ["User's name is Sam", "User is learning Spanish"]. Empty array [] if nothing memorable. No other text, no markdown fences.`;

  // Try each provider in the chat chain for memory extraction
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
      return; // success
    } catch {
      continue;
    }
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
  const sysPromptBase = (hasSearch: boolean) => buildSystemPrompt(mems, hasSearch);

  const ac = new AbortController();
  req.signal.addEventListener("abort", () => ac.abort());

  const stream = new ReadableStream({
    async start(controller) {
      try {
        await runTurn(
          controller,
          messages,
          modelKey,
          !!use_search,
          !!use_reasoning,
          sysPromptBase,
          ac.signal,
        );
      } catch (e) {
        try { controller.enqueue(sseChunk("error", { message: String(e) })); } catch {}
      } finally {
        try { controller.close(); } catch {}
      }
    },
    cancel() { ac.abort(); },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "X-Accel-Buffering": "no",
    },
  });
}

async function handleChatsList(req: Request): Promise<Response> {
  const sk = new URL(req.url).searchParams.get("sk") || "";
  if (!sk) return new Response("missing sk", { status: 400 });
  const uh = await hashKey(sk);
  const chats = await listChats(uh);
  return Response.json(chats.map((c: any) => ({
    id: c.id,
    title: c.title,
    updated: c.updated,
    pinned: !!c.pinned,
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

  // Safety net: strip any image data the client may have sent.
  // Deno KV value limit is 64KB; even one base64 image exceeds this.
  if (Array.isArray(chat.messages)) {
    chat.messages = chat.messages.map((m: any) => {
      if (Array.isArray(m.images) && m.images.length > 0) {
        const note = m.images.length === 1 ? "[image attached]" : `[${m.images.length} images attached]`;
        const newContent = m.content ? `${m.content}\n\n${note}` : note;
        return { ...m, images: [], content: newContent };
      }
      return m;
    });
  }

  // Size check before write. If still too large, truncate oldest messages.
  const SIZE_LIMIT = 60000; // ~60KB, leaves headroom under Deno KV's 64KB cap
  let serialized = JSON.stringify(chat);
  if (serialized.length > SIZE_LIMIT && Array.isArray(chat.messages)) {
    // Keep recent messages until we fit
    while (chat.messages.length > 4 && serialized.length > SIZE_LIMIT) {
      chat.messages.shift();
      serialized = JSON.stringify(chat);
    }
    // If still too large, truncate individual message content
    if (serialized.length > SIZE_LIMIT) {
      chat.messages = chat.messages.map((m: any) => ({
        ...m,
        content: typeof m.content === "string" && m.content.length > 4000
          ? m.content.slice(0, 4000) + "…[truncated]"
          : m.content,
      }));
      serialized = JSON.stringify(chat);
    }
    // Give up if still too large
    if (serialized.length > SIZE_LIMIT) {
      return Response.json({ ok: false, error: "Chat too large even after truncation" });
    }
  }

  try {
    await putChat(uh, chat);
    maybeExtractMemories(uh, chat);
    return Response.json({ ok: true });
  } catch (e) {
    // Never return non-200 — frontend treats that as error and could clobber the streamed response.
    return Response.json({ ok: false, error: String(e) });
  }
}

async function handleChatDelete(req: Request, id: string): Promise<Response> {
  const sk = new URL(req.url).searchParams.get("sk") || "";
  if (!sk) return new Response("missing sk", { status: 400 });
  const uh = await hashKey(sk);
  await delChat(uh, id);
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
// FRONTEND HTML
// ============================================================
const HTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover,maximum-scale=1">
<meta name="theme-color" content="#09090b">
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
  --bg:#09090b; --panel:#121214; --surface:#17171a; --surface-2:#1d1d22; --surface-3:#26262b;
  --border:#26262b; --border-strong:#3a3a42; --border-bright:#525258;
  --text:#ededf0; --text-dim:#a1a1aa; --text-muted:#71717a; --text-faint:#52525b;
  --accent:#7c7af8; --accent-2:#a855f7; --accent-soft:rgba(124,122,248,.16);
  --search:#fbbf24; --search-soft:rgba(251,191,36,.14);
  --reason:#c084fc; --reason-soft:rgba(192,132,252,.16);
  --danger:#ef4444; --danger-soft:rgba(239,68,68,.13);
  --success:#10b981; --success-soft:rgba(16,185,129,.13);
  --warn:#f59e0b;
  --grad:linear-gradient(135deg,#7c7af8 0%,#a855f7 60%,#c084fc 100%);
  --grad-soft:linear-gradient(135deg,rgba(124,122,248,.2) 0%,rgba(168,85,247,.15) 100%);
  --shadow:0 10px 40px rgba(0,0,0,.5);
  --shadow-soft:0 4px 16px rgba(0,0,0,.3);
  --radius:14px; --radius-lg:18px; --radius-sm:8px;
}
*,*::before,*::after{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
html,body{margin:0;padding:0;height:100%;background:var(--bg);color:var(--text);font-family:"Inter",-apple-system,BlinkMacSystemFont,system-ui,sans-serif;-webkit-font-smoothing:antialiased;overflow:hidden;font-size:15px;line-height:1.5;letter-spacing:-.005em}
body{display:flex;flex-direction:column}
button,input,textarea,select{font-family:inherit;letter-spacing:inherit}
button{user-select:none;-webkit-user-select:none}
::selection{background:rgba(124,122,248,.35);color:#fff}

/* ============ HEADER ============ */
.header{display:flex;align-items:center;padding:10px 12px;border-bottom:1px solid var(--border);flex-shrink:0;gap:2px;padding-top:max(10px,env(safe-area-inset-top));background:var(--bg);position:relative;z-index:5;backdrop-filter:blur(12px)}
.icon-btn{background:transparent;border:none;color:var(--text-dim);cursor:pointer;padding:9px;border-radius:11px;line-height:1;display:flex;align-items:center;justify-content:center;width:40px;height:40px;transition:background .12s ease,color .12s ease,transform .08s ease}
.icon-btn:hover{color:var(--text);background:var(--surface)}
.icon-btn:active{transform:scale(.92);background:var(--surface-2)}
.icon-btn svg{width:21px;height:21px;stroke-width:2}
.title-wrap{flex:1;text-align:center;padding:0 6px;overflow:hidden;cursor:pointer}
.title{font-size:14.5px;color:var(--text);font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;letter-spacing:-.015em}
.model-badge{font-size:10.5px;color:var(--text-muted);margin-top:1px;display:flex;align-items:center;justify-content:center;gap:5px;font-weight:500}
.model-badge .dot{width:6px;height:6px;border-radius:50%;background:var(--success);box-shadow:0 0 0 3px rgba(16,185,129,.18);animation:dot-pulse 2.4s ease-in-out infinite}
@keyframes dot-pulse{0%,100%{box-shadow:0 0 0 3px rgba(16,185,129,.18)}50%{box-shadow:0 0 0 5px rgba(16,185,129,.05)}}

/* ============ SIDEBAR ============ */
.sidebar{position:fixed;top:0;left:0;bottom:0;width:312px;max-width:86vw;background:var(--panel);transform:translateX(-100%);transition:transform .28s cubic-bezier(.32,.72,.32,1);z-index:100;display:flex;flex-direction:column;border-right:1px solid var(--border);padding-top:env(safe-area-inset-top);padding-bottom:env(safe-area-inset-bottom);box-shadow:var(--shadow)}
.sidebar.open{transform:translateX(0)}
.sb-head{padding:12px;border-bottom:1px solid var(--border);display:flex;flex-direction:column;gap:10px}
.new-chat{width:100%;background:var(--grad);color:#fff;border:none;padding:12px 14px;border-radius:12px;font-size:14.5px;cursor:pointer;font-weight:600;display:flex;align-items:center;justify-content:center;gap:8px;letter-spacing:-.01em;box-shadow:0 4px 14px rgba(124,122,248,.32);transition:transform .12s,filter .12s,box-shadow .12s}
.new-chat:hover{filter:brightness(1.08)}
.new-chat:active{transform:scale(.97);filter:brightness(.92)}
.search-wrap{position:relative}
.search-wrap svg{position:absolute;left:10px;top:50%;transform:translateY(-50%);width:14px;height:14px;color:var(--text-muted);pointer-events:none}
.search-box{width:100%;background:var(--surface);border:1px solid var(--border);color:var(--text);padding:9px 12px 9px 32px;border-radius:10px;font-size:13.5px;outline:none;transition:border-color .12s,background .12s}
.search-box:focus{border-color:var(--accent);background:var(--surface-2)}
.chats-list{flex:1;overflow-y:auto;padding:6px 8px;-webkit-overflow-scrolling:touch}
.chat-item{padding:10px 12px;border-radius:11px;cursor:pointer;font-size:13.5px;color:var(--text-dim);margin-bottom:3px;display:flex;align-items:center;gap:8px;transition:background .12s;position:relative}
.chat-item:hover,.chat-item.active{background:var(--surface);color:var(--text)}
.chat-item.active{background:var(--accent-soft);color:var(--text)}
.chat-item .row{flex:1;overflow:hidden;display:flex;flex-direction:column;gap:2px;min-width:0}
.chat-title{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:500;display:flex;align-items:center;gap:6px;font-size:13.5px}
.chat-prev{font-size:11.5px;color:var(--text-muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:400}
.pin-icon{font-size:10px;opacity:.7}
.chat-actions{display:flex;gap:2px;flex-shrink:0}
.chat-actions button{background:transparent;border:none;color:var(--text-faint);cursor:pointer;font-size:13px;padding:5px 7px;border-radius:6px;transition:color .12s,background .12s;display:flex;align-items:center;justify-content:center;width:28px;height:28px}
.chat-actions button:hover{color:var(--text);background:var(--surface-2)}
.chat-actions button:active{transform:scale(.9)}
.chat-actions .del:hover{color:var(--danger);background:var(--danger-soft)}
.chat-actions svg{width:14px;height:14px}
.list-empty{text-align:center;color:var(--text-muted);font-size:13px;padding:36px 20px;font-style:italic;line-height:1.6}
.backdrop{position:fixed;inset:0;background:rgba(0,0,0,.65);backdrop-filter:blur(3px);-webkit-backdrop-filter:blur(3px);z-index:99;opacity:0;pointer-events:none;transition:opacity .22s}
.backdrop.open{opacity:1;pointer-events:auto}

/* ============ MESSAGES ============ */
.messages{flex:1;overflow-y:auto;padding:18px 14px 4px;scroll-behavior:smooth;-webkit-overflow-scrolling:touch}
.messages-inner{max-width:780px;margin:0 auto;display:flex;flex-direction:column;gap:18px}
.msg{display:flex;flex-direction:column;gap:5px;max-width:100%;animation:fadeUp .26s cubic-bezier(.32,.72,.32,1)}
@keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.msg.user{align-items:flex-end}
.msg.assistant{align-items:flex-start}
.bubble{padding:11px 15px;border-radius:18px;line-height:1.55;word-wrap:break-word;overflow-wrap:break-word;font-size:15px;max-width:92%;position:relative}
.msg.user .bubble{background:var(--grad);color:#fff;border-bottom-right-radius:6px;box-shadow:0 3px 12px rgba(124,122,248,.24)}
.msg.assistant .bubble{background:var(--surface);color:var(--text);border-bottom-left-radius:6px;border:1px solid var(--border)}
.msg.assistant .bubble.error-bubble{background:var(--danger-soft);border-color:rgba(239,68,68,.4);color:#fecaca}
.bubble.empty-streaming{min-height:42px;min-width:60px}
.msg-img{max-width:240px;max-height:240px;border-radius:12px;margin:4px 0;display:block;border:1px solid var(--border);object-fit:cover;cursor:zoom-in;transition:transform .12s}
.msg-img:active{transform:scale(.98)}
.bubble p{margin:0 0 .55em}
.bubble p:last-child{margin:0}
.bubble pre{background:#0c0c0e;border:1px solid var(--border);border-radius:11px;padding:14px 14px 12px;overflow-x:auto;margin:10px 0;font-family:"JetBrains Mono",ui-monospace,monospace;font-size:13px;line-height:1.55;position:relative}
.bubble code{font-family:"JetBrains Mono",ui-monospace,monospace;font-size:.88em;background:rgba(255,255,255,.07);padding:2px 6px;border-radius:5px;font-weight:500}
.bubble pre code{background:transparent;padding:0;border-radius:0;font-size:13px;font-weight:400}
.msg.user .bubble code{background:rgba(255,255,255,.2)}
.msg.user .bubble pre{background:rgba(0,0,0,.3);border-color:rgba(255,255,255,.18)}
.bubble ul,.bubble ol{margin:4px 0 6px 20px;padding:0}
.bubble li{margin:3px 0}
.bubble h1,.bubble h2,.bubble h3,.bubble h4{margin:12px 0 6px;font-weight:600;letter-spacing:-.015em}
.bubble h1{font-size:1.25em}.bubble h2{font-size:1.15em}.bubble h3{font-size:1.05em}.bubble h4{font-size:1em}
.bubble h1:first-child,.bubble h2:first-child,.bubble h3:first-child{margin-top:0}
.bubble a{color:#a5b4fc;text-decoration:underline;text-underline-offset:2px;text-decoration-color:rgba(165,180,252,.4)}
.bubble a:hover{text-decoration-color:#a5b4fc}
.bubble blockquote{border-left:3px solid var(--border-strong);padding:2px 0 2px 12px;color:var(--text-dim);margin:8px 0;font-style:italic}
.bubble em{font-style:italic;color:var(--text-dim)}
.bubble strong{font-weight:600;color:#fff}
.bubble hr{border:none;border-top:1px solid var(--border-strong);margin:14px 0}
.bubble table{border-collapse:collapse;margin:8px 0;font-size:.92em}
.bubble th,.bubble td{border:1px solid var(--border-strong);padding:6px 10px}
.bubble th{background:var(--surface-2);font-weight:600}
.copy-btn{position:absolute;top:7px;right:7px;background:rgba(255,255,255,.08);border:1px solid var(--border);color:var(--text-dim);padding:4px 9px;border-radius:6px;font-size:11px;cursor:pointer;opacity:.55;transition:opacity .12s,color .12s,background .12s;font-family:"Inter",sans-serif;font-weight:500;letter-spacing:.01em}
.copy-btn:hover{opacity:1;color:var(--text);background:rgba(255,255,255,.14)}
.bubble pre:hover .copy-btn,.bubble pre:active .copy-btn{opacity:1}
.code-lang{position:absolute;top:7px;left:14px;font-size:10.5px;color:var(--text-muted);font-weight:500;text-transform:uppercase;letter-spacing:.06em;font-family:"Inter",sans-serif;pointer-events:none}
.bubble pre:has(.code-lang){padding-top:30px}
.msg-actions{display:flex;gap:5px;margin-top:4px;padding:0 6px;opacity:.55;transition:opacity .15s}
.msg-actions:hover{opacity:1}
.msg-actions button{background:var(--surface);border:1px solid var(--border);color:var(--text-muted);cursor:pointer;font-size:11.5px;padding:5px 10px;border-radius:8px;display:flex;align-items:center;gap:5px;transition:all .12s;font-weight:500}
.msg-actions button:hover{color:var(--text);background:var(--surface-2);border-color:var(--border-strong)}
.msg-actions button:active{transform:scale(.94)}
.msg-actions button svg{width:11px;height:11px;stroke-width:2.2}
.msg-actions .retry{color:#fca5a5;border-color:rgba(239,68,68,.3)}
.msg-actions .retry:hover{color:#fff;background:rgba(239,68,68,.18)}
.caption{font-size:11px;color:var(--text-muted);padding:0 8px;display:flex;gap:6px;align-items:center;flex-wrap:wrap;font-weight:500}
.badge{font-size:9.5px;padding:2.5px 7px;border-radius:5px;font-weight:600;letter-spacing:.02em;text-transform:uppercase}
.b-search{background:var(--search-soft);color:var(--search)}
.b-reason{background:var(--reason-soft);color:var(--reason)}
.b-fallback{background:rgba(192,132,252,.12);color:#c4b5fd}

/* ============ THINKING / TOOL STATES ============ */
.thinking{display:inline-flex;gap:4px;padding:7px 2px;align-items:center}
.thinking span{width:6px;height:6px;border-radius:50%;background:var(--text-muted);animation:pulse 1.3s ease-in-out infinite}
.thinking span:nth-child(2){animation-delay:.15s}
.thinking span:nth-child(3){animation-delay:.3s}
@keyframes pulse{0%,80%,100%{opacity:.3;transform:scale(.8)}40%{opacity:1;transform:scale(1)}}
.tool-running{display:inline-flex;align-items:center;gap:8px;font-size:12px;color:var(--text-dim);padding:7px 13px;background:var(--surface);border:1px solid var(--border);border-radius:11px;margin-bottom:4px;width:fit-content;animation:fadeUp .22s;font-weight:500}
.tool-running.search{color:var(--search);border-color:rgba(251,191,36,.3)}
.tool-running.reason{color:var(--reason);border-color:rgba(192,132,252,.3)}
.tool-running.note{color:var(--success);border-color:rgba(16,185,129,.3);background:rgba(16,185,129,.04)}

/* ============ EMPTY STATE ============ */
.empty{text-align:center;padding:36px 22px 60px;max-width:560px;margin:24px auto 0;animation:fadeUp .4s}
.empty .greet{font-size:30px;font-weight:800;letter-spacing:-.025em;background:var(--grad);-webkit-background-clip:text;background-clip:text;color:transparent;margin-bottom:8px;line-height:1.15}
.empty .sub{color:var(--text-dim);font-size:14px;margin-bottom:28px;line-height:1.55}
.suggestions{display:flex;flex-direction:column;gap:9px;max-width:420px;margin:0 auto}
.sugg{background:var(--surface);border:1px solid var(--border);color:var(--text-dim);padding:12px 16px;border-radius:13px;cursor:pointer;font-size:13.5px;text-align:left;transition:all .15s;font-weight:500;display:flex;align-items:center;gap:10px;line-height:1.4}
.sugg:hover{background:var(--surface-2);color:var(--text);border-color:var(--border-strong);transform:translateY(-1px)}
.sugg:active{transform:translateY(0) scale(.99)}
.sugg .sg-icon{font-size:15px;flex-shrink:0;opacity:.7}

/* ============ INPUT BAR ============ */
.input-bar{border-top:1px solid var(--border);background:var(--bg);flex-shrink:0;padding-bottom:max(8px,env(safe-area-inset-bottom));position:relative}
.toggles{display:flex;gap:7px;padding:10px 12px 0;max-width:780px;margin:0 auto;flex-wrap:wrap}
.toggle{background:transparent;border:1px solid var(--border);color:var(--text-muted);padding:6px 12px;border-radius:16px;font-size:12.5px;cursor:pointer;display:flex;align-items:center;gap:6px;transition:all .15s;font-weight:600;letter-spacing:-.005em}
.toggle:hover{color:var(--text);border-color:var(--border-strong);background:var(--surface)}
.toggle:active{transform:scale(.96)}
.toggle.active.search{background:var(--search-soft);color:var(--search);border-color:var(--search)}
.toggle.active.reason{background:var(--reason-soft);color:var(--reason);border-color:var(--reason)}
.toggle .lbl{display:none}
@media (min-width:380px){.toggle .lbl{display:inline}}
.input-row{padding:9px 12px;max-width:780px;margin:0 auto;display:flex;gap:8px;align-items:flex-end}
.attach,.voice{background:transparent;border:1px solid var(--border);color:var(--text-dim);border-radius:50%;width:42px;height:42px;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:all .15s;position:relative}
.attach:hover,.voice:hover{background:var(--surface);color:var(--text);border-color:var(--border-strong)}
.attach:active,.voice:active{transform:scale(.93)}
.attach svg,.voice svg{width:19px;height:19px;stroke-width:2}
.attach .img-count{position:absolute;top:-3px;right:-3px;background:var(--accent);color:#fff;font-size:10px;border-radius:50%;width:18px;height:18px;display:flex;align-items:center;justify-content:center;font-weight:700;border:2px solid var(--bg)}
.voice.recording{background:var(--danger);color:#fff;border-color:var(--danger);animation:rec-pulse 1.4s ease-in-out infinite}
@keyframes rec-pulse{0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,.4)}50%{box-shadow:0 0 0 8px rgba(239,68,68,0)}}
.input-wrap{flex:1;background:var(--surface);border:1.5px solid var(--border);border-radius:22px;transition:border-color .15s,background .15s;display:flex;flex-direction:column;overflow:hidden;min-width:0}
.input-wrap:focus-within{border-color:var(--accent);background:var(--surface-2)}
.image-previews{display:flex;gap:7px;padding:9px 11px 0;flex-wrap:wrap}
.image-previews:empty{display:none}
.preview-item{position:relative;width:58px;height:58px;border-radius:9px;overflow:hidden;border:1px solid var(--border);animation:fadeUp .18s}
.preview-item img{width:100%;height:100%;object-fit:cover}
.preview-item .rm{position:absolute;top:3px;right:3px;background:rgba(0,0,0,.85);color:#fff;border:none;border-radius:50%;width:19px;height:19px;font-size:12px;cursor:pointer;display:flex;align-items:center;justify-content:center;padding:0;line-height:1;font-weight:600}
.preview-item .rm:active{transform:scale(.85)}
.input{background:transparent;color:var(--text);border:none;padding:11px 16px;font-size:15px;resize:none;max-height:180px;font-family:inherit;outline:none;line-height:1.5;width:100%}
.input::placeholder{color:var(--text-muted)}
.send,.stop{border:none;border-radius:50%;width:42px;height:42px;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:all .15s}
.send{background:var(--grad);color:#fff;box-shadow:0 3px 12px rgba(124,122,248,.32)}
.send:disabled{background:#26262b;color:#52525b;box-shadow:none;cursor:not-allowed}
.send:active:not(:disabled){transform:scale(.93);filter:brightness(.9)}
.send:hover:not(:disabled){filter:brightness(1.08)}
.stop{background:var(--danger);color:#fff;box-shadow:0 3px 12px rgba(239,68,68,.32);animation:fadeUp .18s}
.stop:active{transform:scale(.93)}
.send svg,.stop svg{width:19px;height:19px;stroke-width:2.5}
.shortcut-hint{font-size:10.5px;color:var(--text-faint);text-align:center;padding:2px 0 4px;font-weight:500;letter-spacing:.02em}

/* ============ MODALS ============ */
.modal-back{position:fixed;inset:0;background:rgba(0,0,0,.78);backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);z-index:200;display:none;align-items:center;justify-content:center;padding:20px;animation:fadeIn .22s}
.modal-back.open{display:flex}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
.modal{background:var(--panel);border:1px solid var(--border-strong);border-radius:20px;max-width:480px;width:100%;padding:26px;box-shadow:var(--shadow);max-height:88vh;overflow-y:auto;animation:modalIn .26s cubic-bezier(.32,.72,.32,1)}
@keyframes modalIn{from{opacity:0;transform:translateY(14px) scale(.97)}to{opacity:1;transform:translateY(0) scale(1)}}
.modal h2{margin:0 0 4px;font-size:22px;font-weight:800;letter-spacing:-.025em;background:var(--grad);-webkit-background-clip:text;background-clip:text;color:transparent}
.modal .sub{color:var(--text-dim);font-size:13.5px;margin-bottom:18px;line-height:1.55}
.modal-section{margin-top:18px}
.modal-section:first-of-type{margin-top:6px}
.modal label,.modal .section-label{display:block;font-size:11px;font-weight:700;color:var(--text-muted);margin:14px 0 8px;text-transform:uppercase;letter-spacing:.07em}
.modal input,.modal textarea,.modal select{width:100%;background:var(--surface);color:var(--text);border:1px solid var(--border);border-radius:11px;padding:11px 14px;font-size:14px;outline:none;transition:border-color .12s,background .12s}
.modal input:focus,.modal textarea:focus,.modal select:focus{border-color:var(--accent);background:var(--surface-2)}
.modal .actions{display:flex;gap:8px;margin-top:18px}
.btn{padding:11px 16px;border-radius:11px;font-size:14px;font-weight:600;cursor:pointer;border:none;transition:all .14s;flex:1;letter-spacing:-.005em;display:flex;align-items:center;justify-content:center;gap:6px}
.btn-primary{background:var(--grad);color:#fff;box-shadow:0 3px 10px rgba(124,122,248,.32)}
.btn-primary:hover{filter:brightness(1.08)}
.btn-primary:active{transform:scale(.97);filter:brightness(.92)}
.btn-ghost{background:transparent;color:var(--text-dim);border:1px solid var(--border)}
.btn-ghost:hover{background:var(--surface);color:var(--text);border-color:var(--border-strong)}
.btn-ghost:active{transform:scale(.97)}
.btn-danger{background:var(--danger);color:#fff;box-shadow:0 3px 10px rgba(239,68,68,.32)}
.btn-danger:hover{filter:brightness(1.1)}
.btn-danger:active{transform:scale(.97);filter:brightness(.9)}
.divider{height:1px;background:var(--border);margin:20px -26px}
.key-display{background:var(--surface);border:1px solid var(--border);border-radius:11px;padding:14px;font-family:"JetBrains Mono",monospace;font-size:15px;text-align:center;letter-spacing:.04em;margin:8px 0;color:#c4b5fd;word-break:break-all;font-weight:500}
.tabs{display:flex;gap:4px;background:var(--surface);padding:4px;border-radius:11px;margin-bottom:18px}
.tab{flex:1;background:transparent;border:none;color:var(--text-dim);padding:9px;border-radius:8px;cursor:pointer;font-weight:600;font-size:13px;transition:all .12s;letter-spacing:-.005em}
.tab.active{background:var(--panel);color:var(--text);box-shadow:0 1px 4px rgba(0,0,0,.2)}
.mem-list{display:flex;flex-direction:column;gap:7px;max-height:280px;overflow-y:auto;padding-right:4px}
.mem-item{display:flex;align-items:center;gap:8px;background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:9px 13px;font-size:13px;animation:fadeUp .18s}
.mem-item .text{flex:1;color:var(--text-dim);line-height:1.45}
.mem-item .rm{background:transparent;border:none;color:var(--text-muted);cursor:pointer;font-size:16px;padding:3px 7px;border-radius:6px;transition:color .12s,background .12s;line-height:1;font-weight:500}
.mem-item .rm:hover{color:var(--danger);background:var(--danger-soft)}
.mem-item .rm:active{transform:scale(.85)}
.mem-empty{text-align:center;color:var(--text-muted);font-size:13px;padding:24px;font-style:italic;line-height:1.55}
.model-grid{display:grid;gap:9px}
.model-card{background:var(--surface);border:1.5px solid var(--border);border-radius:12px;padding:12px 14px;cursor:pointer;text-align:left;transition:all .15s;display:flex;align-items:flex-start;gap:10px}
.model-card:hover{background:var(--surface-2);border-color:var(--border-strong)}
.model-card:active{transform:scale(.99)}
.model-card.selected{border-color:var(--accent);background:var(--accent-soft)}
.model-card .info{flex:1;min-width:0}
.model-card .name{font-weight:700;color:var(--text);font-size:14px;margin-bottom:3px;letter-spacing:-.01em}
.model-card .blurb{font-size:12px;color:var(--text-dim);line-height:1.45}
.model-card .chain-list{font-size:11px;color:var(--text-muted);margin-top:5px;font-family:"JetBrains Mono",monospace;letter-spacing:-.02em}
.model-card .check{color:var(--accent);font-size:20px;line-height:1;opacity:0;transition:opacity .12s;font-weight:600}
.model-card.selected .check{opacity:1}
.providers-status{background:var(--surface);border:1px solid var(--border);border-radius:11px;padding:13px;margin-top:8px;font-size:12.5px;display:flex;flex-direction:column;gap:7px}
.providers-status .row{display:flex;justify-content:space-between;color:var(--text-dim);align-items:center;gap:8px}
.providers-status .label{flex:1;min-width:0}
.providers-status .note{color:var(--text-muted);font-weight:400;font-size:11.5px}
.providers-status .ok{color:var(--success);font-weight:600;flex-shrink:0}
.providers-status .miss{color:var(--danger);font-weight:600;flex-shrink:0}
.about-block{background:var(--surface);border:1px solid var(--border);border-radius:11px;padding:13px;font-size:12px;color:var(--text-muted);line-height:1.6}
.about-block strong{color:var(--text);font-weight:600}

/* ============ TOAST ============ */
.toast{position:fixed;top:max(16px,env(safe-area-inset-top));left:50%;transform:translateX(-50%) translateY(-30px);background:var(--panel);border:1px solid var(--border-strong);color:var(--text);padding:11px 18px;border-radius:11px;font-size:13px;z-index:300;opacity:0;pointer-events:none;transition:opacity .22s,transform .22s;box-shadow:var(--shadow);max-width:90vw;font-weight:500;display:flex;align-items:center;gap:8px}
.toast.show{opacity:1;transform:translateX(-50%) translateY(0)}
.toast.success{border-color:rgba(16,185,129,.45)}
.toast.success .toast-icon{color:var(--success)}
.toast.error{border-color:rgba(239,68,68,.45)}
.toast.error .toast-icon{color:var(--danger)}
.toast.warn{border-color:rgba(245,158,11,.45)}
.toast.warn .toast-icon{color:var(--warn)}
.toast-icon{font-size:14px;line-height:1}

/* ============ LIGHTBOX ============ */
.lightbox{position:fixed;inset:0;background:rgba(0,0,0,.96);z-index:400;display:none;align-items:center;justify-content:center;padding:24px;cursor:zoom-out;animation:fadeIn .2s}
.lightbox.open{display:flex}
.lightbox img{max-width:100%;max-height:100%;border-radius:10px;box-shadow:0 20px 60px rgba(0,0,0,.6)}

/* ============ SCROLLBAR ============ */
::-webkit-scrollbar{width:8px;height:8px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--surface-3);border-radius:4px}
::-webkit-scrollbar-thumb:hover{background:var(--border-bright)}

/* ============ FOCUS RINGS ============ */
button:focus-visible,input:focus-visible,textarea:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
</style>
</head>
<body>
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
<div class="messages" id="messages"><div class="messages-inner" id="messagesInner"></div></div>
<div class="input-bar">
  <div class="toggles">
    <button class="toggle search" id="searchToggle" title="Web search for next message">
      🔍 <span class="lbl">Search</span>
    </button>
    <button class="toggle reason" id="reasonToggle" title="Route to reasoning chain">
      🧠 <span class="lbl">Deep</span>
    </button>
  </div>
  <div class="input-row">
    <button class="attach" id="attachBtn" aria-label="attach image">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
      <span class="img-count" id="imgCount" style="display:none">0</span>
    </button>
    <input type="file" id="fileInput" accept="image/*" multiple style="display:none">
    <div class="input-wrap">
      <div class="image-previews" id="imagePreviews"></div>
      <textarea class="input" id="input" placeholder="Ask anything…" rows="1"></textarea>
    </div>
    <button class="voice" id="voiceBtn" aria-label="voice input">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
    </button>
    <button class="send" id="sendBtn" aria-label="send">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/></svg>
    </button>
    <button class="stop" id="stopBtn" aria-label="stop" style="display:none">
      <svg viewBox="0 0 24 24" fill="currentColor"><rect x="7" y="7" width="10" height="10" rx="2"/></svg>
    </button>
  </div>
  <div class="shortcut-hint" id="shortcutHint">Tap send. Ctrl+Enter also sends.</div>
</div>

<div class="modal-back" id="welcomeModal">
  <div class="modal">
    <h2>Welcome.</h2>
    <p class="sub">Your chats live in the cloud, tied to a personal sync key. Use the same key on any device to get the same chats.</p>
    <div class="tabs">
      <button class="tab active" data-tab="new">First time</button>
      <button class="tab" data-tab="have">I have a key</button>
    </div>
    <div id="tab-new">
      <label>Your new sync key</label>
      <div class="key-display" id="newKey"></div>
      <p class="sub" style="font-size:12px;margin-top:10px">Save this somewhere — password manager, a note, anywhere. It's the only way back to your chats on another device.</p>
      <div class="actions">
        <button class="btn btn-ghost" id="regenKey">Regenerate</button>
        <button class="btn btn-primary" id="useNewKey">Saved it, continue</button>
      </div>
    </div>
    <div id="tab-have" style="display:none">
      <label>Enter your sync key</label>
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
    <p class="sub">Modes, providers, memory, sync.</p>

    <div class="section-label">Default mode</div>
    <div class="model-grid" id="modelGrid"></div>

    <div class="section-label">Provider keys</div>
    <div class="providers-status" id="providersStatus"></div>

    <div class="divider"></div>

    <div class="section-label">Sync key (use on other devices)</div>
    <div class="key-display" id="myKey"></div>
    <div class="actions" style="margin-top:8px">
      <button class="btn btn-ghost" id="copyKey">
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
        Copy key
      </button>
    </div>

    <div class="divider"></div>

    <div class="section-label">What I remember about you</div>
    <div class="mem-list" id="memList"></div>
    <div class="actions" style="margin-top:12px">
      <button class="btn btn-ghost" id="clearMem">Clear all</button>
      <button class="btn btn-ghost" id="exportAll">Export chats</button>
    </div>

    <div class="divider"></div>

    <div class="section-label">About</div>
    <div class="about-block">
      <strong>Router v6.2</strong> — multi-provider fallback chain.<br>
      Auto-routes across free tiers. No money, no limits to remember.<br>
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

<div class="toast" id="toast">
  <span class="toast-icon" id="toastIcon"></span>
  <span class="toast-text" id="toastText"></span>
</div>
<div class="lightbox" id="lightbox"><img id="lightboxImg" alt=""></div>

<script>
const $ = (id) => document.getElementById(id);
const SK_STORE = "router_sync_key_v1";
const PREFS_STORE = "router_prefs_v1";

// ============ TOAST ============
let toastTimer;
function toast(msg, type = "info") {
  const t = $("toast");
  const icons = { success: "✓", error: "✕", warn: "!", info: "•" };
  $("toastIcon").textContent = icons[type] || "•";
  $("toastText").textContent = msg;
  t.className = "toast show " + type;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), 2600);
}

// ============ CUSTOM CONFIRM ============
let confirmResolver = null;
function customConfirm(title, message, btnText = "Confirm", btnDangerous = true) {
  return new Promise((resolve) => {
    $("confirmTitle").textContent = title;
    $("confirmMessage").textContent = message;
    const okBtn = $("confirmOK");
    okBtn.textContent = btnText;
    okBtn.className = "btn " + (btnDangerous ? "btn-danger" : "btn-primary");
    $("confirmModal").classList.add("open");
    confirmResolver = resolve;
  });
}
function closeConfirm(result) {
  $("confirmModal").classList.remove("open");
  if (confirmResolver) {
    confirmResolver(result);
    confirmResolver = null;
  }
}
$("confirmCancel").onclick = () => closeConfirm(false);
$("confirmOK").onclick = () => closeConfirm(true);

// ============ SYNC KEY GENERATION ============
const WORDS = ["amber","arrow","aspen","autumn","azure","basil","birch","blade","blaze","bloom","brave","brick","bridge","brook","calm","canyon","cedar","cherry","clay","clear","cloud","clover","coast","comet","copper","coral","cosmos","crane","creek","crimson","crystal","cypress","dawn","deep","desert","drift","dune","dusk","ember","emerald","fable","falcon","feather","fern","flame","flint","forest","frost","glade","golden","granite","grove","harbor","harmony","haven","hazel","heron","horizon","indigo","ivory","jade","jasper","juniper","keen","lake","lantern","laurel","leaf","light","linen","lupine","maple","marble","meadow","midnight","mist","moon","moss","mountain","north","oak","ocean","olive","onyx","opal","orchid","otter","pearl","pebble","petal","pine","plum","poppy","prairie","quartz","quiet","quill","rain","rapid","raven","reed","ridge","ripple","river","robin","rose","rust","saffron","sage","sapphire","scarlet","shade","shadow","shore","silver","sky","slate","snow","sparrow","spring","spruce","star","stone","storm","stream","summer","sunset","swift","teal","thicket","thorn","thunder","tide","tiger","topaz","trail","tundra","valley","velvet","violet","vista","walnut","wave","wheat","whisper","willow","wind","winter","wren"];
function genKey() {
  const p = () => WORDS[Math.floor(Math.random() * WORDS.length)];
  return p() + "-" + p() + "-" + p() + "-" + p();
}

// ============ MODELS ============
const MODELS_CLIENT = {
  chat:       { label: "Chat (auto)",         blurb: "Auto-routes Cerebras → Groq → Gemini → OpenRouter. Falls over instantly on rate limit.", chain: ["Cerebras", "Groq", "Gemini", "OpenRouter"] },
  uncensored: { label: "Uncensored (Venice)", blurb: "Venice via OpenRouter. No filter. Limited daily quota — use deliberately.",                chain: ["OpenRouter"] },
  reasoner:   { label: "Deep Reasoner",       blurb: "For hard multi-step problems. OpenRouter free → Cerebras Qwen Thinking → Gemini.",         chain: ["OpenRouter", "Cerebras", "Gemini"] },
};

// ============ MARKDOWN ============
marked.setOptions({ breaks: true, gfm: true });
const renderMarkdown = (t) => DOMPurify.sanitize(marked.parse(t || ""), { ADD_ATTR: ["target"] });

function decorateCodeBlocks(container) {
  container.querySelectorAll("pre").forEach((pre) => {
    if (pre.querySelector(".copy-btn")) return;
    // Language label
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
    // Copy button
    const b = document.createElement("button");
    b.className = "copy-btn";
    b.textContent = "copy";
    b.onclick = (e) => {
      e.stopPropagation();
      const code = pre.querySelector("code")?.innerText || pre.innerText;
      navigator.clipboard.writeText(code).then(() => {
        b.textContent = "copied";
        setTimeout(() => b.textContent = "copy", 1400);
      });
    };
    pre.appendChild(b);
  });
}

// ============ STATE ============
let syncKey = localStorage.getItem(SK_STORE) || "";
let prefs = JSON.parse(localStorage.getItem(PREFS_STORE) || "{}");
// Migrate legacy mode keys
if (prefs.model === "cerebras_llama" || prefs.model === "cerebras_gpt" || prefs.model === "cerebras_scout") prefs.model = "chat";
if (prefs.model === "venice") prefs.model = "uncensored";
if (prefs.model === "r1") prefs.model = "reasoner";
if (!prefs.model || !MODELS_CLIENT[prefs.model]) prefs.model = "chat";

function savePrefs() {
  localStorage.setItem(PREFS_STORE, JSON.stringify(prefs));
  $("modelLabel").textContent = MODELS_CLIENT[prefs.model]?.label || prefs.model;
}

let currentChatId = null;
let currentMessages = [];
let isStreaming = false;
let pendingImages = [];
let currentAbort = null;
let userMemoryName = null;
let useSearchNext = false;
let useReasonNext = false;
let healthCache = null;

const uuid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 8);

// ============ API ============
async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error("HTTP " + r.status);
  return r.json();
}
const apiList    = () => api("/chats?sk=" + encodeURIComponent(syncKey));
const apiGet     = (id) => api("/chats/" + id + "?sk=" + encodeURIComponent(syncKey));
const apiSave    = (c) => api("/chats/save", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ sync_key: syncKey, chat: c }) });
const apiDel     = (id) => api("/chats/" + id + "?sk=" + encodeURIComponent(syncKey), { method: "DELETE" });
const apiPin     = (id, pinned) => api("/chats/" + id + "/pin", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ sync_key: syncKey, pinned }) });
const apiMemGet  = () => api("/memories?sk=" + encodeURIComponent(syncKey));
const apiMemSave = (m) => api("/memories/save", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ sync_key: syncKey, memories: m }) });
const apiHealth  = () => api("/health");

// ============ GREETINGS (witty, adult, varied) ============
function smartGreeting() {
  const h = new Date().getHours();
  const name = userMemoryName ? ", " + userMemoryName : "";
  let pool;
  if (h < 5) {
    pool = ["Still up" + name + "?", "Late one tonight" + name + ".", "Burning the midnight oil" + name + "?", "Insomnia's a hell of a drug" + name + "."];
  } else if (h < 9) {
    pool = ["Morning" + name + ".", "Up early" + name + ".", "Hello" + (userMemoryName ? ", " + userMemoryName : "") + "."];
  } else if (h < 12) {
    pool = ["Hey" + name + ".", "Morning" + name + ".", "Hello" + (userMemoryName ? ", " + userMemoryName : "") + ".", "What's the move" + name + "?"];
  } else if (h < 17) {
    pool = ["Hey" + name + ".", "Afternoon" + name + ".", "What's up" + name + "?", "Hello" + (userMemoryName ? ", " + userMemoryName : "") + "."];
  } else if (h < 22) {
    pool = ["Evening" + name + ".", "Hey" + name + ".", "Welcome back" + name + ".", "Long day" + name + "?"];
  } else {
    pool = ["Evening" + name + ".", "Hey" + name + ".", "Still going" + name + "?", "Up late again" + name + "?"];
  }
  return pool[Math.floor(Math.random() * pool.length)];
}

async function extractNameFromMem() {
  try {
    const mems = await apiMemGet();
    for (const m of mems) {
      const match = m.match(/(?:name is|i'?m|i am|called)\\s+([A-Z][a-zA-Z]+)/i);
      if (match) { userMemoryName = match[1]; return; }
    }
  } catch {}
}

// ============ EMPTY STATE SUGGESTIONS ============
const SUGGESTIONS = [
  { icon: "📰", text: "What's interesting in the news today?" },
  { icon: "🤔", text: "Help me think through a decision" },
  { icon: "💡", text: "Explain something I'm confused about" },
  { icon: "🎨", text: "Brainstorm with me for a project" },
  { icon: "📖", text: "Write me a short story about something unexpected" },
  { icon: "🔧", text: "Help me debug something" },
  { icon: "🧪", text: "What's a smart way to approach this problem?" },
  { icon: "💬", text: "I want to vent about something" },
  { icon: "🗺", text: "Plan a trip with me" },
  { icon: "🍳", text: "What should I make for dinner?" },
  { icon: "📚", text: "Recommend something to read" },
  { icon: "🎯", text: "Help me set a goal and break it down" },
];

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
  sub.innerHTML = "Ask anything. Tap 🔍 to search the web, 🧠 for deep reasoning, paperclip for images.";
  const sg = document.createElement("div");
  sg.className = "suggestions";
  const picked = [...SUGGESTIONS].sort(() => Math.random() - 0.5).slice(0, 4);
  for (const s of picked) {
    const b = document.createElement("button");
    b.className = "sugg";
    b.innerHTML = '<span class="sg-icon">' + s.icon + '</span><span>' + s.text + '</span>';
    b.onclick = () => { $("input").value = s.text; $("input").focus(); autoresize(); };
    sg.appendChild(b);
  }
  e.appendChild(greet);
  e.appendChild(sub);
  e.appendChild(sg);
  root.appendChild(e);
}

// ============ MESSAGE BUBBLES ============
function makeBubble(role, content, images, meta, isStreamingFlag) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + role;
  const b = document.createElement("div");
  const isError = role === "assistant" && content && /^(Error:|⚠)/.test(content);
  b.className = "bubble" + (isStreamingFlag ? " empty-streaming" : "") + (isError ? " error-bubble" : "");
  if (Array.isArray(images) && images.length) {
    for (const img of images) {
      const i = document.createElement("img");
      i.src = "data:" + img.mimeType + ";base64," + img.data;
      i.className = "msg-img";
      i.alt = "uploaded";
      i.onclick = () => {
        $("lightboxImg").src = i.src;
        $("lightbox").classList.add("open");
      };
      b.appendChild(i);
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
  } else {
    if (content) {
      const t = document.createElement("div");
      t.textContent = content;
      t.style.whiteSpace = "pre-wrap";
      b.appendChild(t);
    }
  }
  wrap.appendChild(b);
  if (role === "assistant" && meta && (meta.responder || meta.tool_used)) {
    const cap = document.createElement("div");
    cap.className = "caption";
    cap.innerHTML = captionHTML(meta);
    wrap.appendChild(cap);
  }
  return { wrap, bubble: b };
}

function captionHTML(meta) {
  let s = "via " + (meta.responder || "lead");
  if (meta.fallback_used) s += ' <span class="badge b-fallback">fallback</span>';
  if (meta.tool_used === "web_search") s += ' <span class="badge b-search">searched</span>';
  if (meta.tool_used === "ask_reasoner") s += ' <span class="badge b-reason">reasoner</span>';
  return s;
}

function renderAll() {
  const root = $("messagesInner");
  if (currentMessages.length === 0) { renderEmpty(); return; }
  root.innerHTML = "";
  currentMessages.forEach((m, idx) => {
    const { wrap, bubble } = makeBubble(m.role, m.content, m.images, m.meta, false);
    const isLast = idx === currentMessages.length - 1;
    const isError = m.role === "assistant" && m.content && /^(Error:|⚠)/.test(m.content);
    if (isLast && !isStreaming) {
      const actions = document.createElement("div");
      actions.className = "msg-actions";
      if (m.role === "user") {
        const edit = document.createElement("button");
        edit.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg> Edit';
        edit.onclick = () => editMessage(idx);
        actions.appendChild(edit);
      } else if (m.role === "assistant") {
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
          copy.onclick = () => { navigator.clipboard.writeText(m.content || ""); toast("Copied", "success"); };
          actions.appendChild(regen);
          actions.appendChild(copy);
        }
      }
      wrap.appendChild(actions);
    }
    root.appendChild(wrap);
  });
  scrollBottom();
}

function scrollBottom() {
  const el = $("messages");
  el.scrollTop = el.scrollHeight;
}

// ============ CHAT LIST ============
let allChatsCache = [];
async function renderList(filter = "") {
  if (!syncKey) return;
  try {
    if (!filter) allChatsCache = await apiList();
    const f = filter.toLowerCase().trim();
    const chats = f
      ? allChatsCache.filter(c => (c.title || "").toLowerCase().includes(f) || (c.preview || "").toLowerCase().includes(f))
      : allChatsCache;
    const root = $("chatsList");
    root.innerHTML = "";
    if (chats.length === 0) {
      const e = document.createElement("div");
      e.className = "list-empty";
      e.textContent = f ? "No matches." : "No chats yet.\\nStart one — it'll show up here.";
      root.appendChild(e);
      return;
    }
    for (const c of chats) {
      const it = document.createElement("div");
      it.className = "chat-item" + (c.id === currentChatId ? " active" : "");
      const row = document.createElement("div");
      row.className = "row";
      const t = document.createElement("div");
      t.className = "chat-title";
      if (c.pinned) {
        const pin = document.createElement("span");
        pin.className = "pin-icon";
        pin.textContent = "📌";
        t.appendChild(pin);
      }
      const titleSpan = document.createElement("span");
      titleSpan.textContent = c.title || "New chat";
      titleSpan.style.overflow = "hidden";
      titleSpan.style.textOverflow = "ellipsis";
      titleSpan.style.whiteSpace = "nowrap";
      titleSpan.style.flex = "1";
      t.appendChild(titleSpan);
      const p = document.createElement("div");
      p.className = "chat-prev";
      p.textContent = c.preview || "";
      row.appendChild(t);
      if (c.preview) row.appendChild(p);
      row.onclick = () => loadChat(c.id);
      const actions = document.createElement("div");
      actions.className = "chat-actions";
      const pinBtn = document.createElement("button");
      pinBtn.title = c.pinned ? "Unpin" : "Pin";
      pinBtn.innerHTML = c.pinned
        ? '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2a3 3 0 0 0-3 3v6l-3 3v2h12v-2l-3-3V5a3 3 0 0 0-3-3zm-1 17v3l1 1 1-1v-3h-2z"/></svg>'
        : '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 0-3 3v6l-3 3v2h12v-2l-3-3V5a3 3 0 0 0-3-3zm-1 17v3l1 1 1-1v-3"/></svg>';
      pinBtn.onclick = async (e) => {
        e.stopPropagation();
        try { await apiPin(c.id, !c.pinned); renderList(filter); }
        catch { toast("Couldn't pin", "error"); }
      };
      const delBtn = document.createElement("button");
      delBtn.className = "del";
      delBtn.title = "Delete";
      delBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-2 14a2 2 0 0 1-2 2H9a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6M14 11v6"/><path d="M9 6V4a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2"/></svg>';
      delBtn.onclick = async (e) => {
        e.stopPropagation();
        const titleShort = (c.title || "this chat").slice(0, 30);
        if (await customConfirm("Delete chat?", '"' + titleShort + '" will be gone for good.', "Delete")) {
          try {
            await apiDel(c.id);
            if (c.id === currentChatId) newChat();
            renderList(filter);
            toast("Deleted", "success");
          } catch { toast("Couldn't delete", "error"); }
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
  pendingImages = [];
  useSearchNext = false;
  useReasonNext = false;
  updateToggles();
  updateImgCount();
  $("chatTitle").textContent = "Router";
  renderPreviews();
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
  const title = currentMessages.find(m => m.role === "user")?.content?.slice(0, 50) || "New chat";
  // Strip image data before storage. Deno KV value limit is 64KB and one base64 image alone is 130KB+.
  // Images are only needed for the active turn's prompt, not for persistence across reloads.
  const messagesForStorage = currentMessages.map(m => {
    if (Array.isArray(m.images) && m.images.length > 0) {
      const note = m.images.length === 1 ? "[image attached]" : "[" + m.images.length + " images attached]";
      const newContent = m.content ? (m.content + "\\n\\n" + note) : note;
      return { role: m.role, content: newContent, meta: m.meta };
    }
    return m;
  });
  try {
    await apiSave({
      id: currentChatId,
      title,
      messages: messagesForStorage,
      updated: Date.now(),
      pinned: false,
    });
    $("chatTitle").textContent = title;
    renderList();
  } catch (e) {
    // Save failures MUST be silent — never overwrite the streamed response.
    // The conversation still works in this session; it just won't survive a reload.
    console.warn("persist failed (chat continues normally):", e);
    toast("Couldn't save chat", "warn");
  }
}

// ============ SIDEBAR ============
function openSB() { $("sidebar").classList.add("open"); $("backdrop").classList.add("open"); }
function closeSB() { $("sidebar").classList.remove("open"); $("backdrop").classList.remove("open"); }

// ============ IMAGE UPLOAD ============
function renderPreviews() {
  const root = $("imagePreviews");
  root.innerHTML = "";
  pendingImages.forEach((img, idx) => {
    const item = document.createElement("div");
    item.className = "preview-item";
    const i = document.createElement("img");
    i.src = "data:" + img.mimeType + ";base64," + img.data;
    const rm = document.createElement("button");
    rm.className = "rm";
    rm.textContent = "×";
    rm.onclick = () => { pendingImages.splice(idx, 1); renderPreviews(); updateImgCount(); };
    item.appendChild(i);
    item.appendChild(rm);
    root.appendChild(item);
  });
}

function updateImgCount() {
  const c = $("imgCount");
  if (pendingImages.length === 0) { c.style.display = "none"; }
  else { c.style.display = "flex"; c.textContent = pendingImages.length; }
}

async function fileToImage(file) {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => {
      const [meta, b64] = r.result.split(",");
      res({ mimeType: meta.match(/data:([^;]+);/)[1], data: b64 });
    };
    r.onerror = rej;
    r.readAsDataURL(file);
  });
}

async function addFiles(fileList) {
  for (const f of fileList) {
    if (!f.type.startsWith("image/")) continue;
    if (f.size > 4 * 1024 * 1024) { toast("Image too big (4MB max)", "warn"); continue; }
    pendingImages.push(await fileToImage(f));
  }
  renderPreviews();
  updateImgCount();
}

$("attachBtn").onclick = () => $("fileInput").click();
$("fileInput").onchange = async (e) => { await addFiles(e.target.files); e.target.value = ""; };

// Drag and drop on whole page
document.addEventListener("dragover", (e) => { e.preventDefault(); });
document.addEventListener("drop", async (e) => { e.preventDefault(); if (e.dataTransfer?.files) await addFiles(e.dataTransfer.files); });

// Paste image support
$("input").addEventListener("paste", async (e) => {
  if (!e.clipboardData) return;
  const items = e.clipboardData.items;
  const files = [];
  for (const item of items) {
    if (item.type.startsWith("image/")) {
      const f = item.getAsFile();
      if (f) files.push(f);
    }
  }
  if (files.length) { e.preventDefault(); await addFiles(files); }
});

// ============ TOGGLES ============
function updateToggles() {
  $("searchToggle").classList.toggle("active", useSearchNext);
  $("reasonToggle").classList.toggle("active", useReasonNext);
}
$("searchToggle").onclick = () => { useSearchNext = !useSearchNext; updateToggles(); };
$("reasonToggle").onclick = () => { useReasonNext = !useReasonNext; updateToggles(); };

// ============ VOICE ============
let recog = null;
function setupVoice() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { $("voiceBtn").style.display = "none"; return; }
  recog = new SR();
  recog.continuous = false;
  recog.interimResults = true;
  recog.lang = "en-US";
  let baseText = "";
  recog.onstart = () => { baseText = $("input").value; $("voiceBtn").classList.add("recording"); };
  recog.onresult = (ev) => {
    let interim = "", final = "";
    for (let i = ev.resultIndex; i < ev.results.length; i++) {
      const t = ev.results[i][0].transcript;
      if (ev.results[i].isFinal) final += t;
      else interim += t;
    }
    $("input").value = (baseText ? baseText + " " : "") + final + interim;
    autoresize();
  };
  recog.onerror = () => $("voiceBtn").classList.remove("recording");
  recog.onend = () => $("voiceBtn").classList.remove("recording");
}
$("voiceBtn").onclick = () => {
  if (!recog) { toast("Voice input not supported in this browser", "warn"); return; }
  if ($("voiceBtn").classList.contains("recording")) recog.stop();
  else { try { recog.start(); } catch { toast("Already listening", "info"); } }
};

// ============ INPUT ============
function autoresize() {
  const e = $("input");
  e.style.height = "auto";
  e.style.height = Math.min(e.scrollHeight, 180) + "px";
}

// Enter creates new line. Send only via send button or Ctrl/Cmd+Enter.
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    send();
  }
});
$("input").addEventListener("input", autoresize);

// Placeholder rotation — adult, dry
const PLACEHOLDERS = [
  "Ask anything…",
  "What's on your mind?",
  "Type away…",
  "Throw me a curveball…",
  "What do you want to know?",
  "Spill it…",
  "I'm listening…",
  "Hit me with something…",
];
function rotatePlaceholder() {
  $("input").placeholder = PLACEHOLDERS[Math.floor(Math.random() * PLACEHOLDERS.length)];
}
rotatePlaceholder();

// ============ EDIT / REGENERATE / RETRY ============
async function regenerate() {
  if (isStreaming) return;
  if (currentMessages[currentMessages.length - 1]?.role === "assistant") {
    currentMessages.pop();
  }
  renderAll();
  await streamResponse();
}

async function retryLast() {
  if (isStreaming) return;
  // Same as regenerate — pop the error message, retry the request
  if (currentMessages[currentMessages.length - 1]?.role === "assistant") {
    currentMessages.pop();
  }
  renderAll();
  await streamResponse();
}

function editMessage(idx) {
  const msg = currentMessages[idx];
  if (msg.role !== "user") return;
  $("input").value = msg.content || "";
  pendingImages = msg.images ? msg.images.slice() : [];
  currentMessages = currentMessages.slice(0, idx);
  renderPreviews();
  updateImgCount();
  renderAll();
  autoresize();
  $("input").focus();
}

// ============ SEND ============
async function send() {
  if (isStreaming) return;
  const input = $("input");
  const text = input.value.trim();
  if (!text && pendingImages.length === 0) return;
  currentMessages.push({
    role: "user",
    content: text,
    images: pendingImages.slice(),
  });
  pendingImages = [];
  renderPreviews();
  updateImgCount();
  input.value = "";
  autoresize();
  rotatePlaceholder();
  await streamResponse();
}

async function streamResponse() {
  isStreaming = true;
  $("sendBtn").style.display = "none";
  $("stopBtn").style.display = "flex";
  $("voiceBtn").disabled = true;
  const requestedSearch = useSearchNext;
  const requestedReason = useReasonNext;
  const aMsg = { role: "assistant", content: "", meta: {} };
  currentMessages.push(aMsg);
  if ($("messagesInner").querySelector(".empty")) $("messagesInner").innerHTML = "";
  $("messagesInner").innerHTML = "";
  for (let i = 0; i < currentMessages.length - 1; i++) {
    const m = currentMessages[i];
    const { wrap } = makeBubble(m.role, m.content, m.images, m.meta, false);
    $("messagesInner").appendChild(wrap);
  }
  const { wrap: aWrap, bubble: aBub } = makeBubble("assistant", "", null, null, true);
  $("messagesInner").appendChild(aWrap);
  scrollBottom();
  currentAbort = new AbortController();
  try {
    const resp = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sync_key: syncKey,
        model: prefs.model,
        use_search: requestedSearch,
        use_reasoning: requestedReason,
        messages: currentMessages.slice(0, -1).map(m => ({
          role: m.role,
          content: m.content,
          images: m.images || [],
        })),
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
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split("\\n\\n");
      buf = parts.pop() || "";
      for (const evt of parts) {
        const lines = evt.split("\\n");
        let et = "", data = "";
        for (const ln of lines) {
          if (ln.startsWith("event: ")) et = ln.slice(7);
          else if (ln.startsWith("data: ")) data = ln.slice(6);
        }
        if (!data) continue;
        let p;
        try { p = JSON.parse(data); } catch { continue; }
        if (et === "status") {
          aMsg.meta.responder = p.leader;
        } else if (et === "note") {
          if (toolEl) toolEl.remove();
          toolEl = document.createElement("div");
          toolEl.className = "tool-running note";
          toolEl.textContent = "→ " + p.msg;
          aWrap.insertBefore(toolEl, aBub);
          setTimeout(() => { if (toolEl) { toolEl.remove(); toolEl = null; } }, 3500);
        } else if (et === "token") {
          aMsg.content += p.text;
          aBub.classList.remove("empty-streaming");
          aBub.innerHTML = renderMarkdown(aMsg.content);
          decorateCodeBlocks(aBub);
          scrollBottom();
        } else if (et === "tool_start") {
          aMsg.meta.tool_used = p.tool;
          if (toolEl) toolEl.remove();
          toolEl = document.createElement("div");
          toolEl.className = "tool-running " + (p.tool === "web_search" ? "search" : "reason");
          toolEl.innerHTML = (p.tool === "web_search" ? "🔍 searching the web…" : "🧠 reasoning…");
          aWrap.insertBefore(toolEl, aBub);
        } else if (et === "tool_done") {
          if (toolEl) { toolEl.remove(); toolEl = null; }
          if (!aMsg.content) {
            const t = document.createElement("div");
            t.className = "thinking";
            t.innerHTML = "<span></span><span></span><span></span>";
            aBub.appendChild(t);
          }
        } else if (et === "done") {
          Object.assign(aMsg.meta, p);
        } else if (et === "aborted") {
          aMsg.content += (aMsg.content ? "\\n\\n" : "") + "_[stopped]_";
          aBub.innerHTML = renderMarkdown(aMsg.content);
        } else if (et === "error") {
          aMsg.content += (aMsg.content ? "\\n\\n" : "") + "⚠ " + p.message;
          aBub.classList.add("error-bubble");
          aBub.innerHTML = renderMarkdown(aMsg.content);
        }
      }
    }
    if (toolEl) toolEl.remove();
    if (!aMsg.content) {
      aMsg.content = "_(empty response — hit retry or try another mode)_";
    }
    aBub.innerHTML = renderMarkdown(aMsg.content);
    decorateCodeBlocks(aBub);
    aBub.classList.remove("empty-streaming");
    if (aMsg.meta.responder || aMsg.meta.tool_used) {
      const cap = document.createElement("div");
      cap.className = "caption";
      cap.innerHTML = captionHTML(aMsg.meta);
      aWrap.appendChild(cap);
    }
    await persist();
  } catch (e) {
    if (e.name === "AbortError") {
      aMsg.content += (aMsg.content ? "\\n\\n" : "") + "_[stopped]_";
      aBub.innerHTML = renderMarkdown(aMsg.content);
    }
    // Note: other errors (like persist failures) are swallowed inside persist().
    // This catch only fires for fetch/network failures during streaming.
    else if (e.message && !e.message.startsWith("HTTP")) {
      aMsg.content = "⚠ Network glitch: " + e.message + ". Try again.";
      aBub.classList.add("error-bubble");
      aBub.innerHTML = renderMarkdown(aMsg.content);
    }
  } finally {
    isStreaming = false;
    currentAbort = null;
    $("sendBtn").style.display = "flex";
    $("stopBtn").style.display = "none";
    $("voiceBtn").disabled = false;
    useSearchNext = false;
    useReasonNext = false;
    updateToggles();
    scrollBottom();
    renderAll();
  }
}

$("stopBtn").onclick = () => { if (currentAbort) currentAbort.abort(); };
$("sendBtn").onclick = send;

// ============ WELCOME MODAL ============
function showWelcome() { $("welcomeModal").classList.add("open"); refreshNewKey(); }
function refreshNewKey() { $("newKey").textContent = genKey(); }
function pickTab(name) {
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.dataset.tab === name));
  $("tab-new").style.display = name === "new" ? "" : "none";
  $("tab-have").style.display = name === "have" ? "" : "none";
}
document.querySelectorAll(".tab").forEach(t => t.onclick = () => pickTab(t.dataset.tab));
$("regenKey").onclick = refreshNewKey;
$("useNewKey").onclick = async () => {
  syncKey = $("newKey").textContent.trim();
  localStorage.setItem(SK_STORE, syncKey);
  $("welcomeModal").classList.remove("open");
  toast("Welcome.", "success");
  await renderList();
  renderAll();
};
$("useExistingKey").onclick = async () => {
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

// ============ SETTINGS MODAL ============
async function buildModelGrid() {
  if (!healthCache) {
    try { healthCache = await apiHealth(); } catch { healthCache = {}; }
  }
  const grid = $("modelGrid");
  grid.innerHTML = "";
  for (const [key, info] of Object.entries(MODELS_CLIENT)) {
    const card = document.createElement("button");
    card.className = "model-card" + (prefs.model === key ? " selected" : "");
    const chainStr = info.chain.join(" → ");
    card.innerHTML = '<div class="info"><div class="name">' + info.label + '</div><div class="blurb">' + info.blurb + '</div><div class="chain-list">' + chainStr + '</div></div><div class="check">✓</div>';
    card.onclick = () => {
      prefs.model = key;
      savePrefs();
      buildModelGrid();
      toast("Mode: " + info.label, "success");
    };
    grid.appendChild(card);
  }
}

async function buildProvidersStatus() {
  if (!healthCache) {
    try { healthCache = await apiHealth(); } catch { healthCache = {}; }
  }
  const root = $("providersStatus");
  root.innerHTML = "";
  const providers = healthCache?.providers || {};
  const items = [
    ["cerebras",   "Cerebras",   "1M tokens/day · primary chat"],
    ["gemini",     "Google AI",  "1500 req/day · chat fallback + vision"],
    ["openrouter", "OpenRouter", "~50 req/day · Venice + auto-router"],
    ["groq",       "Groq",       "optional · chat fallback"],
    ["tavily",     "Tavily",     "1000/month · web search"],
  ];
  for (const [k, label, note] of items) {
    const row = document.createElement("div");
    row.className = "row";
    const has = !!providers[k];
    row.innerHTML = '<span class="label">' + label + ' <span class="note">' + note + '</span></span><span class="' + (has ? "ok" : "miss") + '">' + (has ? "✓ set" : "✗ not set") + '</span>';
    root.appendChild(row);
  }
  // Version line
  $("aboutVersion").textContent = "Running: " + (healthCache?.version || "?");
}

async function openSettings() {
  $("myKey").textContent = syncKey;
  await buildModelGrid();
  await buildProvidersStatus();
  const mems = await apiMemGet().catch(() => []);
  const list = $("memList");
  list.innerHTML = "";
  if (mems.length === 0) {
    const e = document.createElement("div");
    e.className = "mem-empty";
    e.textContent = "Nothing learned yet. Have a few conversations and check back.";
    list.appendChild(e);
  } else {
    for (const m of mems) {
      const it = document.createElement("div");
      it.className = "mem-item";
      const t = document.createElement("div");
      t.className = "text";
      t.textContent = m;
      const rm = document.createElement("button");
      rm.className = "rm";
      rm.textContent = "×";
      rm.title = "Forget this";
      rm.onclick = async () => {
        const f = mems.filter(x => x !== m);
        try {
          await apiMemSave(f);
          openSettings();
          await extractNameFromMem();
        } catch { toast("Couldn't forget that", "error"); }
      };
      it.appendChild(t);
      it.appendChild(rm);
      list.appendChild(it);
    }
  }
  $("settingsModal").classList.add("open");
}

$("copyKey").onclick = () => { navigator.clipboard.writeText(syncKey); toast("Key copied", "success"); };
$("clearMem").onclick = async () => {
  if (await customConfirm("Wipe memory?", "I'll forget everything I've learned about you.", "Wipe")) {
    try {
      await apiMemSave([]);
      userMemoryName = null;
      openSettings();
      toast("Cleared", "success");
    } catch { toast("Couldn't clear", "error"); }
  }
};
$("exportAll").onclick = async () => {
  try {
    const chats = await apiList();
    const full = [];
    for (const c of chats) full.push(await apiGet(c.id));
    const blob = new Blob([JSON.stringify({ exported: new Date().toISOString(), chats: full }, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "router-chats-" + Date.now() + ".json";
    a.click();
    toast("Exported", "success");
  } catch { toast("Export failed", "error"); }
};
$("closeSettings").onclick = () => $("settingsModal").classList.remove("open");
$("settingsBtn").onclick = openSettings;

// ============ NAV & BACKDROP ============
$("menuBtn").onclick = openSB;
$("backdrop").onclick = () => {
  closeSB();
  $("welcomeModal").classList.remove("open");
  $("settingsModal").classList.remove("open");
  $("confirmModal").classList.remove("open");
  if (confirmResolver) { confirmResolver(false); confirmResolver = null; }
};
$("newBtn").onclick = newChat;

// Debounced search
let searchTimer;
$("searchBox").oninput = (e) => {
  clearTimeout(searchTimer);
  const v = e.target.value;
  searchTimer = setTimeout(() => renderList(v), 150);
};
$("lightbox").onclick = () => $("lightbox").classList.remove("open");

// ============ EASTER EGG: triple-tap title shows stats ============
let titleTaps = 0, titleTapTimer = null;
$("titleWrap").onclick = async () => {
  titleTaps++;
  clearTimeout(titleTapTimer);
  titleTapTimer = setTimeout(() => titleTaps = 0, 600);
  if (titleTaps >= 3) {
    titleTaps = 0;
    try {
      const h = await apiHealth();
      const active = Object.entries(h.providers || {}).filter(([,v]) => v).map(([k]) => k).join(", ");
      toast("v" + h.version + " · providers: " + (active || "none"), "info");
    } catch {}
  }
};

// ============ ESC closes things ============
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    if ($("confirmModal").classList.contains("open")) closeConfirm(false);
    else if ($("settingsModal").classList.contains("open")) $("settingsModal").classList.remove("open");
    else if ($("welcomeModal").classList.contains("open") && syncKey) $("welcomeModal").classList.remove("open");
    else if ($("sidebar").classList.contains("open")) closeSB();
    else if ($("lightbox").classList.contains("open")) $("lightbox").classList.remove("open");
  }
});

// ============ INIT ============
setupVoice();
savePrefs();
(async () => {
  if (!syncKey) { showWelcome(); return; }
  await extractNameFromMem();
  await renderList();
  renderAll();
})();
</script>
</body>
</html>`;


const MANIFEST = JSON.stringify({
  name: "Router",
  short_name: "Router",
  start_url: "/",
  display: "standalone",
  background_color: "#09090b",
  theme_color: "#09090b",
  icons: [{ src: "/icon.svg", sizes: "any", type: "image/svg+xml", purpose: "any" }],
});

const ICON_SVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 192 192"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#6366f1"/><stop offset="1" stop-color="#a855f7"/></linearGradient></defs><rect width="192" height="192" fill="url(#g)" rx="42"/><text x="96" y="132" font-size="115" text-anchor="middle" fill="white" font-family="Inter,sans-serif" font-weight="700">R</text></svg>`;

// ============================================================
// SERVER
// ============================================================
Deno.serve(async (req) => {
  const url = new URL(req.url);
  const p = url.pathname;

  if (p === "/" && req.method === "GET") {
    return new Response(HTML, { headers: { "Content-Type": "text/html; charset=utf-8" } });
  }
  if (p === "/manifest.json") {
    return new Response(MANIFEST, { headers: { "Content-Type": "application/json" } });
  }
  if (p === "/icon.svg") {
    return new Response(ICON_SVG, { headers: { "Content-Type": "image/svg+xml" } });
  }

  if (p === "/chat" && req.method === "POST") return handleChat(req);
  if (p === "/chats" && req.method === "GET") return handleChatsList(req);
  if (p === "/chats/save" && req.method === "POST") return handleChatSave(req);

  const pinMatch = p.match(/^\/chats\/([^\/]+)\/pin$/);
  if (pinMatch && req.method === "POST") return handleChatPin(req, pinMatch[1]);
  const cm = p.match(/^\/chats\/([^\/]+)$/);
  if (cm) {
    if (req.method === "GET") return handleChatGet(req, cm[1]);
    if (req.method === "DELETE") return handleChatDelete(req, cm[1]);
  }

  if (p === "/memories" && req.method === "GET") return handleMemGet(req);
  if (p === "/memories/save" && req.method === "POST") return handleMemSave(req);

  if (p === "/health") {
    return Response.json({
      ok: true,
      version: "v6.2",
      providers: {
        openrouter: !!OPENROUTER_KEY,
        tavily: !!TAVILY_KEY,
        gemini: !!GOOGLE_AI_KEY,
        cerebras: !!CEREBRAS_KEY,
        groq: !!GROQ_KEY,
      },
      models: Object.fromEntries(
        Object.entries(MODELS).map(([k, v]) => [k, {
          label: v.label,
          chain: v.chain.map(c => `${c.provider}:${c.id}`),
        }])
      ),
    });
  }

  return new Response("Not found", { status: 404 });
});
