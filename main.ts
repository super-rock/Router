// ============================================================
// PERSONAL AI ROUTER  v4
//
// Design: manual orchestration, not agent loops.
//   - User toggles: 🔍 search, 🧠 deep reasoning
//   - Image attached → auto vision model
//   - One streaming call per turn, retries on transient errors only
//   - No silent fallback chains, clear errors when something breaks
//
// Verified OpenRouter free model IDs (May 2026):
//   - cognitivecomputations/dolphin-mistral-24b-venice-edition:free
//   - meta-llama/llama-3.3-70b-instruct:free
//   - deepseek/deepseek-r1:free
//   - meta-llama/llama-3.2-11b-vision-instruct:free
//
// Env vars (set in Deno Deploy app Settings):
//   OPENROUTER_KEY   (required)
//   TAVILY_KEY       (required for search)
// ============================================================

const OPENROUTER_KEY = Deno.env.get("OPENROUTER_KEY") || "";
const TAVILY_KEY     = Deno.env.get("TAVILY_KEY")     || "";

const kv = await Deno.openKv();

// ------------------------------------------------------------
// Model registry — ONLY verified working OpenRouter free IDs
// ------------------------------------------------------------
type ModelSpec = {
  id: string;
  label: string;
  blurb: string;
  vision: boolean;
};

const MODELS: Record<string, ModelSpec> = {
  venice: {
    id: "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
    label: "Venice (Uncensored)",
    blurb: "No content filter. Default. Best for chat, creative work, candid talk.",
    vision: false,
  },
  llama: {
    id: "meta-llama/llama-3.3-70b-instruct:free",
    label: "Llama 3.3 70B",
    blurb: "Solid general-purpose. Strong reasoning. Some content filtering.",
    vision: false,
  },
  r1: {
    id: "deepseek/deepseek-r1:free",
    label: "DeepSeek R1 (Reasoning)",
    blurb: "Deep multi-step thinker. Slower but careful. Engaged by Deep toggle.",
    vision: false,
  },
  vision: {
    id: "meta-llama/llama-3.2-11b-vision-instruct:free",
    label: "Llama 3.2 Vision",
    blurb: "Sees images. Auto-engaged when you attach a photo.",
    vision: true,
  },
};

const DEFAULT_LEAD = "venice";
const REASONER     = "r1";
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
    ? `\n\n# Note\nThe user's message above includes web search results. Use them to give an accurate, current answer. Cite the sources at the end.`
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
  if (!TAVILY_KEY) return "Search unavailable: TAVILY_KEY not set.";
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
// OpenRouter call with retry on transient errors only
// ------------------------------------------------------------
async function callOpenRouter(
  modelId: string,
  messages: any[],
  opts: { stream?: boolean; signal?: AbortSignal } = {},
): Promise<Response> {
  if (!OPENROUTER_KEY) {
    return new Response(JSON.stringify({ error: "OPENROUTER_KEY not set on server" }), { status: 500 });
  }

  const body: any = { model: modelId, messages };
  if (opts.stream) body.stream = true;

  const backoffs = [0, 1500, 4000];

  for (let attempt = 0; attempt < backoffs.length; attempt++) {
    if (backoffs[attempt]) await new Promise(r => setTimeout(r, backoffs[attempt]));

    let r: Response;
    try {
      r = await fetch("https://openrouter.ai/api/v1/chat/completions", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${OPENROUTER_KEY}`,
          "Content-Type": "application/json",
          "HTTP-Referer": "https://router.super-rock.deno.net",
          "X-Title": "Personal Router",
        },
        body: JSON.stringify(body),
        signal: opts.signal,
      });
    } catch (e) {
      if (attempt === backoffs.length - 1) {
        return new Response(JSON.stringify({ error: String(e) }), { status: 599 });
      }
      continue;
    }

    if (r.ok) return r;

    // Retry on transient: 429 (rate limit), 502/503/504 (gateway issues)
    const transient = r.status === 429 || (r.status >= 502 && r.status <= 504);
    if (!transient) return r;
    if (attempt === backoffs.length - 1) return r;
  }
  return new Response("unreachable", { status: 500 });
}

async function callOpenRouterText(modelId: string, messages: any[]): Promise<string> {
  const r = await callOpenRouter(modelId, messages);
  if (!r.ok) return `(Model error HTTP ${r.status})`;
  const d = await r.json();
  return d.choices?.[0]?.message?.content || "";
}

// ------------------------------------------------------------
// Convert our message format (with images) to OpenAI multimodal format
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

// ------------------------------------------------------------
// SSE helper
// ------------------------------------------------------------
const sseChunk = (event: string, data: any): Uint8Array =>
  new TextEncoder().encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);

// ------------------------------------------------------------
// Friendly error formatter
// ------------------------------------------------------------
function friendlyError(status: number, body: string, modelLabel: string): string {
  if (status === 401 || status === 403) {
    return `Auth issue with ${modelLabel}. Check that OPENROUTER_KEY is set correctly in Deno Deploy env vars.`;
  }
  if (status === 404) {
    return `${modelLabel} not available on OpenRouter right now. Try switching to a different model in Settings.`;
  }
  if (status === 429) {
    return `Hit OpenRouter's rate limit on ${modelLabel}. Free tier is ~20 req/min, 200 req/day. Wait a minute and try again, or switch models.`;
  }
  if (status >= 500 && status <= 504) {
    return `${modelLabel} is having a server hiccup (HTTP ${status}). Try again in a moment.`;
  }
  // Try to extract OpenRouter's own error message
  try {
    const parsed = JSON.parse(body);
    const m = parsed?.error?.message || parsed?.message;
    if (m) return `${modelLabel}: ${String(m).slice(0, 200)}`;
  } catch {}
  return `${modelLabel} returned HTTP ${status}. ${body.slice(0, 150)}`;
}

// ------------------------------------------------------------
// Main turn handler — no agent loop, single streaming call
// ------------------------------------------------------------
async function runTurn(
  controller: ReadableStreamDefaultController,
  messages: any[],
  leadKey: string,
  useSearch: boolean,
  useReasoning: boolean,
  systemPromptBase: (hasSearch: boolean) => string,
  signal: AbortSignal,
) {
  const send = (event: string, data: any) => {
    try { controller.enqueue(sseChunk(event, data)); } catch {}
  };

  // Decide model based on inputs
  const lastUser = messages[messages.length - 1];
  const hasImages = Array.isArray(lastUser?.images) && lastUser.images.length > 0;

  let actualLead = leadKey;
  let routingNote: string | null = null;

  if (hasImages && !MODELS[leadKey]?.vision) {
    actualLead = VISION;
    routingNote = "Image detected → routed to vision model";
  } else if (useReasoning) {
    actualLead = REASONER;
    routingNote = "Deep reasoning → DeepSeek R1";
  }

  if (!MODELS[actualLead]) {
    send("error", { message: `Unknown model: ${actualLead}` });
    return;
  }

  send("status", { leader: MODELS[actualLead].label });
  if (routingNote) send("note", { msg: routingNote });

  // Run Tavily search if requested (and no image — search + image is awkward)
  let augmentedMessages = [...messages];
  let didSearch = false;

  if (useSearch && !hasImages && lastUser?.content) {
    send("tool_start", { tool: "web_search" });
    const searchResult = await tavilySearch(lastUser.content);
    send("tool_done", { tool: "web_search", preview: searchResult.slice(0, 200) });
    didSearch = true;

    // Replace last user message with augmented version
    const lastIdx = augmentedMessages.length - 1;
    const orig = augmentedMessages[lastIdx];
    augmentedMessages[lastIdx] = {
      ...orig,
      content: `Web search results (for context — use these to answer):\n\n${searchResult}\n\n---\n\nMy question: ${orig.content}`,
    };
  }

  // Build OpenAI-format messages
  const { msgs: oaiMsgs } = toOAIMessages(augmentedMessages);
  const sysContent = systemPromptBase(didSearch);
  const finalMsgs = [{ role: "system", content: sysContent }, ...oaiMsgs];

  // Single streaming call
  const r = await callOpenRouter(MODELS[actualLead].id, finalMsgs, {
    stream: true,
    signal,
  });

  if (!r.ok || !r.body) {
    let errText = "";
    try { errText = await r.text(); } catch {}
    send("error", { message: friendlyError(r.status, errText, MODELS[actualLead].label) });
    return;
  }

  // Stream tokens
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let gotAnyToken = false;

  while (true) {
    if (signal.aborted) {
      try { reader.cancel(); } catch {}
      send("aborted", {});
      return;
    }
    let chunk;
    try {
      chunk = await reader.read();
    } catch (e) {
      send("error", { message: `Stream interrupted: ${e}` });
      return;
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
      // OpenRouter sometimes returns error objects mid-stream
      if (j.error) {
        send("error", { message: friendlyError(j.error.code || 500, JSON.stringify(j.error), MODELS[actualLead].label) });
        return;
      }
      const delta = j.choices?.[0]?.delta;
      if (delta?.content) {
        gotAnyToken = true;
        send("token", { text: delta.content });
      }
    }
  }

  if (!gotAnyToken) {
    send("error", { message: `${MODELS[actualLead].label} returned an empty response. Try again or switch models.` });
    return;
  }

  send("done", {
    responder: MODELS[actualLead].label,
    tool_used: didSearch ? "web_search" : (actualLead === REASONER ? "ask_reasoner" : null),
  });
}

// ------------------------------------------------------------
// Memory extraction — every 3rd user message, async
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

  try {
    const text = await callOpenRouterText(MODELS.venice.id, [{ role: "user", content: prompt }]);
    if (!text) return;
    // Strip markdown fences if present
    const cleaned = text.replace(/^```(?:json)?\s*/i, "").replace(/```\s*$/i, "").trim();
    let parsed: any;
    try {
      parsed = JSON.parse(cleaned);
    } catch {
      // Try to find JSON array in the text
      const m = cleaned.match(/\[[\s\S]*\]/);
      if (!m) return;
      try { parsed = JSON.parse(m[0]); } catch { return; }
    }
    if (!Array.isArray(parsed) || parsed.length === 0) return;
    const valid = parsed.filter((x: any) => typeof x === "string" && x.length > 3 && x.length < 200);
    if (valid.length === 0) return;
    await putMem(uh, [...existing, ...valid]);
  } catch {
    // swallow — memory extraction is best-effort
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
  const leadKey = model && MODELS[model] ? model : DEFAULT_LEAD;

  // System prompt is a function so search awareness can be set later
  const sysPromptBase = (hasSearch: boolean) => buildSystemPrompt(mems, hasSearch);

  const ac = new AbortController();
  req.signal.addEventListener("abort", () => ac.abort());

  const stream = new ReadableStream({
    async start(controller) {
      try {
        await runTurn(
          controller,
          messages,
          leadKey,
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
  catch { return new Response("Invalid JSON", { status: 400 }); }
  const { sync_key, chat } = body;
  if (!sync_key || !chat?.id) return new Response("bad payload", { status: 400 });
  const uh = await hashKey(sync_key);
  await putChat(uh, chat);
  maybeExtractMemories(uh, chat);
  return Response.json({ ok: true });
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
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify@3.1.6/dist/purify.min.js"></script>
<style>
:root{
  --bg:#09090b; --panel:#121214; --surface:#17171a; --surface-2:#1d1d22;
  --border:#26262b; --border-strong:#3a3a42;
  --text:#ededf0; --text-dim:#a1a1aa; --text-muted:#71717a;
  --accent:#6366f1; --accent-soft:rgba(99,102,241,.15);
  --search:#f59e0b; --search-soft:rgba(245,158,11,.15);
  --reason:#a855f7; --reason-soft:rgba(168,85,247,.15);
  --danger:#ef4444; --success:#10b981;
  --user-grad:linear-gradient(135deg,#6366f1 0%,#8b5cf6 100%);
  --shadow:0 8px 32px rgba(0,0,0,.4);
}
*,*::before,*::after{box-sizing:border-box}
html,body{margin:0;padding:0;height:100%;background:var(--bg);color:var(--text);font-family:"Inter",-apple-system,BlinkMacSystemFont,system-ui,sans-serif;-webkit-font-smoothing:antialiased;overflow:hidden;font-size:15px;line-height:1.5}
body{display:flex;flex-direction:column}
button,input,textarea,select{font-family:inherit}

/* Header */
.header{display:flex;align-items:center;padding:12px 14px;border-bottom:1px solid var(--border);flex-shrink:0;gap:4px;padding-top:max(12px,env(safe-area-inset-top));background:var(--bg);position:relative;z-index:5}
.icon-btn{background:transparent;border:none;color:var(--text-dim);cursor:pointer;padding:8px;border-radius:10px;line-height:1;display:flex;align-items:center;justify-content:center;width:38px;height:38px;transition:background .15s,color .15s}
.icon-btn:hover{color:var(--text)}
.icon-btn:active{background:var(--surface);transform:scale(.95)}
.icon-btn svg{width:20px;height:20px}
.title-wrap{flex:1;text-align:center;padding:0 4px;overflow:hidden}
.title{font-size:14px;color:var(--text);font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;letter-spacing:-.01em}
.model-badge{font-size:10px;color:var(--text-muted);margin-top:2px;display:flex;align-items:center;justify-content:center;gap:4px}
.model-badge .dot{width:5px;height:5px;border-radius:50%;background:var(--success)}

/* Sidebar */
.sidebar{position:fixed;top:0;left:0;bottom:0;width:300px;max-width:85vw;background:var(--panel);transform:translateX(-100%);transition:transform .25s cubic-bezier(.4,0,.2,1);z-index:100;display:flex;flex-direction:column;border-right:1px solid var(--border);padding-top:env(safe-area-inset-top);padding-bottom:env(safe-area-inset-bottom);box-shadow:var(--shadow)}
.sidebar.open{transform:translateX(0)}
.sb-head{padding:12px;border-bottom:1px solid var(--border);display:flex;flex-direction:column;gap:8px}
.new-chat{width:100%;background:var(--user-grad);color:#fff;border:none;padding:11px 14px;border-radius:10px;font-size:14px;cursor:pointer;font-weight:600;display:flex;align-items:center;justify-content:center;gap:8px}
.new-chat:active{transform:scale(.98);filter:brightness(.95)}
.search-box{width:100%;background:var(--surface);border:1px solid var(--border);color:var(--text);padding:8px 12px;border-radius:8px;font-size:13px;outline:none}
.search-box:focus{border-color:var(--accent)}
.chats-list{flex:1;overflow-y:auto;padding:6px 8px}
.chat-item{padding:10px 12px;border-radius:10px;cursor:pointer;font-size:13.5px;color:var(--text-dim);margin-bottom:2px;display:flex;align-items:center;gap:8px;transition:background .12s}
.chat-item:hover,.chat-item.active{background:var(--surface);color:var(--text)}
.chat-item .row{flex:1;overflow:hidden;display:flex;flex-direction:column;gap:2px;min-width:0}
.chat-title{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:500;display:flex;align-items:center;gap:6px}
.chat-prev{font-size:11px;color:var(--text-muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pin-icon{font-size:11px}
.chat-actions{display:flex;gap:2px;opacity:0;transition:opacity .12s}
.chat-item:hover .chat-actions,.chat-item:active .chat-actions{opacity:1}
.chat-actions button{background:transparent;border:none;color:var(--text-muted);cursor:pointer;font-size:12px;padding:3px 6px;border-radius:5px}
.chat-actions button:hover{color:var(--text);background:var(--surface-2)}
.chat-actions .del:hover{color:var(--danger)}

.backdrop{position:fixed;inset:0;background:rgba(0,0,0,.6);backdrop-filter:blur(2px);-webkit-backdrop-filter:blur(2px);z-index:99;opacity:0;pointer-events:none;transition:opacity .2s}
.backdrop.open{opacity:1;pointer-events:auto}

/* Messages */
.messages{flex:1;overflow-y:auto;padding:18px 14px;scroll-behavior:smooth}
.messages-inner{max-width:780px;margin:0 auto;display:flex;flex-direction:column;gap:18px}
.msg{display:flex;flex-direction:column;gap:5px;max-width:100%;animation:fadeUp .22s cubic-bezier(.4,0,.2,1)}
@keyframes fadeUp{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
.msg.user{align-items:flex-end}
.msg.assistant{align-items:flex-start}
.bubble{padding:11px 15px;border-radius:18px;line-height:1.55;word-wrap:break-word;overflow-wrap:break-word;font-size:15px;max-width:92%;position:relative}
.msg.user .bubble{background:var(--user-grad);color:#fff;border-bottom-right-radius:5px;box-shadow:0 2px 8px rgba(99,102,241,.18)}
.msg.assistant .bubble{background:var(--surface);color:var(--text);border-bottom-left-radius:5px;border:1px solid var(--border)}
.bubble.empty-streaming{min-height:40px}
.msg-img{max-width:240px;max-height:240px;border-radius:12px;margin:4px 0;display:block;border:1px solid var(--border);object-fit:cover;cursor:zoom-in}
.bubble p{margin:0 0 .5em}
.bubble p:last-child{margin:0}
.bubble pre{background:#0c0c0e;border:1px solid var(--border);border-radius:10px;padding:12px 14px;overflow-x:auto;margin:8px 0;font-family:"JetBrains Mono",ui-monospace,monospace;font-size:13px;line-height:1.5;position:relative}
.bubble code{font-family:"JetBrains Mono",ui-monospace,monospace;font-size:.9em;background:rgba(255,255,255,.06);padding:2px 6px;border-radius:5px}
.bubble pre code{background:transparent;padding:0;border-radius:0;font-size:13px}
.msg.user .bubble code{background:rgba(255,255,255,.18)}
.msg.user .bubble pre{background:rgba(0,0,0,.25);border-color:rgba(255,255,255,.15)}
.bubble ul,.bubble ol{margin:4px 0 4px 18px;padding:0}
.bubble li{margin:2px 0}
.bubble h1,.bubble h2,.bubble h3{margin:10px 0 6px;font-weight:600}
.bubble h1{font-size:1.2em}.bubble h2{font-size:1.1em}.bubble h3{font-size:1.05em}
.bubble a{color:#a5b4fc;text-decoration:underline;text-underline-offset:2px}
.bubble blockquote{border-left:3px solid var(--border-strong);padding-left:12px;color:var(--text-dim);margin:6px 0}
.bubble em{font-style:italic;color:var(--text-dim)}

.msg-actions{display:flex;gap:4px;margin-top:2px;opacity:0;transition:opacity .12s;padding:0 6px}
.msg:hover .msg-actions,.msg:active .msg-actions{opacity:.85}
.msg-actions button{background:transparent;border:1px solid var(--border);color:var(--text-muted);cursor:pointer;font-size:11px;padding:4px 9px;border-radius:6px;display:flex;align-items:center;gap:4px}
.msg-actions button:hover{color:var(--text);background:var(--surface)}

.caption{font-size:11px;color:var(--text-muted);padding:0 8px;display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.badge{font-size:10px;padding:2px 7px;border-radius:5px;font-weight:500}
.b-search{background:var(--search-soft);color:var(--search)}
.b-reason{background:var(--reason-soft);color:var(--reason)}
.b-note{background:rgba(16,185,129,.13);color:var(--success)}

.thinking{display:inline-flex;gap:3px;padding:6px 0;align-items:center}
.thinking span{width:6px;height:6px;border-radius:50%;background:var(--text-muted);animation:pulse 1.3s ease-in-out infinite}
.thinking span:nth-child(2){animation-delay:.15s}
.thinking span:nth-child(3){animation-delay:.3s}
@keyframes pulse{0%,80%,100%{opacity:.3;transform:scale(.85)}40%{opacity:1;transform:scale(1)}}

.tool-running{display:inline-flex;align-items:center;gap:8px;font-size:12px;color:var(--text-dim);padding:7px 12px;background:var(--surface);border:1px solid var(--border);border-radius:10px;margin-bottom:4px;width:fit-content;animation:fadeUp .2s}
.tool-running.search{color:var(--search);border-color:rgba(245,158,11,.3)}
.tool-running.reason{color:var(--reason);border-color:rgba(168,85,247,.3)}
.tool-running.note{color:var(--success);border-color:rgba(16,185,129,.3)}

.empty{text-align:center;padding:40px 20px;max-width:520px;margin:30px auto 0}
.empty .greet{font-size:28px;font-weight:700;letter-spacing:-.02em;background:var(--user-grad);-webkit-background-clip:text;background-clip:text;color:transparent;margin-bottom:6px;line-height:1.2}
.empty .sub{color:var(--text-dim);font-size:14px;margin-bottom:24px}
.suggestions{display:flex;flex-direction:column;gap:8px;max-width:380px;margin:0 auto}
.sugg{background:var(--surface);border:1px solid var(--border);color:var(--text-dim);padding:10px 14px;border-radius:12px;cursor:pointer;font-size:13.5px;text-align:left;transition:all .15s}
.sugg:hover{background:var(--surface-2);color:var(--text);border-color:var(--border-strong)}
.sugg:active{transform:scale(.98)}

/* Input area with toggles */
.input-bar{border-top:1px solid var(--border);background:var(--bg);flex-shrink:0;padding-bottom:max(8px,env(safe-area-inset-bottom))}
.toggles{display:flex;gap:6px;padding:8px 12px 0;max-width:780px;margin:0 auto;flex-wrap:wrap}
.toggle{background:transparent;border:1px solid var(--border);color:var(--text-muted);padding:5px 10px;border-radius:14px;font-size:12px;cursor:pointer;display:flex;align-items:center;gap:5px;transition:all .15s;font-weight:500}
.toggle:hover{color:var(--text);border-color:var(--border-strong)}
.toggle.active.search{background:var(--search-soft);color:var(--search);border-color:var(--search)}
.toggle.active.reason{background:var(--reason-soft);color:var(--reason);border-color:var(--reason)}
.toggle .lbl{display:none}
@media (min-width:380px){.toggle .lbl{display:inline}}
.input-row{padding:8px 12px;max-width:780px;margin:0 auto;display:flex;gap:8px;align-items:flex-end}
.attach,.voice{background:transparent;border:1px solid var(--border);color:var(--text-dim);border-radius:50%;width:40px;height:40px;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:all .15s}
.attach:hover,.voice:hover{background:var(--surface);color:var(--text)}
.attach:active,.voice:active{transform:scale(.95)}
.voice.recording{background:var(--danger);color:#fff;border-color:var(--danger);animation:rec-pulse 1.2s ease-in-out infinite}
@keyframes rec-pulse{50%{box-shadow:0 0 0 6px rgba(239,68,68,.3)}}
.input-wrap{flex:1;background:var(--surface);border:1px solid var(--border);border-radius:22px;transition:border-color .15s;display:flex;flex-direction:column;overflow:hidden;min-width:0}
.input-wrap:focus-within{border-color:var(--accent)}
.image-previews{display:flex;gap:6px;padding:8px 10px 0;flex-wrap:wrap}
.image-previews:empty{display:none}
.preview-item{position:relative;width:54px;height:54px;border-radius:8px;overflow:hidden;border:1px solid var(--border)}
.preview-item img{width:100%;height:100%;object-fit:cover}
.preview-item .rm{position:absolute;top:2px;right:2px;background:rgba(0,0,0,.75);color:#fff;border:none;border-radius:50%;width:18px;height:18px;font-size:11px;cursor:pointer;display:flex;align-items:center;justify-content:center;padding:0;line-height:1}
.input{background:transparent;color:var(--text);border:none;padding:11px 16px;font-size:15px;resize:none;max-height:160px;font-family:inherit;outline:none;line-height:1.45;width:100%}
.input::placeholder{color:var(--text-muted)}
.send,.stop{border:none;border-radius:50%;width:40px;height:40px;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:all .15s}
.send{background:var(--user-grad);color:#fff;box-shadow:0 2px 8px rgba(99,102,241,.25)}
.send:disabled{background:#26262b;color:#52525b;box-shadow:none;cursor:not-allowed}
.send:active:not(:disabled){transform:scale(.95)}
.stop{background:var(--danger);color:#fff}
.stop:active{transform:scale(.95)}
.send svg,.stop svg{width:18px;height:18px}

/* Modals */
.modal-back{position:fixed;inset:0;background:rgba(0,0,0,.75);backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);z-index:200;display:none;align-items:center;justify-content:center;padding:20px;animation:fadeIn .2s}
.modal-back.open{display:flex}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
.modal{background:var(--panel);border:1px solid var(--border);border-radius:18px;max-width:480px;width:100%;padding:24px;box-shadow:var(--shadow);max-height:88vh;overflow-y:auto}
.modal h2{margin:0 0 6px;font-size:22px;font-weight:700;letter-spacing:-.02em;background:var(--user-grad);-webkit-background-clip:text;background-clip:text;color:transparent}
.modal .sub{color:var(--text-dim);font-size:14px;margin-bottom:16px}
.modal label{display:block;font-size:12px;font-weight:600;color:var(--text-dim);margin:14px 0 6px;text-transform:uppercase;letter-spacing:.04em}
.modal input,.modal textarea,.modal select{width:100%;background:var(--surface);color:var(--text);border:1px solid var(--border);border-radius:10px;padding:10px 14px;font-size:14px;outline:none}
.modal input:focus,.modal textarea:focus,.modal select:focus{border-color:var(--accent)}
.modal .actions{display:flex;gap:8px;margin-top:16px}
.btn{padding:10px 16px;border-radius:10px;font-size:14px;font-weight:600;cursor:pointer;border:none;transition:all .15s;flex:1}
.btn-primary{background:var(--user-grad);color:#fff}
.btn-primary:active{transform:scale(.98);filter:brightness(.95)}
.btn-ghost{background:transparent;color:var(--text-dim);border:1px solid var(--border)}
.btn-ghost:hover{background:var(--surface);color:var(--text)}
.divider{height:1px;background:var(--border);margin:18px -24px}
.key-display{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px;font-family:"JetBrains Mono",monospace;font-size:15px;text-align:center;letter-spacing:.05em;margin:8px 0;color:#a5b4fc;word-break:break-all}
.tabs{display:flex;gap:4px;background:var(--surface);padding:4px;border-radius:10px;margin-bottom:16px}
.tab{flex:1;background:transparent;border:none;color:var(--text-dim);padding:8px;border-radius:7px;cursor:pointer;font-weight:500;font-size:13px;transition:all .12s}
.tab.active{background:var(--panel);color:var(--text)}
.mem-list{display:flex;flex-direction:column;gap:6px;max-height:260px;overflow-y:auto;padding-right:4px}
.mem-item{display:flex;align-items:center;gap:8px;background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:8px 12px;font-size:13px}
.mem-item .text{flex:1;color:var(--text-dim)}
.mem-item .rm{background:transparent;border:none;color:var(--text-muted);cursor:pointer;font-size:14px;padding:2px 6px;border-radius:4px}
.mem-item .rm:hover{color:var(--danger)}
.mem-empty{text-align:center;color:var(--text-muted);font-size:13px;padding:24px;font-style:italic}

.model-grid{display:grid;gap:8px}
.model-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:10px 12px;cursor:pointer;text-align:left;transition:all .15s;display:flex;align-items:flex-start;gap:10px}
.model-card:hover{background:var(--surface-2);border-color:var(--border-strong)}
.model-card.selected{border-color:var(--accent);background:var(--accent-soft)}
.model-card .info{flex:1;min-width:0}
.model-card .name{font-weight:600;color:var(--text);font-size:14px;margin-bottom:2px}
.model-card .blurb{font-size:12px;color:var(--text-dim);line-height:1.4}
.model-card .check{color:var(--accent);font-size:18px;line-height:1;opacity:0}
.model-card.selected .check{opacity:1}

.toast{position:fixed;top:max(16px,env(safe-area-inset-top));left:50%;transform:translateX(-50%) translateY(-20px);background:var(--panel);border:1px solid var(--border-strong);color:var(--text);padding:10px 18px;border-radius:10px;font-size:13px;z-index:300;opacity:0;pointer-events:none;transition:opacity .2s,transform .2s;box-shadow:var(--shadow);max-width:90vw}
.toast.show{opacity:1;transform:translateX(-50%) translateY(0)}

.copy-btn{position:absolute;top:6px;right:6px;background:rgba(255,255,255,.08);border:1px solid var(--border);color:var(--text-dim);padding:3px 8px;border-radius:6px;font-size:11px;cursor:pointer;opacity:.6;transition:opacity .15s}
.bubble pre:hover .copy-btn{opacity:1}

.lightbox{position:fixed;inset:0;background:rgba(0,0,0,.95);z-index:400;display:none;align-items:center;justify-content:center;padding:20px;cursor:zoom-out}
.lightbox.open{display:flex}
.lightbox img{max-width:100%;max-height:100%;border-radius:8px}
</style>
</head>
<body>

<div class="header">
  <button class="icon-btn" id="menuBtn" aria-label="menu">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
  </button>
  <div class="title-wrap">
    <div class="title" id="chatTitle">Router</div>
    <div class="model-badge"><span class="dot"></span><span id="modelLabel">Venice (Uncensored)</span></div>
  </div>
  <button class="icon-btn" id="settingsBtn" aria-label="settings">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
  </button>
</div>

<div class="backdrop" id="backdrop"></div>
<aside class="sidebar" id="sidebar">
  <div class="sb-head">
    <button class="new-chat" id="newBtn">
      <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
      New chat
    </button>
    <input class="search-box" id="searchBox" placeholder="Search conversations…">
  </div>
  <div class="chats-list" id="chatsList"></div>
</aside>

<div class="messages" id="messages"><div class="messages-inner" id="messagesInner"></div></div>

<div class="input-bar">
  <div class="toggles">
    <button class="toggle search" id="searchToggle" title="Web search for next message">
      🔍 <span class="lbl">Search</span>
    </button>
    <button class="toggle reason" id="reasonToggle" title="Use DeepSeek R1 for next message">
      🧠 <span class="lbl">Deep</span>
    </button>
  </div>
  <div class="input-row">
    <button class="attach" id="attachBtn" aria-label="attach image">
      <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
    </button>
    <input type="file" id="fileInput" accept="image/*" multiple style="display:none">
    <div class="input-wrap">
      <div class="image-previews" id="imagePreviews"></div>
      <textarea class="input" id="input" placeholder="Ask anything…" rows="1"></textarea>
    </div>
    <button class="voice" id="voiceBtn" aria-label="voice input">
      <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
    </button>
    <button class="send" id="sendBtn" aria-label="send">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/></svg>
    </button>
    <button class="stop" id="stopBtn" aria-label="stop" style="display:none">
      <svg viewBox="0 0 24 24" fill="currentColor"><rect x="7" y="7" width="10" height="10" rx="2"/></svg>
    </button>
  </div>
</div>

<!-- Welcome -->
<div class="modal-back" id="welcomeModal">
  <div class="modal">
    <h2>Welcome.</h2>
    <p class="sub">Your chats live in the cloud, tied to a personal sync key. Use the same key on any device.</p>
    <div class="tabs">
      <button class="tab active" data-tab="new">First time</button>
      <button class="tab" data-tab="have">I have a key</button>
    </div>
    <div id="tab-new">
      <label>Your new sync key</label>
      <div class="key-display" id="newKey"></div>
      <p class="sub" style="font-size:12px;margin-top:8px">Save this somewhere safe — a password manager, a note. It's the only way back to your chats on another device.</p>
      <div class="actions">
        <button class="btn btn-ghost" id="regenKey">Regenerate</button>
        <button class="btn btn-primary" id="useNewKey">I saved it, continue</button>
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

<!-- Settings -->
<div class="modal-back" id="settingsModal">
  <div class="modal">
    <h2>Settings</h2>
    <p class="sub">Models, memories, sync, exports.</p>

    <label>Default lead model</label>
    <div class="model-grid" id="modelGrid"></div>

    <div class="divider"></div>

    <label>Sync key (use on other devices)</label>
    <div class="key-display" id="myKey"></div>
    <div class="actions" style="margin-top:8px">
      <button class="btn btn-ghost" id="copyKey">Copy key</button>
    </div>

    <div class="divider"></div>

    <label>What the leader remembers about you</label>
    <div class="mem-list" id="memList"></div>
    <div class="actions" style="margin-top:12px">
      <button class="btn btn-ghost" id="clearMem">Clear all</button>
      <button class="btn btn-ghost" id="exportAll">Export chats</button>
    </div>

    <div class="actions" style="margin-top:16px">
      <button class="btn btn-primary" id="closeSettings">Done</button>
    </div>
  </div>
</div>

<div class="toast" id="toast"></div>
<div class="lightbox" id="lightbox"><img id="lightboxImg" alt=""></div>

<script>
const $ = (id) => document.getElementById(id);
const SK_STORE = "router_sync_key_v1";
const PREFS_STORE = "router_prefs_v1";

let toastTimer;
function toast(msg){const t=$("toast");t.textContent=msg;t.classList.add("show");clearTimeout(toastTimer);toastTimer=setTimeout(()=>t.classList.remove("show"),2400)}

const WORDS=["amber","arrow","aspen","autumn","azure","basil","birch","blade","blaze","bloom","brave","brick","bridge","brook","calm","canyon","cedar","cherry","clay","clear","cloud","clover","coast","comet","copper","coral","cosmos","crane","creek","crimson","crystal","cypress","dawn","deep","desert","drift","dune","dusk","ember","emerald","fable","falcon","feather","fern","flame","flint","forest","frost","glade","golden","granite","grove","harbor","harmony","haven","hazel","heron","horizon","indigo","ivory","jade","jasper","juniper","keen","lake","lantern","laurel","leaf","light","linen","lupine","maple","marble","meadow","midnight","mist","moon","moss","mountain","north","oak","ocean","olive","onyx","opal","orchid","otter","pearl","pebble","petal","pine","plum","poppy","prairie","quartz","quiet","quill","rain","rapid","raven","reed","ridge","ripple","river","robin","rose","rust","saffron","sage","sapphire","scarlet","shade","shadow","shore","silver","sky","slate","snow","sparrow","spring","spruce","star","stone","storm","stream","summer","sunset","swift","teal","thicket","thorn","thunder","tide","tiger","topaz","trail","tundra","valley","velvet","violet","vista","walnut","wave","wheat","whisper","willow","wind","winter","wren"];
function genKey(){const p=()=>WORDS[Math.floor(Math.random()*WORDS.length)];return p()+"-"+p()+"-"+p()+"-"+p()}

const MODELS_CLIENT = {
  venice: {label:"Venice (Uncensored)", blurb:"No content filter. Default. Best for chat, creative work, candid talk."},
  llama:  {label:"Llama 3.3 70B",       blurb:"Solid general-purpose. Strong reasoning. Some content filtering."},
  r1:     {label:"DeepSeek R1 (Reasoning)", blurb:"Deep thinker. Slower but careful. Also engaged by Deep toggle."},
};

marked.setOptions({breaks:true,gfm:true});
const renderMarkdown = (t) => DOMPurify.sanitize(marked.parse(t || ""), {ADD_ATTR:["target"]});
function addCopyButtons(c) {
  c.querySelectorAll("pre").forEach((pre) => {
    if (pre.querySelector(".copy-btn")) return;
    const b = document.createElement("button");
    b.className = "copy-btn"; b.textContent = "copy";
    b.onclick = (e) => {
      e.stopPropagation();
      const code = pre.querySelector("code")?.innerText || pre.innerText;
      navigator.clipboard.writeText(code).then(() => {
        b.textContent = "copied";
        setTimeout(() => b.textContent = "copy", 1200);
      });
    };
    pre.appendChild(b);
  });
}

let syncKey = localStorage.getItem(SK_STORE) || "";
let prefs = JSON.parse(localStorage.getItem(PREFS_STORE) || "{}");
if (!prefs.model) prefs.model = "venice";
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

const uuid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 8);

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error("HTTP " + r.status);
  return r.json();
}
const apiList = () => api("/chats?sk=" + encodeURIComponent(syncKey));
const apiGet = (id) => api("/chats/" + id + "?sk=" + encodeURIComponent(syncKey));
const apiSave = (c) => api("/chats/save", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({sync_key: syncKey, chat: c})});
const apiDel = (id) => api("/chats/" + id + "?sk=" + encodeURIComponent(syncKey), {method:"DELETE"});
const apiPin = (id, pinned) => api("/chats/" + id + "/pin", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({sync_key: syncKey, pinned})});
const apiMemGet = () => api("/memories?sk=" + encodeURIComponent(syncKey));
const apiMemSave = (m) => api("/memories/save", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({sync_key: syncKey, memories: m})});

function timeOfDay() {
  const h = new Date().getHours();
  if (h < 5) return "still up?";
  if (h < 12) return "morning";
  if (h < 17) return "afternoon";
  if (h < 21) return "evening";
  return "late one";
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

const SUGGESTIONS = [
  "What's something interesting in the news today?",
  "Help me think through a decision",
  "Explain something I'm confused about",
  "Brainstorm with me for a project",
  "Write me a short story about something unexpected",
];

function renderEmpty() {
  const root = $("messagesInner");
  root.innerHTML = "";
  const e = document.createElement("div");
  e.className = "empty";
  const greet = document.createElement("div");
  greet.className = "greet";
  greet.textContent = "Good " + timeOfDay() + (userMemoryName ? ", " + userMemoryName : "") + ".";
  const sub = document.createElement("div");
  sub.className = "sub";
  sub.textContent = "Ask anything. Tap 🔍 to search, 🧠 for deep reasoning, paperclip for images.";
  const sg = document.createElement("div");
  sg.className = "suggestions";
  const picked = [...SUGGESTIONS].sort(() => Math.random() - 0.5).slice(0, 4);
  for (const text of picked) {
    const b = document.createElement("button");
    b.className = "sugg";
    b.textContent = text;
    b.onclick = () => { $("input").value = text; $("input").focus(); autoresize(); };
    sg.appendChild(b);
  }
  e.appendChild(greet);
  e.appendChild(sub);
  e.appendChild(sg);
  root.appendChild(e);
}

function makeBubble(role, content, images, meta, isStreamingFlag) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + role;
  const b = document.createElement("div");
  b.className = "bubble" + (isStreamingFlag ? " empty-streaming" : "");

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
      b.innerHTML += '<div class="thinking"><span></span><span></span><span></span></div>';
    } else {
      const div = document.createElement("div");
      div.innerHTML = renderMarkdown(content);
      addCopyButtons(div);
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
  if (meta.tool_used === "web_search") s += ' <span class="badge b-search">searched</span>';
  if (meta.tool_used === "ask_reasoner") s += ' <span class="badge b-reason">→ reasoner</span>';
  return s;
}

function renderAll() {
  const root = $("messagesInner");
  if (currentMessages.length === 0) { renderEmpty(); return; }
  root.innerHTML = "";
  currentMessages.forEach((m, idx) => {
    const { wrap } = makeBubble(m.role, m.content, m.images, m.meta, false);
    const isLast = idx === currentMessages.length - 1;
    if (isLast && m.role === "user" && !isStreaming) {
      const actions = document.createElement("div");
      actions.className = "msg-actions";
      const edit = document.createElement("button");
      edit.textContent = "✎ Edit";
      edit.onclick = () => editMessage(idx);
      actions.appendChild(edit);
      wrap.appendChild(actions);
    } else if (isLast && m.role === "assistant" && !isStreaming) {
      const actions = document.createElement("div");
      actions.className = "msg-actions";
      const regen = document.createElement("button");
      regen.textContent = "↻ Regenerate";
      regen.onclick = regenerate;
      const copy = document.createElement("button");
      copy.textContent = "📋 Copy";
      copy.onclick = () => { navigator.clipboard.writeText(m.content || ""); toast("Copied"); };
      actions.appendChild(regen);
      actions.appendChild(copy);
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
      e.className = "mem-empty";
      e.textContent = f ? "No matches" : "No chats yet";
      e.style.padding = "20px 12px";
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
      pinBtn.textContent = c.pinned ? "📌" : "📍";
      pinBtn.title = c.pinned ? "Unpin" : "Pin";
      pinBtn.onclick = async (e) => {
        e.stopPropagation();
        await apiPin(c.id, !c.pinned);
        renderList(filter);
      };
      const delBtn = document.createElement("button");
      delBtn.className = "del";
      delBtn.textContent = "🗑";
      delBtn.onclick = async (e) => {
        e.stopPropagation();
        if (confirm("Delete this chat?")) {
          await apiDel(c.id);
          if (c.id === currentChatId) newChat();
          renderList(filter);
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
  } catch (e) { toast("Failed to load chat"); }
}

async function persist() {
  if (!currentChatId) currentChatId = uuid();
  const title = currentMessages.find(m => m.role === "user")?.content?.slice(0, 50) || "New chat";
  await apiSave({
    id: currentChatId,
    title,
    messages: currentMessages,
    updated: Date.now(),
    pinned: false,
  });
  $("chatTitle").textContent = title;
  renderList();
}

function openSB() { $("sidebar").classList.add("open"); $("backdrop").classList.add("open"); }
function closeSB() { $("sidebar").classList.remove("open"); $("backdrop").classList.remove("open"); }

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
    rm.onclick = () => { pendingImages.splice(idx, 1); renderPreviews(); };
    item.appendChild(i);
    item.appendChild(rm);
    root.appendChild(item);
  });
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

$("attachBtn").onclick = () => $("fileInput").click();
$("fileInput").onchange = async (e) => {
  for (const f of e.target.files) {
    if (f.size > 4 * 1024 * 1024) { toast("Image too large (max 4MB)"); continue; }
    pendingImages.push(await fileToImage(f));
  }
  renderPreviews();
  e.target.value = "";
};

// Toggle handlers
function updateToggles() {
  $("searchToggle").classList.toggle("active", useSearchNext);
  $("reasonToggle").classList.toggle("active", useReasonNext);
}
$("searchToggle").onclick = () => { useSearchNext = !useSearchNext; updateToggles(); };
$("reasonToggle").onclick = () => { useReasonNext = !useReasonNext; updateToggles(); };

// Voice input
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
  if (!recog) { toast("Voice not supported in this browser"); return; }
  if ($("voiceBtn").classList.contains("recording")) recog.stop();
  else { try { recog.start(); } catch (e) { toast("Already listening"); } }
};

function autoresize() {
  const e = $("input");
  e.style.height = "auto";
  e.style.height = Math.min(e.scrollHeight, 160) + "px";
}

async function regenerate() {
  if (isStreaming) return;
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
  renderAll();
  autoresize();
  $("input").focus();
}

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
  input.value = "";
  autoresize();

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

  // Render the user message and the streaming assistant bubble
  if ($("messagesInner").querySelector(".empty")) $("messagesInner").innerHTML = "";

  // Rebuild from scratch — simpler and avoids stale DOM
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
      aMsg.content = "Error: HTTP " + resp.status;
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
          toolEl.textContent = "ℹ " + p.msg;
          aWrap.insertBefore(toolEl, aBub);
          setTimeout(() => { if (toolEl) { toolEl.remove(); toolEl = null; } }, 2500);
        } else if (et === "token") {
          aMsg.content += p.text;
          aBub.innerHTML = renderMarkdown(aMsg.content);
          addCopyButtons(aBub);
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
          if (!aMsg.content) aBub.innerHTML = '<div class="thinking"><span></span><span></span><span></span></div>';
        } else if (et === "done") {
          Object.assign(aMsg.meta, p);
        } else if (et === "aborted") {
          aMsg.content += (aMsg.content ? "\\n\\n" : "") + "_[stopped]_";
          aBub.innerHTML = renderMarkdown(aMsg.content);
        } else if (et === "error") {
          aMsg.content += (aMsg.content ? "\\n\\n" : "") + "⚠ " + p.message;
          aBub.innerHTML = renderMarkdown(aMsg.content);
        }
      }
    }

    if (toolEl) toolEl.remove();
    if (!aMsg.content) {
      aMsg.content = "_(empty response — try again or switch models)_";
    }
    aBub.innerHTML = renderMarkdown(aMsg.content);
    addCopyButtons(aBub);
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
    } else {
      aMsg.content = "Error: " + e.message;
      aBub.innerHTML = renderMarkdown(aMsg.content);
    }
  } finally {
    isStreaming = false;
    currentAbort = null;
    $("sendBtn").style.display = "flex";
    $("stopBtn").style.display = "none";
    $("voiceBtn").disabled = false;
    // Reset toggles after each send
    useSearchNext = false;
    useReasonNext = false;
    updateToggles();
    scrollBottom();
    renderAll();
  }
}

$("stopBtn").onclick = () => { if (currentAbort) currentAbort.abort(); };

// Welcome / sync
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
  toast("Welcome ☆ key saved");
  await renderList();
  renderAll();
};
$("useExistingKey").onclick = async () => {
  const v = $("existingKey").value.trim();
  if (v.length < 4) { toast("That doesn't look right"); return; }
  syncKey = v;
  localStorage.setItem(SK_STORE, syncKey);
  $("welcomeModal").classList.remove("open");
  toast("Synced.");
  await renderList();
  await extractNameFromMem();
  renderAll();
};

// Settings
function buildModelGrid() {
  const grid = $("modelGrid");
  grid.innerHTML = "";
  for (const [key, info] of Object.entries(MODELS_CLIENT)) {
    const card = document.createElement("button");
    card.className = "model-card" + (prefs.model === key ? " selected" : "");
    card.innerHTML = '<div class="info"><div class="name">' + info.label + '</div><div class="blurb">' + info.blurb + '</div></div><div class="check">✓</div>';
    card.onclick = () => {
      prefs.model = key;
      savePrefs();
      buildModelGrid();
      toast("Default: " + info.label);
    };
    grid.appendChild(card);
  }
}

async function openSettings() {
  $("myKey").textContent = syncKey;
  buildModelGrid();
  const mems = await apiMemGet().catch(() => []);
  const list = $("memList");
  list.innerHTML = "";
  if (mems.length === 0) {
    const e = document.createElement("div");
    e.className = "mem-empty";
    e.textContent = "The leader hasn't learned anything about you yet. Have a few conversations and check back.";
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
      rm.onclick = async () => {
        const f = mems.filter(x => x !== m);
        await apiMemSave(f);
        openSettings();
        await extractNameFromMem();
      };
      it.appendChild(t);
      it.appendChild(rm);
      list.appendChild(it);
    }
  }
  $("settingsModal").classList.add("open");
}

$("copyKey").onclick = () => { navigator.clipboard.writeText(syncKey); toast("Key copied"); };
$("clearMem").onclick = async () => {
  if (!confirm("Forget everything the leader has learned about you?")) return;
  await apiMemSave([]);
  userMemoryName = null;
  openSettings();
  toast("Cleared");
};
$("exportAll").onclick = async () => {
  const chats = await apiList();
  const full = [];
  for (const c of chats) full.push(await apiGet(c.id));
  const blob = new Blob([JSON.stringify({exported: new Date().toISOString(), chats: full}, null, 2)], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "router-chats-" + Date.now() + ".json";
  a.click();
};
$("closeSettings").onclick = () => $("settingsModal").classList.remove("open");
$("settingsBtn").onclick = openSettings;

// Wire input
$("sendBtn").onclick = send;
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});
$("input").addEventListener("input", autoresize);
$("menuBtn").onclick = openSB;
$("backdrop").onclick = () => {
  closeSB();
  $("welcomeModal").classList.remove("open");
  $("settingsModal").classList.remove("open");
};
$("newBtn").onclick = newChat;
$("searchBox").oninput = (e) => renderList(e.target.value);
$("lightbox").onclick = () => $("lightbox").classList.remove("open");

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

  // Static
  if (p === "/" && req.method === "GET") {
    return new Response(HTML, { headers: { "Content-Type": "text/html; charset=utf-8" } });
  }
  if (p === "/manifest.json") {
    return new Response(MANIFEST, { headers: { "Content-Type": "application/json" } });
  }
  if (p === "/icon.svg") {
    return new Response(ICON_SVG, { headers: { "Content-Type": "image/svg+xml" } });
  }

  // API
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

  // Diagnostics
  if (p === "/health") {
    return Response.json({
      ok: true,
      version: "v4",
      openrouter_key: !!OPENROUTER_KEY,
      tavily_key: !!TAVILY_KEY,
      models: Object.fromEntries(
        Object.entries(MODELS).map(([k, v]) => [k, v.id])
      ),
    });
  }

  return new Response("Not found", { status: 404 });
});
