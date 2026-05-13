// ============================================================
// PERSONAL AI ROUTER  v2
//
// v2 adds:
//   - Cloud sync via Deno KV + personal sync key (no login, no email)
//   - Cross-chat memory (leader remembers you across all conversations)
//   - Image upload (Gemini is multimodal — it actually sees them)
//   - Markdown rendering, code blocks, refined visual design
//   - Settings drawer with memories editor + sync key viewer
//
// Required env vars in Deno Deploy:
//   GOOGLE_AI_KEY, OPENROUTER_KEY, TAVILY_KEY
// ============================================================

const GOOGLE_AI_KEY  = Deno.env.get("GOOGLE_AI_KEY")  || "";
const OPENROUTER_KEY = Deno.env.get("OPENROUTER_KEY") || "";
const TAVILY_KEY     = Deno.env.get("TAVILY_KEY")     || "";

const LEAD_MODEL     = "gemini-2.5-flash";
const REASONER_MODEL = "deepseek/deepseek-r1:free";

const kv = await Deno.openKv();

async function hashKey(syncKey: string): Promise<string> {
  const enc = new TextEncoder().encode(syncKey.trim().toLowerCase());
  const buf = await crypto.subtle.digest("SHA-256", enc);
  return Array.from(new Uint8Array(buf))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("")
    .slice(0, 32);
}

async function listChats(userHash: string) {
  const out: any[] = [];
  for await (const e of kv.list({ prefix: ["u", userHash, "chats"] })) out.push(e.value);
  return out.sort((a, b) => (b.updated || 0) - (a.updated || 0));
}
async function getChatStored(userHash: string, id: string) {
  const r = await kv.get(["u", userHash, "chats", id]);
  return r.value;
}
async function saveChatStored(userHash: string, chat: any) {
  await kv.set(["u", userHash, "chats", chat.id], chat);
}
async function deleteChatStored(userHash: string, id: string) {
  await kv.delete(["u", userHash, "chats", id]);
}
async function getMemories(userHash: string): Promise<string[]> {
  const r = await kv.get<string[]>(["u", userHash, "mem"]);
  return r.value || [];
}
async function saveMemories(userHash: string, mems: string[]) {
  const unique = Array.from(new Set(mems.map((m) => m.trim()).filter(Boolean))).slice(-120);
  await kv.set(["u", userHash, "mem"], unique);
}

function timeGreeting(): string {
  const h = new Date().getUTCHours();
  if (h < 11) return "morning";
  if (h < 17) return "afternoon";
  return "evening";
}

function buildSystemPrompt(memories: string[]): string {
  const memBlock = memories.length
    ? `\n\n# What you know about this person (from past conversations)\n${memories.map((m) => `- ${m}`).join("\n")}\n\nWeave this in naturally only when relevant. Don't recite it. Don't announce that you "remember." Just be a person who already knows them.`
    : "";

  return `You are the lead of a personal AI assistant built for one specific person. You see every question first and answer directly when you can.

# Voice
Warm, present, a touch dry. Not overeager, not robotic. Brief by default; longer when the question earns it. You can be playful but never performative. Speak like a sharp friend, not a help desk.

# Tools
- web_search: anything current, recent, news, specific facts you might miss, time-sensitive. Use without hesitation when needed.
- ask_reasoner: delegate to a deep-thinking specialist (DeepSeek R1) for hard multi-step problems — proofs, intricate logic, complex architecture. Don't use for casual questions.

# Style
- Format with markdown when it helps (code blocks with language tags, lists, bold). Plain prose otherwise.
- Never start replies with "Sure!", "Certainly!", "Of course!", or similar filler.
- Don't ask permission to answer; just answer.

Today is ${new Date().toISOString().slice(0, 10)} (${timeGreeting()} UTC).${memBlock}`;
}

async function tavilySearch(query: string): Promise<string> {
  try {
    const r = await fetch("https://api.tavily.com/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        api_key: TAVILY_KEY, query,
        max_results: 5, include_answer: true, search_depth: "basic",
      }),
    });
    if (!r.ok) return `Search failed (HTTP ${r.status})`;
    const d = await r.json();
    let out = d.answer ? `Summary: ${d.answer}\n\n` : "";
    if (Array.isArray(d.results)) {
      out += "Sources:\n";
      for (const x of d.results) {
        out += `- ${x.title}\n  ${x.url}\n  ${(x.content || "").slice(0, 220)}\n`;
      }
    }
    return out || "(no results)";
  } catch (e) { return `Search error: ${e}`; }
}

async function callOpenRouter(model: string, messages: any[]): Promise<string> {
  try {
    const r = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${OPENROUTER_KEY}`,
        "Content-Type": "application/json",
        "HTTP-Referer": "https://deno.dev",
        "X-Title": "Personal Router",
      },
      body: JSON.stringify({ model, messages }),
    });
    if (!r.ok) return `Reasoner error ${r.status}: ${(await r.text()).slice(0, 200)}`;
    const d = await r.json();
    return d.choices?.[0]?.message?.content || "(reasoner returned nothing)";
  } catch (e) { return `Reasoner exception: ${e}`; }
}

function buildGeminiContents(messages: any[]) {
  return messages.map((m) => {
    const parts: any[] = [];
    if (m.content) parts.push({ text: m.content });
    if (Array.isArray(m.images)) {
      for (const img of m.images) {
        if (img && img.data && img.mimeType) {
          parts.push({ inlineData: { mimeType: img.mimeType, data: img.data } });
        }
      }
    }
    if (parts.length === 0) parts.push({ text: "" });
    return { role: m.role === "assistant" ? "model" : "user", parts };
  });
}

const sseChunk = (event: string, data: any): Uint8Array =>
  new TextEncoder().encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);

const TOOLS = [
  {
    name: "web_search",
    description: "Search the live web for current info, news, recent events, or facts you don't know. Be specific in the query.",
    parameters: {
      type: "OBJECT",
      properties: { query: { type: "STRING", description: "Concise search query, 2-6 words." } },
      required: ["query"],
    },
  },
  {
    name: "ask_reasoner",
    description: "Delegate to DeepSeek R1 for deep multi-step reasoning: proofs, complex math, intricate logic, deep code analysis. Only when genuine deep thinking is needed.",
    parameters: {
      type: "OBJECT",
      properties: { question: { type: "STRING", description: "Full problem with all context the reasoner needs." } },
      required: ["question"],
    },
  },
];

async function runAgent(controller: ReadableStreamDefaultController, messages: any[], systemPrompt: string) {
  const send = (event: string, data: any) => controller.enqueue(sseChunk(event, data));
  send("status", { leader: LEAD_MODEL });

  const contents = buildGeminiContents(messages);
  let responder = LEAD_MODEL;
  let toolUsed: string | null = null;

  for (let iter = 0; iter < 4; iter++) {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${LEAD_MODEL}:streamGenerateContent?alt=sse&key=${GOOGLE_AI_KEY}`;
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        systemInstruction: { parts: [{ text: systemPrompt }] },
        contents,
        tools: [{ functionDeclarations: TOOLS }],
        generationConfig: { temperature: 0.7 },
      }),
    });

    if (!r.ok || !r.body) {
      const t = r.body ? "" : await r.text().catch(() => "");
      send("error", { message: `Lead HTTP ${r.status}: ${t.slice(0, 200)}` });
      return;
    }

    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let pendingFn: any = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        let j: any;
        try { j = JSON.parse(line.slice(6)); } catch { continue; }
        const parts = j.candidates?.[0]?.content?.parts || [];
        for (const p of parts) {
          if (p.text) send("token", { text: p.text });
          if (p.functionCall) pendingFn = p.functionCall;
        }
      }
    }

    if (!pendingFn) { send("done", { responder, tool_used: toolUsed }); return; }

    const { name, args } = pendingFn;
    toolUsed = name;
    send("tool_start", { tool: name, args });
    let toolResult = "";
    if (name === "web_search") {
      toolResult = await tavilySearch(args.query);
    } else if (name === "ask_reasoner") {
      send("delegate", { to: "deepseek-r1" });
      responder = "deepseek-r1 (synthesized)";
      toolResult = await callOpenRouter(REASONER_MODEL, [{ role: "user", content: args.question }]);
    } else {
      toolResult = `Unknown tool: ${name}`;
    }
    send("tool_done", { tool: name, preview: toolResult.slice(0, 200) });

    contents.push({ role: "model", parts: [{ functionCall: pendingFn }] });
    contents.push({ role: "user", parts: [{ functionResponse: { name, response: { result: toolResult } } }] });
  }
  send("done", { responder, tool_used: toolUsed, note: "max iterations" });
}

// Memory extraction — runs after each exchange, doesn't block UI
async function extractMemories(userHash: string, userText: string, assistantText: string, existing: string[]) {
  try {
    const prompt = `Extract durable facts about the user from this exchange. Only facts useful in future conversations: their name, ongoing projects, preferences, recurring interests, location, work, relationships, skills, opinions they hold. Skip anything transient. Skip anything already in the list.

Existing memories:
${existing.length ? existing.map((m) => `- ${m}`).join("\n") : "(none yet)"}

Recent exchange:
USER: ${userText.slice(0, 2000)}
ASSISTANT: ${assistantText.slice(0, 2000)}

Output ONLY a JSON array of new short memory strings, like ["User's name is Sam", "User is learning Spanish"]. Empty array [] if nothing memorable.`;

    const url = `https://generativelanguage.googleapis.com/v1beta/models/${LEAD_MODEL}:generateContent?key=${GOOGLE_AI_KEY}`;
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: [{ role: "user", parts: [{ text: prompt }] }],
        generationConfig: { temperature: 0.1, responseMimeType: "application/json" },
      }),
    });
    if (!r.ok) return;
    const d = await r.json();
    const text = d.candidates?.[0]?.content?.parts?.[0]?.text || "[]";
    let parsed: any;
    try { parsed = JSON.parse(text); } catch { return; }
    if (!Array.isArray(parsed) || parsed.length === 0) return;
    const cleaned = parsed.filter((x) => typeof x === "string" && x.length > 3 && x.length < 200);
    if (cleaned.length === 0) return;
    await saveMemories(userHash, [...existing, ...cleaned]);
  } catch { /* swallow */ }
}

async function handleChat(req: Request): Promise<Response> {
  const body = await req.json();
  const { messages, sync_key } = body;
  if (!sync_key || typeof sync_key !== "string") {
    return new Response("Missing sync_key", { status: 400 });
  }
  const userHash = await hashKey(sync_key);
  const memories = await getMemories(userHash);
  const systemPrompt = buildSystemPrompt(memories);

  const stream = new ReadableStream({
    async start(controller) {
      try { await runAgent(controller, messages, systemPrompt); }
      catch (e) { controller.enqueue(sseChunk("error", { message: String(e) })); }
      finally { controller.close(); }
    },
  });
  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}

async function handleChatsList(req: Request): Promise<Response> {
  const sk = new URL(req.url).searchParams.get("sk") || "";
  if (!sk) return new Response("missing sk", { status: 400 });
  const userHash = await hashKey(sk);
  const chats = await listChats(userHash);
  const summaries = chats.map((c: any) => ({
    id: c.id, title: c.title, updated: c.updated,
    preview: (c.messages?.[c.messages.length - 1]?.content || "").slice(0, 100),
  }));
  return Response.json(summaries);
}
async function handleChatGet(req: Request, id: string): Promise<Response> {
  const sk = new URL(req.url).searchParams.get("sk") || "";
  if (!sk) return new Response("missing sk", { status: 400 });
  const userHash = await hashKey(sk);
  const chat = await getChatStored(userHash, id);
  if (!chat) return new Response("not found", { status: 404 });
  return Response.json(chat);
}
async function handleChatSave(req: Request): Promise<Response> {
  const { sync_key, chat } = await req.json();
  if (!sync_key || !chat?.id) return new Response("bad payload", { status: 400 });
  const userHash = await hashKey(sync_key);
  await saveChatStored(userHash, chat);
  const msgs = chat.messages || [];
  const lastUser = [...msgs].reverse().find((m: any) => m.role === "user");
  const lastAsst = [...msgs].reverse().find((m: any) => m.role === "assistant");
  if (lastUser?.content && lastAsst?.content) {
    const existing = await getMemories(userHash);
    extractMemories(userHash, lastUser.content, lastAsst.content, existing);
  }
  return Response.json({ ok: true });
}
async function handleChatDelete(req: Request, id: string): Promise<Response> {
  const sk = new URL(req.url).searchParams.get("sk") || "";
  if (!sk) return new Response("missing sk", { status: 400 });
  const userHash = await hashKey(sk);
  await deleteChatStored(userHash, id);
  return Response.json({ ok: true });
}
async function handleMemGet(req: Request): Promise<Response> {
  const sk = new URL(req.url).searchParams.get("sk") || "";
  if (!sk) return new Response("missing sk", { status: 400 });
  return Response.json(await getMemories(await hashKey(sk)));
}
async function handleMemSave(req: Request): Promise<Response> {
  const { sync_key, memories } = await req.json();
  if (!sync_key || !Array.isArray(memories)) return new Response("bad payload", { status: 400 });
  await saveMemories(await hashKey(sync_key), memories);
  return Response.json({ ok: true });
}

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
  --bg:#09090b; --panel:#121214; --surface:#17171a; --border:#26262b; --border-strong:#3a3a42;
  --text:#ededf0; --text-dim:#a1a1aa; --text-muted:#71717a;
  --accent:#6366f1; --search:#f59e0b; --reason:#a855f7; --danger:#ef4444;
  --user-grad:linear-gradient(135deg,#6366f1 0%,#8b5cf6 100%);
  --shadow:0 8px 32px rgba(0,0,0,0.4);
}
*,*::before,*::after{box-sizing:border-box}
html,body{margin:0;padding:0;height:100%;background:var(--bg);color:var(--text);font-family:"Inter",-apple-system,BlinkMacSystemFont,system-ui,sans-serif;-webkit-font-smoothing:antialiased;overflow:hidden;font-size:15px;line-height:1.5}
body{display:flex;flex-direction:column}
button{font-family:inherit}
.header{display:flex;align-items:center;padding:12px 14px;border-bottom:1px solid var(--border);flex-shrink:0;gap:4px;padding-top:max(12px,env(safe-area-inset-top));background:var(--bg);position:relative;z-index:5}
.icon-btn{background:transparent;border:none;color:var(--text-dim);cursor:pointer;padding:8px;border-radius:10px;line-height:1;display:flex;align-items:center;justify-content:center;width:38px;height:38px;transition:background .15s,color .15s}
.icon-btn:hover{color:var(--text)}
.icon-btn:active{background:var(--surface);transform:scale(.95)}
.icon-btn svg{width:20px;height:20px}
.title{font-size:14px;color:var(--text);flex:1;text-align:center;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:0 8px;letter-spacing:-0.01em}
.sidebar{position:fixed;top:0;left:0;bottom:0;width:300px;max-width:85vw;background:var(--panel);transform:translateX(-100%);transition:transform .25s cubic-bezier(.4,0,.2,1);z-index:100;display:flex;flex-direction:column;border-right:1px solid var(--border);padding-top:env(safe-area-inset-top);padding-bottom:env(safe-area-inset-bottom);box-shadow:var(--shadow)}
.sidebar.open{transform:translateX(0)}
.sb-head{padding:14px;border-bottom:1px solid var(--border)}
.new-chat{width:100%;background:var(--user-grad);color:#fff;border:none;padding:11px 14px;border-radius:10px;font-size:14px;cursor:pointer;font-weight:600;display:flex;align-items:center;justify-content:center;gap:8px;transition:transform .1s,filter .15s}
.new-chat:active{transform:scale(.98);filter:brightness(.95)}
.chats-list{flex:1;overflow-y:auto;padding:6px 8px}
.chat-item{padding:11px 12px;border-radius:10px;cursor:pointer;font-size:13.5px;color:var(--text-dim);margin-bottom:2px;display:flex;align-items:center;gap:8px;transition:background .12s,color .12s}
.chat-item:hover,.chat-item.active{background:var(--surface);color:var(--text)}
.chat-item .row{flex:1;overflow:hidden;display:flex;flex-direction:column;gap:2px}
.chat-title{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:500}
.chat-prev{font-size:11px;color:var(--text-muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.del-btn{background:transparent;border:none;color:var(--text-muted);font-size:16px;padding:4px 6px;cursor:pointer;border-radius:6px;opacity:.5;transition:opacity .12s,color .12s}
.chat-item:hover .del-btn{opacity:1}
.del-btn:active{color:var(--danger)}
.backdrop{position:fixed;inset:0;background:rgba(0,0,0,.6);backdrop-filter:blur(2px);-webkit-backdrop-filter:blur(2px);z-index:99;opacity:0;pointer-events:none;transition:opacity .2s}
.backdrop.open{opacity:1;pointer-events:auto}
.messages{flex:1;overflow-y:auto;padding:18px 14px;scroll-behavior:smooth}
.messages-inner{max-width:780px;margin:0 auto;display:flex;flex-direction:column;gap:18px}
.msg{display:flex;flex-direction:column;gap:5px;max-width:100%;animation:fadeUp .22s cubic-bezier(.4,0,.2,1)}
@keyframes fadeUp{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
.msg.user{align-items:flex-end}
.msg.assistant{align-items:flex-start}
.bubble{padding:11px 15px;border-radius:18px;line-height:1.55;word-wrap:break-word;overflow-wrap:break-word;font-size:15px;max-width:92%}
.msg.user .bubble{background:var(--user-grad);color:#fff;border-bottom-right-radius:5px;box-shadow:0 2px 8px rgba(99,102,241,.18)}
.msg.assistant .bubble{background:var(--surface);color:var(--text);border-bottom-left-radius:5px;border:1px solid var(--border)}
.bubble.empty-streaming{min-height:40px}
.msg-img{max-width:240px;max-height:240px;border-radius:12px;margin:4px 0;display:block;border:1px solid var(--border);object-fit:cover}
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
.caption{font-size:11px;color:var(--text-muted);padding:0 8px;display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.badge{font-size:10px;padding:2px 7px;border-radius:5px;font-weight:500}
.b-search{background:rgba(245,158,11,.13);color:var(--search)}
.b-reason{background:rgba(168,85,247,.13);color:var(--reason)}
.thinking{display:inline-flex;gap:3px;padding:6px 0;align-items:center}
.thinking span{width:6px;height:6px;border-radius:50%;background:var(--text-muted);animation:pulse 1.3s ease-in-out infinite}
.thinking span:nth-child(2){animation-delay:.15s}
.thinking span:nth-child(3){animation-delay:.3s}
@keyframes pulse{0%,80%,100%{opacity:.3;transform:scale(.85)}40%{opacity:1;transform:scale(1)}}
.tool-running{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--text-dim);padding:8px 12px;background:var(--surface);border:1px solid var(--border);border-radius:10px;margin-bottom:4px;width:fit-content}
.tool-running.search{color:var(--search);border-color:rgba(245,158,11,.3)}
.tool-running.reason{color:var(--reason);border-color:rgba(168,85,247,.3)}
.empty{text-align:center;padding:40px 20px;max-width:520px;margin:40px auto 0}
.empty .greet{font-size:26px;font-weight:700;letter-spacing:-.02em;background:var(--user-grad);-webkit-background-clip:text;background-clip:text;color:transparent;margin-bottom:6px;line-height:1.2}
.empty .sub{color:var(--text-dim);font-size:14px;margin-bottom:24px}
.suggestions{display:flex;flex-direction:column;gap:8px;max-width:380px;margin:0 auto}
.sugg{background:var(--surface);border:1px solid var(--border);color:var(--text-dim);padding:10px 14px;border-radius:12px;cursor:pointer;font-size:13.5px;text-align:left;transition:all .15s;font-family:inherit}
.sugg:hover{background:var(--panel);color:var(--text);border-color:var(--border-strong)}
.sugg:active{transform:scale(.98)}
.input-bar{padding:10px 12px;border-top:1px solid var(--border);background:var(--bg);flex-shrink:0;padding-bottom:max(10px,env(safe-area-inset-bottom))}
.input-row{max-width:780px;margin:0 auto;display:flex;gap:8px;align-items:flex-end}
.attach{background:transparent;border:1px solid var(--border);color:var(--text-dim);border-radius:50%;width:40px;height:40px;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:all .15s}
.attach:hover{background:var(--surface);color:var(--text)}
.attach:active{transform:scale(.95)}
.input-wrap{flex:1;background:var(--surface);border:1px solid var(--border);border-radius:22px;transition:border-color .15s;display:flex;flex-direction:column;overflow:hidden}
.input-wrap:focus-within{border-color:var(--accent)}
.image-previews{display:flex;gap:6px;padding:8px 10px 0;flex-wrap:wrap}
.image-previews:empty{display:none}
.preview-item{position:relative;width:54px;height:54px;border-radius:8px;overflow:hidden;border:1px solid var(--border)}
.preview-item img{width:100%;height:100%;object-fit:cover}
.preview-item .rm{position:absolute;top:2px;right:2px;background:rgba(0,0,0,.75);color:#fff;border:none;border-radius:50%;width:18px;height:18px;font-size:11px;cursor:pointer;display:flex;align-items:center;justify-content:center;padding:0;line-height:1}
.input{background:transparent;color:var(--text);border:none;padding:11px 16px;font-size:15px;resize:none;max-height:160px;font-family:inherit;outline:none;line-height:1.45;width:100%}
.input::placeholder{color:var(--text-muted)}
.send{background:var(--user-grad);color:#fff;border:none;border-radius:50%;width:40px;height:40px;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:all .15s;box-shadow:0 2px 8px rgba(99,102,241,.25)}
.send:disabled{background:#26262b;color:#52525b;box-shadow:none;cursor:not-allowed}
.send:active:not(:disabled){transform:scale(.95)}
.send svg{width:18px;height:18px}
.modal-back{position:fixed;inset:0;background:rgba(0,0,0,.75);backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);z-index:200;display:none;align-items:center;justify-content:center;padding:20px;animation:fadeIn .2s}
.modal-back.open{display:flex}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
.modal{background:var(--panel);border:1px solid var(--border);border-radius:18px;max-width:440px;width:100%;padding:28px;box-shadow:var(--shadow);max-height:85vh;overflow-y:auto}
.modal h2{margin:0 0 6px;font-size:22px;font-weight:700;letter-spacing:-.02em;background:var(--user-grad);-webkit-background-clip:text;background-clip:text;color:transparent}
.modal .sub{color:var(--text-dim);font-size:14px;margin-bottom:20px}
.modal label{display:block;font-size:12px;font-weight:600;color:var(--text-dim);margin:14px 0 6px;text-transform:uppercase;letter-spacing:.04em}
.modal input,.modal textarea{width:100%;background:var(--surface);color:var(--text);border:1px solid var(--border);border-radius:10px;padding:10px 14px;font-size:14px;font-family:inherit;outline:none}
.modal input:focus,.modal textarea:focus{border-color:var(--accent)}
.modal .actions{display:flex;gap:8px;margin-top:20px}
.btn{padding:10px 16px;border-radius:10px;font-size:14px;font-weight:600;cursor:pointer;border:none;transition:all .15s;flex:1;font-family:inherit}
.btn-primary{background:var(--user-grad);color:#fff}
.btn-primary:active{transform:scale(.98);filter:brightness(.95)}
.btn-ghost{background:transparent;color:var(--text-dim);border:1px solid var(--border)}
.btn-ghost:hover{background:var(--surface);color:var(--text)}
.divider{height:1px;background:var(--border);margin:18px -28px}
.key-display{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px;font-family:"JetBrains Mono",monospace;font-size:15px;text-align:center;letter-spacing:.05em;margin:8px 0;color:#a5b4fc;word-break:break-all}
.tabs{display:flex;gap:4px;background:var(--surface);padding:4px;border-radius:10px;margin-bottom:16px}
.tab{flex:1;background:transparent;border:none;color:var(--text-dim);padding:8px;border-radius:7px;cursor:pointer;font-weight:500;font-size:13px;transition:all .12s;font-family:inherit}
.tab.active{background:var(--panel);color:var(--text)}
.mem-list{display:flex;flex-direction:column;gap:6px;max-height:300px;overflow-y:auto;padding-right:4px}
.mem-item{display:flex;align-items:center;gap:8px;background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:8px 12px;font-size:13px}
.mem-item .text{flex:1;color:var(--text-dim)}
.mem-item .rm{background:transparent;border:none;color:var(--text-muted);cursor:pointer;font-size:14px;padding:2px 6px;border-radius:4px}
.mem-item .rm:hover{color:var(--danger)}
.mem-empty{text-align:center;color:var(--text-muted);font-size:13px;padding:24px;font-style:italic}
.toast{position:fixed;top:max(16px,env(safe-area-inset-top));left:50%;transform:translateX(-50%) translateY(-20px);background:var(--panel);border:1px solid var(--border-strong);color:var(--text);padding:10px 18px;border-radius:10px;font-size:13px;z-index:300;opacity:0;pointer-events:none;transition:opacity .2s,transform .2s;box-shadow:var(--shadow)}
.toast.show{opacity:1;transform:translateX(-50%) translateY(0)}
.copy-btn{position:absolute;top:6px;right:6px;background:rgba(255,255,255,.08);border:1px solid var(--border);color:var(--text-dim);padding:3px 8px;border-radius:6px;font-size:11px;cursor:pointer;opacity:.6;transition:opacity .15s;font-family:inherit}
.bubble pre:hover .copy-btn{opacity:1}
</style>
</head>
<body>
<div class="header">
  <button class="icon-btn" id="menuBtn" aria-label="menu">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
  </button>
  <div class="title" id="chatTitle">Router</div>
  <button class="icon-btn" id="settingsBtn" aria-label="settings">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
  </button>
</div>
<div class="backdrop" id="backdrop"></div>
<aside class="sidebar" id="sidebar">
  <div class="sb-head"><button class="new-chat" id="newBtn">
    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
    New chat
  </button></div>
  <div class="chats-list" id="chatsList"></div>
</aside>
<div class="messages" id="messages"><div class="messages-inner" id="messagesInner"></div></div>
<div class="input-bar"><div class="input-row">
  <button class="attach" id="attachBtn" aria-label="attach image">
    <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
  </button>
  <input type="file" id="fileInput" accept="image/*" multiple style="display:none">
  <div class="input-wrap">
    <div class="image-previews" id="imagePreviews"></div>
    <textarea class="input" id="input" placeholder="Ask anything…" rows="1"></textarea>
  </div>
  <button class="send" id="sendBtn" aria-label="send">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/></svg>
  </button>
</div></div>
<div class="modal-back" id="welcomeModal">
  <div class="modal">
    <h2>Welcome.</h2>
    <p class="sub">Your chats live in the cloud, tied to a personal sync key. Use the same key on any device to access them.</p>
    <div class="tabs">
      <button class="tab active" data-tab="new">First time</button>
      <button class="tab" data-tab="have">I have a key</button>
    </div>
    <div id="tab-new">
      <label>Your new sync key</label>
      <div class="key-display" id="newKey"></div>
      <p class="sub" style="font-size:12px;margin-top:8px">Save this somewhere safe — a password manager, a note, anywhere. It's the only way back to your chats on another device.</p>
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
<div class="modal-back" id="settingsModal">
  <div class="modal">
    <h2>Settings</h2>
    <p class="sub">Your sync key, memories, and exports.</p>
    <label>Sync key (for other devices)</label>
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
    <div class="actions" style="margin-top:8px">
      <button class="btn btn-primary" id="closeSettings">Done</button>
    </div>
  </div>
</div>
<div class="toast" id="toast"></div>

<script>
const $ = (id) => document.getElementById(id);
const SK_STORE = "router_sync_key_v1";
let toastTimer;
function toast(msg){const t=$("toast");t.textContent=msg;t.classList.add("show");clearTimeout(toastTimer);toastTimer=setTimeout(()=>t.classList.remove("show"),2200)}
const WORDS=["amber","arrow","aspen","autumn","azure","basil","birch","blade","blaze","bloom","brave","brick","bridge","brook","calm","canyon","cedar","cherry","clay","clear","cloud","clover","coast","comet","copper","coral","cosmos","crane","creek","crimson","crystal","cypress","dawn","deep","desert","drift","dune","dusk","ember","emerald","fable","falcon","feather","fern","flame","flint","forest","frost","glade","golden","granite","grove","harbor","harmony","haven","hazel","heron","horizon","indigo","ivory","jade","jasper","juniper","keen","lake","lantern","laurel","leaf","light","linen","lupine","maple","marble","meadow","midnight","mist","moon","moss","mountain","north","oak","ocean","olive","onyx","opal","orchid","otter","pearl","pebble","petal","pine","plum","poppy","prairie","quartz","quiet","quill","rain","rapid","raven","reed","ridge","ripple","river","robin","rose","rust","saffron","sage","sapphire","scarlet","shade","shadow","shore","silver","sky","slate","snow","sparrow","spring","spruce","star","stone","storm","stream","summer","sunset","swift","teal","thicket","thorn","thunder","tide","tiger","topaz","trail","tundra","valley","velvet","violet","vista","walnut","wave","wheat","whisper","willow","wind","winter","wren"];
function genKey(){const p=()=>WORDS[Math.floor(Math.random()*WORDS.length)];return p()+"-"+p()+"-"+p()+"-"+p()}

marked.setOptions({breaks:true,gfm:true});
function renderMarkdown(text){const raw=marked.parse(text||"");return DOMPurify.sanitize(raw,{ADD_ATTR:["target"]})}
function addCopyButtons(container){
  container.querySelectorAll("pre").forEach((pre)=>{
    if(pre.querySelector(".copy-btn")) return;
    const btn=document.createElement("button");btn.className="copy-btn";btn.textContent="copy";
    btn.onclick=(e)=>{e.stopPropagation();const code=pre.querySelector("code")?.innerText||pre.innerText;navigator.clipboard.writeText(code).then(()=>{btn.textContent="copied";setTimeout(()=>btn.textContent="copy",1200)})};
    pre.appendChild(btn);
  });
}

let syncKey=localStorage.getItem(SK_STORE)||"";
let currentChatId=null, currentMessages=[], isStreaming=false, pendingImages=[];
const uuid=()=>Date.now().toString(36)+Math.random().toString(36).slice(2,8);

async function api(path,opts={}){const r=await fetch(path,opts);if(!r.ok) throw new Error("HTTP "+r.status);return r.json()}
const apiListChats=()=>api("/chats?sk="+encodeURIComponent(syncKey));
const apiGetChat=(id)=>api("/chats/"+id+"?sk="+encodeURIComponent(syncKey));
const apiSaveChat=(chat)=>api("/chats/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({sync_key:syncKey,chat})});
const apiDeleteChat=(id)=>api("/chats/"+id+"?sk="+encodeURIComponent(syncKey),{method:"DELETE"});
const apiGetMem=()=>api("/memories?sk="+encodeURIComponent(syncKey));
const apiSaveMem=(mems)=>api("/memories/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({sync_key:syncKey,memories:mems})});

function timeOfDay(){const h=new Date().getHours();if(h<5)return "still up?";if(h<12)return "morning";if(h<17)return "afternoon";if(h<21)return "evening";return "late one"}
const SUGGESTIONS=["What's something interesting in the news today?","Help me think through a decision","Explain something I'm confused about","Brainstorm with me for a project"];
function renderEmpty(){
  const root=$("messagesInner"); root.innerHTML="";
  const e=document.createElement("div"); e.className="empty";
  const greet=document.createElement("div"); greet.className="greet"; greet.textContent="Good "+timeOfDay()+".";
  const sub=document.createElement("div"); sub.className="sub"; sub.textContent="Ask anything. I can search the web, reason through hard problems, or just talk.";
  const sg=document.createElement("div"); sg.className="suggestions";
  for(const text of SUGGESTIONS){const b=document.createElement("button");b.className="sugg";b.textContent=text;b.onclick=()=>{$("input").value=text;$("input").focus()};sg.appendChild(b)}
  e.appendChild(greet); e.appendChild(sub); e.appendChild(sg);
  root.appendChild(e);
}

function makeBubble(role,content,images,meta,isStreamingFlag){
  const wrap=document.createElement("div"); wrap.className="msg "+role;
  const b=document.createElement("div"); b.className="bubble"+(isStreamingFlag?" empty-streaming":"");
  if(Array.isArray(images)&&images.length){
    for(const img of images){
      const i=document.createElement("img");
      i.src="data:"+img.mimeType+";base64,"+img.data;
      i.className="msg-img"; i.alt="uploaded";
      b.appendChild(i);
    }
  }
  if(role==="assistant"){
    if(isStreamingFlag && !content){
      b.innerHTML += '<div class="thinking"><span></span><span></span><span></span></div>';
    } else {
      const div=document.createElement("div");
      div.innerHTML=renderMarkdown(content); addCopyButtons(div); b.appendChild(div);
    }
  } else {
    if(content){const t=document.createElement("div"); t.textContent=content; t.style.whiteSpace="pre-wrap"; b.appendChild(t)}
  }
  wrap.appendChild(b);
  if(role==="assistant" && meta && (meta.responder||meta.tool_used)){
    const cap=document.createElement("div"); cap.className="caption"; cap.innerHTML=captionHTML(meta);
    wrap.appendChild(cap);
  }
  return {wrap, bubble:b};
}
function captionHTML(meta){
  let s="via "+(meta.responder||"lead");
  if(meta.tool_used==="web_search") s+=' <span class="badge b-search">searched</span>';
  if(meta.tool_used==="ask_reasoner") s+=' <span class="badge b-reason">→ reasoner</span>';
  return s;
}
function renderAll(){
  const root=$("messagesInner");
  if(currentMessages.length===0){renderEmpty();return}
  root.innerHTML="";
  for(const m of currentMessages){const {wrap}=makeBubble(m.role,m.content,m.images,m.meta,false);root.appendChild(wrap)}
  scrollBottom();
}
function scrollBottom(){const el=$("messages");el.scrollTop=el.scrollHeight}

async function renderList(){
  if(!syncKey) return;
  try{
    const chats=await apiListChats();
    const root=$("chatsList"); root.innerHTML="";
    if(chats.length===0){const e=document.createElement("div");e.className="mem-empty";e.textContent="No chats yet";e.style.padding="20px 12px";root.appendChild(e);return}
    for(const c of chats){
      const it=document.createElement("div");
      it.className="chat-item"+(c.id===currentChatId?" active":"");
      const row=document.createElement("div");row.className="row";
      const t=document.createElement("div");t.className="chat-title";t.textContent=c.title||"New chat";
      const p=document.createElement("div");p.className="chat-prev";p.textContent=c.preview||"";
      row.appendChild(t);if(c.preview) row.appendChild(p);
      row.onclick=()=>loadChat(c.id);
      const d=document.createElement("button");d.className="del-btn";d.textContent="×";
      d.onclick=async(e)=>{e.stopPropagation();if(confirm("Delete this chat?")){await apiDeleteChat(c.id);if(c.id===currentChatId) newChat();renderList()}};
      it.appendChild(row);it.appendChild(d);
      root.appendChild(it);
    }
  }catch(e){console.error(e)}
}

function newChat(){currentChatId=null;currentMessages=[];pendingImages=[];$("chatTitle").textContent="Router";renderPreviews();renderAll();renderList();closeSB()}
async function loadChat(id){
  try{const chat=await apiGetChat(id);if(!chat) return;currentChatId=id;currentMessages=chat.messages||[];$("chatTitle").textContent=chat.title||"Chat";renderAll();renderList();closeSB()}
  catch(e){toast("Failed to load chat")}
}
async function persist(){
  if(!currentChatId) currentChatId=uuid();
  const title=currentMessages.find(m=>m.role==="user")?.content?.slice(0,50)||"New chat";
  await apiSaveChat({id:currentChatId,title,messages:currentMessages,updated:Date.now()});
  $("chatTitle").textContent=title; renderList();
}
function openSB(){$("sidebar").classList.add("open");$("backdrop").classList.add("open")}
function closeSB(){$("sidebar").classList.remove("open");$("backdrop").classList.remove("open")}

function renderPreviews(){
  const root=$("imagePreviews"); root.innerHTML="";
  pendingImages.forEach((img,idx)=>{
    const item=document.createElement("div");item.className="preview-item";
    const i=document.createElement("img");i.src="data:"+img.mimeType+";base64,"+img.data;
    const rm=document.createElement("button");rm.className="rm";rm.textContent="×";
    rm.onclick=()=>{pendingImages.splice(idx,1);renderPreviews()};
    item.appendChild(i);item.appendChild(rm);root.appendChild(item);
  });
}
async function fileToImage(file){
  return new Promise((res,rej)=>{
    const r=new FileReader();
    r.onload=()=>{const result=r.result;const [meta,b64]=result.split(",");const mt=meta.match(/data:([^;]+);/)[1];res({mimeType:mt,data:b64})};
    r.onerror=rej; r.readAsDataURL(file);
  });
}
$("attachBtn").onclick=()=>$("fileInput").click();
$("fileInput").onchange=async(e)=>{
  for(const f of e.target.files){
    if(f.size>4*1024*1024){toast("Image too large (max 4MB)");continue}
    pendingImages.push(await fileToImage(f));
  }
  renderPreviews(); e.target.value="";
};

async function send(){
  if(isStreaming) return;
  const input=$("input"); const text=input.value.trim();
  if(!text && pendingImages.length===0) return;
  isStreaming=true; $("sendBtn").disabled=true;

  const userMsg={role:"user",content:text,images:pendingImages.slice()};
  currentMessages.push(userMsg);
  pendingImages=[]; renderPreviews();
  input.value=""; input.style.height="auto";

  const aMsg={role:"assistant",content:"",meta:{}};
  currentMessages.push(aMsg);

  if($("messagesInner").querySelector(".empty")) $("messagesInner").innerHTML="";
  const {wrap:userWrap}=makeBubble("user",text,userMsg.images,null,false);
  $("messagesInner").appendChild(userWrap);
  const {wrap:aWrap,bubble:aBub}=makeBubble("assistant","",null,null,true);
  $("messagesInner").appendChild(aWrap);
  scrollBottom();

  try{
    const resp=await fetch("/chat",{
      method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({
        sync_key:syncKey,
        messages:currentMessages.slice(0,-1).map(m=>({role:m.role,content:m.content,images:m.images||[]}))
      })
    });
    if(!resp.ok||!resp.body){aMsg.content="Error: HTTP "+resp.status;aBub.innerHTML=renderMarkdown(aMsg.content);return}
    const reader=resp.body.getReader(); const dec=new TextDecoder(); let buf="";
    let toolEl=null;
    while(true){
      const {done,value}=await reader.read(); if(done) break;
      buf+=dec.decode(value,{stream:true});
      const parts=buf.split("\\n\\n"); buf=parts.pop()||"";
      for(const evt of parts){
        const lines=evt.split("\\n"); let et="",data="";
        for(const ln of lines){
          if(ln.startsWith("event: ")) et=ln.slice(7);
          else if(ln.startsWith("data: ")) data=ln.slice(6);
        }
        if(!data) continue;
        let p; try{p=JSON.parse(data)}catch{continue}
        if(et==="status") aMsg.meta.responder=p.leader;
        else if(et==="token"){
          aMsg.content+=p.text;
          aBub.innerHTML=renderMarkdown(aMsg.content);
          addCopyButtons(aBub);
          scrollBottom();
        } else if(et==="tool_start"){
          aMsg.meta.tool_used=p.tool;
          if(!toolEl){
            toolEl=document.createElement("div");
            toolEl.className="tool-running "+(p.tool==="web_search"?"search":"reason");
            toolEl.innerHTML=(p.tool==="web_search"?"🔍 searching the web…":"🧠 delegating to reasoner…");
            aWrap.insertBefore(toolEl,aBub);
          }
        } else if(et==="delegate") aMsg.meta.responder=p.to;
        else if(et==="tool_done"){
          if(toolEl){toolEl.remove();toolEl=null}
          if(!aMsg.content) aBub.innerHTML='<div class="thinking"><span></span><span></span><span></span></div>';
        } else if(et==="done") Object.assign(aMsg.meta,p);
        else if(et==="error"){aMsg.content+="\\n\\n*Error: "+p.message+"*";aBub.innerHTML=renderMarkdown(aMsg.content)}
      }
    }
    if(toolEl) toolEl.remove();
    aBub.innerHTML=renderMarkdown(aMsg.content||"(empty response)");
    addCopyButtons(aBub);
    aBub.classList.remove("empty-streaming");
    const cap=document.createElement("div"); cap.className="caption"; cap.innerHTML=captionHTML(aMsg.meta);
    aWrap.appendChild(cap);
    await persist();
  } catch(e){
    aMsg.content="Error: "+e.message; aBub.innerHTML=renderMarkdown(aMsg.content);
  } finally{
    isStreaming=false; $("sendBtn").disabled=false; scrollBottom();
  }
}

function showWelcome(){$("welcomeModal").classList.add("open");refreshNewKey()}
function refreshNewKey(){$("newKey").textContent=genKey()}
function pickTab(name){document.querySelectorAll(".tab").forEach(t=>t.classList.toggle("active",t.dataset.tab===name));$("tab-new").style.display=name==="new"?"":"none";$("tab-have").style.display=name==="have"?"":"none"}
document.querySelectorAll(".tab").forEach(t=>t.onclick=()=>pickTab(t.dataset.tab));
$("regenKey").onclick=refreshNewKey;
$("useNewKey").onclick=async()=>{syncKey=$("newKey").textContent.trim();localStorage.setItem(SK_STORE,syncKey);$("welcomeModal").classList.remove("open");toast("Welcome ☆ your key is saved");await renderList();renderAll()};
$("useExistingKey").onclick=async()=>{const v=$("existingKey").value.trim();if(v.length<4){toast("That doesn't look right");return}syncKey=v;localStorage.setItem(SK_STORE,syncKey);$("welcomeModal").classList.remove("open");toast("Synced.");await renderList();renderAll()};

async function openSettings(){
  $("myKey").textContent=syncKey;
  const mems=await apiGetMem().catch(()=>[]);
  const list=$("memList"); list.innerHTML="";
  if(mems.length===0){const e=document.createElement("div");e.className="mem-empty";e.textContent="The leader hasn't learned anything about you yet. Have a few conversations and check back.";list.appendChild(e)}
  else{for(const m of mems){const it=document.createElement("div");it.className="mem-item";const t=document.createElement("div");t.className="text";t.textContent=m;const rm=document.createElement("button");rm.className="rm";rm.textContent="×";rm.onclick=async()=>{const f=mems.filter(x=>x!==m);await apiSaveMem(f);openSettings()};it.appendChild(t);it.appendChild(rm);list.appendChild(it)}}
  $("settingsModal").classList.add("open");
}
$("copyKey").onclick=()=>{navigator.clipboard.writeText(syncKey);toast("Key copied")};
$("clearMem").onclick=async()=>{if(!confirm("Forget everything the leader has learned about you?")) return;await apiSaveMem([]);openSettings();toast("Cleared")};
$("exportAll").onclick=async()=>{const chats=await apiListChats();const full=[];for(const c of chats) full.push(await apiGetChat(c.id));const blob=new Blob([JSON.stringify({exported:new Date().toISOString(),chats:full},null,2)],{type:"application/json"});const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="router-chats-"+Date.now()+".json";a.click()};
$("closeSettings").onclick=()=>$("settingsModal").classList.remove("open");
$("settingsBtn").onclick=openSettings;

$("sendBtn").onclick=send;
$("input").addEventListener("keydown",(e)=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send()}});
$("input").addEventListener("input",(e)=>{e.target.style.height="auto";e.target.style.height=Math.min(e.target.scrollHeight,160)+"px"});
$("menuBtn").onclick=openSB;
$("backdrop").onclick=()=>{closeSB();$("welcomeModal").classList.remove("open");$("settingsModal").classList.remove("open")};
$("newBtn").onclick=newChat;

(async()=>{
  if(!syncKey){showWelcome();return}
  await renderList(); renderAll();
})();
</script>
</body>
</html>`;

const MANIFEST = JSON.stringify({
  name: "Router", short_name: "Router", start_url: "/", display: "standalone",
  background_color: "#09090b", theme_color: "#09090b",
  icons: [{ src: "/icon.svg", sizes: "any", type: "image/svg+xml", purpose: "any" }],
});

const ICON_SVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 192 192"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#6366f1"/><stop offset="1" stop-color="#a855f7"/></linearGradient></defs><rect width="192" height="192" fill="url(#g)" rx="42"/><text x="96" y="132" font-size="115" text-anchor="middle" fill="white" font-family="Inter,sans-serif" font-weight="700">R</text></svg>`;

Deno.serve(async (req) => {
  const url = new URL(req.url);
  const p = url.pathname;

  if (p === "/" && req.method === "GET")
    return new Response(HTML, { headers: { "Content-Type": "text/html; charset=utf-8" } });
  if (p === "/manifest.json")
    return new Response(MANIFEST, { headers: { "Content-Type": "application/json" } });
  if (p === "/icon.svg")
    return new Response(ICON_SVG, { headers: { "Content-Type": "image/svg+xml" } });

  if (p === "/chat" && req.method === "POST") return handleChat(req);
  if (p === "/chats" && req.method === "GET") return handleChatsList(req);
  if (p === "/chats/save" && req.method === "POST") return handleChatSave(req);
  const m = p.match(/^\/chats\/([^\/]+)$/);
  if (m) {
    if (req.method === "GET") return handleChatGet(req, m[1]);
    if (req.method === "DELETE") return handleChatDelete(req, m[1]);
  }
  if (p === "/memories" && req.method === "GET") return handleMemGet(req);
  if (p === "/memories/save" && req.method === "POST") return handleMemSave(req);

  if (p === "/health") {
    return Response.json({
      ok: true,
      google_ai_key: !!GOOGLE_AI_KEY,
      openrouter_key: !!OPENROUTER_KEY,
      tavily_key: !!TAVILY_KEY,
    });
  }
  return new Response("Not found", { status: 404 });
});
