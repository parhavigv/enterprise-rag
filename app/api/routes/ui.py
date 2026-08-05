"""Browser UI for the RAG demo - modern light-enterprise SPA.

Upload / drag-and-drop documents, attach images for vision chat, record a
voice question (Whisper transcription), pick the model, and read grounded,
source-cited answers.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["ui"])

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Enterprise RAG — Workspace</title>
<style>
  :root {
    --bg:#f5f6f8; --surface:#ffffff; --border:#e4e7ec; --border-2:#d3d8e0;
    --ink:#0f172a; --text:#3f4c5f; --muted:#8a94a6;
    --brand:#2563eb; --brand-dark:#1d4ed8; --brand-soft:#eff4ff;
    --ok:#16a34a; --ok-soft:#ecfdf3; --err:#dc2626; --err-soft:#fef2f2;
    --warn:#b45309; --warn-soft:#fffbeb;
    --radius:12px; --radius-sm:8px;
    --shadow:0 1px 2px rgba(16,24,40,.04), 0 1px 3px rgba(16,24,40,.06);
    --shadow-lg:0 8px 24px rgba(16,24,40,.10);
    --font:-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", Helvetica, Arial, sans-serif;
  }
  * { box-sizing:border-box; }
  html,body { margin:0; height:100%; }
  body { background:var(--bg); color:var(--text); font-family:var(--font); font-size:14px; line-height:1.55; }
  button { font-family:inherit; cursor:pointer; }
  ::selection { background:#bfdbfe; }

  /* ---------- header ---------- */
  header { position:sticky; top:0; z-index:20; background:rgba(255,255,255,.92); backdrop-filter:blur(8px);
    border-bottom:1px solid var(--border); }
  .bar { max-width:1280px; margin:0 auto; padding:12px 24px; display:flex; align-items:center; gap:14px; }
  .logo { display:flex; align-items:center; gap:10px; font-weight:700; color:var(--ink); font-size:15px; }
  .logo .mark { width:30px; height:30px; border-radius:8px; background:linear-gradient(135deg,#2563eb,#7c3aed);
    display:grid; place-items:center; color:#fff; box-shadow:var(--shadow); }
  .logo small { display:block; font-weight:500; color:var(--muted); font-size:11px; letter-spacing:.02em; }
  .spacer { flex:1; }
  .select-wrap { position:relative; }
  .select-wrap select { appearance:none; border:1px solid var(--border-2); background:var(--surface);
    color:var(--ink); font:inherit; font-weight:600; padding:8px 34px 8px 12px; border-radius:var(--radius-sm);
    box-shadow:var(--shadow); }
  .select-wrap svg { position:absolute; right:10px; top:50%; transform:translateY(-50%); pointer-events:none; color:var(--muted); }
  .pill { display:inline-flex; align-items:center; gap:7px; padding:7px 12px; border-radius:999px;
    font-size:12.5px; font-weight:600; border:1px solid var(--border); background:var(--surface); color:var(--text); }
  .pill .dot { width:8px; height:8px; border-radius:50%; background:#9aa3b2; }
  .pill.ready .dot { background:var(--ok); box-shadow:0 0 0 3px var(--ok-soft); }
  .pill.degraded .dot { background:var(--warn); box-shadow:0 0 0 3px var(--warn-soft); }
  .pill.offline .dot { background:var(--err); box-shadow:0 0 0 3px var(--err-soft); }

  /* ---------- layout ---------- */
  .layout { max-width:1280px; margin:0 auto; padding:22px 24px 48px; display:grid; gap:20px;
    grid-template-columns: minmax(0,1fr) 380px; align-items:start; }
  @media (max-width:1020px) { .layout { grid-template-columns:1fr; } .sidebar { order:-1; } }
  .card { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius);
    box-shadow:var(--shadow); }
  .card-h { padding:14px 18px; border-bottom:1px solid var(--border); display:flex; align-items:center; gap:10px; }
  .card-h h2 { margin:0; font-size:13px; font-weight:700; color:var(--ink); letter-spacing:.04em; text-transform:uppercase; }
  .card-h .tag { margin-left:auto; font-size:11.5px; color:var(--muted); }
  select.scope { margin-left:auto; appearance:none; border:1px solid var(--border-2); background:var(--surface);
    color:var(--ink); font:inherit; font-size:12px; font-weight:600; padding:5px 22px 5px 10px; border-radius:6px;
    cursor:pointer; max-width:190px; }
  .card-b { padding:16px 18px; }
  .fld { display:block; margin-bottom:11px; font-size:12px; font-weight:600; color:var(--muted); }
  .fld select,.fld input { display:block; width:100%; margin-top:4px; font:inherit; font-weight:500; color:var(--ink);
    background:var(--surface); border:1px solid var(--border-2); border-radius:var(--radius-sm); padding:8px 10px; }
  .fld select:focus,.fld input:focus { outline:2px solid #bfdbfe; border-color:var(--brand); }
  .mini-hint { font-size:11.5px; color:var(--muted); margin-top:6px; }

  /* ---------- upload ---------- */
  .drop { border:1.5px dashed var(--border-2); border-radius:var(--radius); padding:22px 16px; text-align:center;
    color:var(--muted); transition:border-color .15s, background .15s; }
  .drop.drag { border-color:var(--brand); background:var(--brand-soft); color:var(--brand-dark); }
  .drop svg { margin-bottom:6px; color:var(--brand); }
  .drop strong { color:var(--text); display:block; font-size:13.5px; }
  .drop .sub { font-size:12px; }
  .filechip { display:flex; align-items:center; gap:10px; margin-top:12px; padding:10px 12px;
    background:var(--brand-soft); border:1px solid #dbe6fe; border-radius:var(--radius-sm); font-size:13px; }
  .filechip .fname { font-weight:600; color:var(--ink); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .filechip .fsize { color:var(--muted); font-size:12px; }
  .filechip button { margin-left:auto; background:none; border:0; color:var(--muted); font-size:16px; line-height:1; }
  .filechip button:hover { color:var(--err); }
  .btn { display:inline-flex; align-items:center; gap:7px; background:var(--brand); color:#fff; border:0;
    border-radius:var(--radius-sm); padding:9px 16px; font-size:13.5px; font-weight:600; transition:background .15s; }
  .btn:hover { background:var(--brand-dark); }
  .btn:disabled { opacity:.55; cursor:not-allowed; }
  .btn.ghost { background:var(--surface); color:var(--text); border:1px solid var(--border-2); }
  .btn.ghost:hover { background:#f1f3f7; }
  .btn .spin { display:inline-block; width:13px; height:13px; border:2px solid rgba(255,255,255,.4);
    border-top-color:#fff; border-radius:50%; animation:sp .8s linear infinite; }
  .btn.ghost .spin { border-color:var(--border-2); border-top-color:var(--brand); }
  @keyframes sp { to { transform:rotate(360deg); } }
  .status { margin-top:10px; font-size:12.5px; white-space:pre-wrap; }
  .status.ok { color:var(--ok); } .status.err { color:var(--err); } .status.info { color:var(--muted); }
  table.metrics { width:100%; border-collapse:collapse; margin-top:10px; font-size:12.5px; }
  table.metrics td { padding:5px 8px; border-bottom:1px solid var(--border); }
  table.metrics td:first-child { color:var(--muted); font-weight:600; width:55%; }
  table.metrics td:last-child { font-family:"SFMono-Regular",Consolas,monospace; font-size:12px; }

  /* ---------- chat ---------- */
  .chat { display:flex; flex-direction:column; height:calc(100vh - 180px); min-height:460px; }
  .messages { flex:1; overflow-y:auto; padding:18px; display:flex; flex-direction:column; gap:14px; }
  .msg { max-width:82%; display:flex; flex-direction:column; gap:4px; }
  .msg.user { align-self:flex-end; align-items:flex-end; }
  .msg.assistant { align-self:flex-start; align-items:flex-start; }
  .bubble { padding:11px 14px; border-radius:14px; font-size:13.5px; white-space:pre-wrap; word-break:break-word; }
  .msg.user .bubble { background:var(--brand); color:#fff; border-bottom-right-radius:4px; box-shadow:var(--shadow); }
  .msg.assistant .bubble { background:var(--surface); border:1px solid var(--border); color:var(--ink);
    border-bottom-left-radius:4px; box-shadow:var(--shadow); }
  .msg .who { font-size:11px; color:var(--muted); font-weight:600; letter-spacing:.03em; text-transform:uppercase; padding:0 4px; }
  .msg .thumb { max-width:220px; max-height:160px; border-radius:10px; border:1px solid var(--border); margin-bottom:6px; }
  .msg .cites { font-size:12px; color:var(--muted); margin-top:6px; }
  .cite { display:inline-flex; align-items:center; gap:5px; background:var(--brand-soft); color:var(--brand-dark);
    border:1px solid #dbe6fe; border-radius:6px; padding:3px 8px; margin:3px 4px 0 0; font-size:11.5px; font-weight:600;
    cursor:pointer; }
  .cite:hover { background:#dbe7ff; }
  .typing { display:inline-flex; gap:4px; padding:12px 14px; background:var(--surface); border:1px solid var(--border);
    border-radius:14px; }
  .typing i { width:6px; height:6px; border-radius:50%; background:#c3cad6; animation:blink 1.2s infinite; }
  .typing i:nth-child(2){ animation-delay:.2s; } .typing i:nth-child(3){ animation-delay:.4s; }
  @keyframes blink { 0%,80%,100%{opacity:.3} 40%{opacity:1} }
  .empty-chat { margin:auto; text-align:center; color:var(--muted); max-width:360px; }
  .empty-chat svg { color:#d7dce5; margin-bottom:8px; }

  .composer { border-top:1px solid var(--border); padding:12px 14px; background:#fbfbfc; }
  .attachments { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:8px; }
  .att { position:relative; }
  .att img { width:56px; height:56px; object-fit:cover; border-radius:8px; border:1px solid var(--border); }
  .att button { position:absolute; top:-6px; right:-6px; width:18px; height:18px; border-radius:50%;
    background:var(--ink); color:#fff; border:0; font-size:11px; line-height:1; }
  .inputrow { display:flex; gap:8px; align-items:flex-end; }
  .inputrow textarea { flex:1; resize:none; border:1px solid var(--border-2); border-radius:var(--radius-sm);
    padding:10px 12px; font:inherit; color:var(--ink); background:var(--surface); max-height:140px; min-height:44px; }
  .inputrow textarea:focus { outline:2px solid #bfdbfe; border-color:var(--brand); }
  .iconbtn { width:42px; height:42px; border-radius:var(--radius-sm); border:1px solid var(--border-2);
    background:var(--surface); color:var(--text); display:grid; place-items:center; flex:none; }
  .iconbtn:hover { background:#f1f3f7; }
  .iconbtn.rec { background:var(--err-soft); border-color:#fca5a5; color:var(--err); animation:pulse 1.1s infinite; }
  @keyframes pulse { 0%,100%{ box-shadow:0 0 0 0 rgba(220,38,38,.35);} 50%{ box-shadow:0 0 0 7px rgba(220,38,38,0);} }
  .sendbtn { background:var(--brand); border:0; border-radius:var(--radius-sm); width:42px; height:42px;
    display:grid; place-items:center; color:#fff; flex:none; }
  .sendbtn:disabled { opacity:.55; }
  .toast { position:fixed; bottom:22px; left:50%; transform:translateX(-50%); background:var(--ink); color:#fff;
    padding:10px 18px; border-radius:10px; font-size:13px; box-shadow:var(--shadow-lg); opacity:0;
    pointer-events:none; transition:opacity .2s; z-index:50; max-width:90%; }
  .toast.show { opacity:1; }
  .toast.err { background:var(--err); }
  .toast.ok { background:var(--ok); }
</style>
</head>
<body>
<header>
  <div class="bar">
    <div class="logo">
      <span class="mark">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
      </span>
      <span>Enterprise RAG<small>Grounded AI workspace</small></span>
    </div>
    <div class="spacer"></div>
    <div class="select-wrap">
      <select id="model" title="Answering model"></select>
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="m6 9 6 6 6-6"/></svg>
    </div>
    <span class="pill" id="health"><span class="dot"></span>checking…</span>
  </div>
</header>

<main class="layout">
  <!-- chat column -->
  <section class="card">
    <div class="card-h"><h2>Ask your documents</h2>
      <select id="scope" class="scope" title="Restrict answers to one document"><option value="">All documents</option></select>
      <span class="tag" id="chatMeta"></span></div>
    <div class="chat">
      <div class="messages" id="messages">
        <div class="empty-chat" id="empty">
          <svg width="42" height="42" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
          <p>Upload a document, then ask a question about it. The answer is grounded strictly in your files and cites the exact sources.</p>
        </div>
      </div>
      <div class="composer">
        <div class="attachments" id="attachments"></div>
        <div class="inputrow">
          <button class="iconbtn" id="imgBtn" title="Attach an image (vision chat)">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="m21 15-5-5L5 21"/></svg>
          </button>
          <input type="file" id="imgInput" accept="image/*" multiple hidden>
          <button class="iconbtn" id="micBtn" title="Record a voice question">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="22"/></svg>
          </button>
          <textarea id="q" rows="1" placeholder="Ask about your documents… (Enter to send, Shift+Enter for a new line)"></textarea>
          <button class="sendbtn" id="askBtn" title="Send">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
          </button>
        </div>
      </div>
    </div>
  </section>

  <!-- sidebar -->
  <aside class="sidebar" style="display:grid;gap:20px">
    <section class="card">
      <div class="card-h"><h2>Upload a document</h2><span class="tag" id="upStat"></span></div>
      <div class="card-b">
        <div class="drop" id="drop">
          <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
          <strong>Drop a file here</strong>
          <span class="sub">or <a href="#" id="browse">browse</a> · PDF, DOCX, TXT, MD, CSV, JSON</span>
        </div>
        <input type="file" id="file" accept=".pdf,.docx,.txt,.md,.markdown,.csv,.json" hidden>
        <div id="fileChip"></div>
        <button class="btn" id="uploadBtn" style="width:100%;justify-content:center;margin-top:12px">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
          Upload &amp; index
        </button>
        <button class="btn ghost" id="reviewBtn" style="width:100%;justify-content:center;margin-top:8px;display:none">Review this document</button>
        <div class="status" id="uploadStatus"></div>
      </div>
    </section>

    <section class="card">
      <div class="card-h"><h2>Sources</h2><span class="tag" id="srcCount"></span></div>
      <div class="card-b" id="sources" style="max-height:420px;overflow-y:auto">
        <div class="status info" id="srcEmpty">No sources yet — ask a question to see the retrieved passages.</div>
      </div>
    </section>

    <section class="card">
      <div class="card-h"><h2>Answer engine</h2><span class="tag" id="engineTag">…</span></div>
      <div class="card-b">
        <label class="fld">Provider
          <select id="setProvider">
            <option value="ollama">Ollama (local)</option>
            <option value="openai">OpenAI (GPT)</option>
          </select>
        </label>
        <label class="fld">Model
          <input id="setModel" type="text" placeholder="llama3 or gpt-4o" autocomplete="off">
        </label>
        <label class="fld">API key
          <input id="setKey" type="password" placeholder="sk-… (blank keeps current)" autocomplete="off">
        </label>
        <button class="btn ghost" id="applySet" style="width:100%;justify-content:center;margin-top:4px">Apply settings</button>
        <div class="status" id="setStatus"></div>
        <div class="mini-hint">Connect an OpenAI key for richer, explained answers — no server restart needed.</div>
      </div>
    </section>

    <section class="card">
      <div class="card-h"><h2>Session</h2></div>
      <div class="card-b">
        <div class="status info" id="meta">model: — · generated: — · latency: — · cache: —</div>
      </div>
    </section>
  </aside>
</main>

<div class="toast" id="toast"></div>

<script>
"use strict";
const $ = id => document.getElementById(id);
const MODELS = [
  { label: "Ollama · llama3 (local)",  provider: "ollama", model: "llama3" },
  { label: "Ollama · llama3.1 (local)",provider: "ollama", model: "llama3.1" },
  { label: "OpenAI · GPT-4o",          provider: "openai", model: "gpt-4o" },
  { label: "OpenAI · GPT-4.1",         provider: "openai", model: "gpt-4.1" },
  { label: "OpenAI · GPT-4o-mini",     provider: "openai", model: "gpt-4o-mini" },
];
let pendingImages = [];      // data URIs
let pendingFile = null;
let lastSource = '';         // metadata source of the most recent upload
let lastSourceLabel = '';
let scopeSources = [];       // [{source, count}] from /api/v1/sources
let recording = null;        // MediaRecorder
let recordedChunks = [];

const esc = s => s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const baseName = s => String(s || '').split(/[\\/]/).pop() || s;

function errMsg(j, fallback) {
  if (j && j.error && j.error.message) return j.error.message;
  if (j && Array.isArray(j.detail)) return j.detail.map(d => d.msg || 'invalid input').join('; ');
  if (j && typeof j.detail === 'string') return j.detail;
  return fallback;
}

function toast(msg, cls) { const t = $('toast'); t.textContent = msg; t.className = 'toast show ' + (cls||''); clearTimeout(t._h); t._h = setTimeout(()=>t.className='toast', 2600); }

/* ---------------- model selector ---------------- */
async function loadModels() {
  try {
    const r = await fetch('/api/v1/stats'); const j = await r.json();
    const sel = $('model');
    MODELS.forEach(m => { const o = document.createElement('option'); o.value = m.provider + '::' + m.model;
      o.textContent = m.label; sel.appendChild(o); });
    const cur = (j.llm_model || '').toLowerCase();
    const hit = MODELS.findIndex(m => m.model.toLowerCase() === cur);
    if (hit >= 0) sel.selectedIndex = hit;
  } catch (e) { /* stats unavailable - keep defaults */ }
}

/* ---------------- health ---------------- */
async function health() {
  try {
    const r = await fetch('/health/ready'); const j = await r.json();
    const ok = j.status === 'ok' && Object.values(j.checks).every(Boolean);
    $('health').className = 'pill ' + (ok ? 'ready' : 'degraded');
    $('health').innerHTML = '<span class="dot"></span>' + (ok ? 'ready' : 'degraded');
  } catch (e) {
    $('health').className = 'pill offline';
    $('health').innerHTML = '<span class="dot"></span>offline';
  }
}

/* ---------------- upload ---------------- */
function renderFileChip() {
  $('fileChip').innerHTML = pendingFile
    ? `<div class="filechip"><span class="fname">${esc(pendingFile.name)}</span>
       <span class="fsize">${(pendingFile.size/1024).toFixed(1)} KB</span>
       <button title="Remove" onclick="pendingFile=null;renderFileChip();$('file').value=''">×</button></div>`
    : '';
}
$('drop').addEventListener('click', e => { if (e.target.id !== 'browse') $('file').click(); });
$('browse').addEventListener('click', e => { e.preventDefault(); $('file').click(); });
['dragover','dragenter'].forEach(ev => $('drop').addEventListener(ev, e => { e.preventDefault(); $('drop').classList.add('drag'); }));
['dragleave','drop'].forEach(ev => $('drop').addEventListener(ev, e => { e.preventDefault(); $('drop').classList.remove('drag'); }));
$('drop').addEventListener('drop', e => { const f = e.dataTransfer.files && e.dataTransfer.files[0]; if (f) { pendingFile = f; renderFileChip(); } });
$('file').addEventListener('change', e => { const f = e.target.files && e.target.files[0]; if (f) { pendingFile = f; renderFileChip(); } });

$('uploadBtn').addEventListener('click', async () => {
  if (!pendingFile) { toast('Choose a file first.', 'err'); return; }
  const btn = $('uploadBtn');
  btn.disabled = true; btn.innerHTML = '<span class="spin"></span>Indexing…';
  $('upStat').textContent = 'parsing + embedding';
  $('uploadStatus').className = 'status info';
  $('uploadStatus').textContent = 'Parsing, chunking, embedding & indexing — first run may take a while…';
  try {
    const fd = new FormData();
    fd.append('file', pendingFile);
    fd.append('chunk_size', '512T');
    const r = await fetch('/api/v1/upload', { method: 'POST', body: fd });
    const j = await r.json();
    if (!r.ok) throw new Error(errMsg(j, 'Upload failed'));
    const m = j.metrics || {};
    const rows = ['filename','chunks','chroma_count','bm25_count','total_seconds']
      .filter(k => m[k] !== undefined)
      .map(k => `<tr><td>${esc(k)}</td><td>${esc(String(m[k]))}</td></tr>`).join('');
    $('uploadStatus').className = 'status ok';
    $('uploadStatus').innerHTML = '✓ Indexed ' + (m.chunks || '?') + ' chunk(s)<table class="metrics">' + rows + '</table>';
    $('upStat').textContent = 'indexed';
    lastSource = m.filename || '';
    lastSourceLabel = lastSource;
    if (lastSource) { loadScope(); $('scope').value = lastSource; updateReviewVisibility(); }
    toast('Document indexed — ask away.', 'ok');
  } catch (e) {
    $('uploadStatus').className = 'status err';
    $('uploadStatus').textContent = '✗ ' + e.message;
    $('upStat').textContent = 'failed';
    toast('Upload failed: ' + e.message, 'err');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg> Upload &amp; index';
  }
});

/* ---------------- image attach ---------------- */
$('imgBtn').addEventListener('click', () => $('imgInput').click());
$('imgInput').addEventListener('change', e => {
  const files = Array.from(e.target.files || []);
  files.forEach(f => {
    const reader = new FileReader();
    reader.onload = () => {
      if (pendingImages.length >= 4) { toast('Max 4 images.', 'err'); return; }
      pendingImages.push(reader.result);
      renderAttachments();
    };
    reader.readAsDataURL(f);
  });
  e.target.value = '';
});
function renderAttachments() {
  $('attachments').innerHTML = pendingImages.map((d,i) =>
    `<div class="att"><img src="${d}" alt="attached"><button title="Remove" onclick="pendingImages.splice(${i},1);renderAttachments()">×</button></div>`
  ).join('');
}

/* ---------------- voice ---------------- */
const hasMic = typeof navigator !== 'undefined' && navigator.mediaDevices && navigator.mediaDevices.getUserMedia && typeof MediaRecorder !== 'undefined';
if (!hasMic) $('micBtn').style.display = 'none';
$('micBtn').addEventListener('click', async () => {
  if (recording) { recording.stop(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recording = new MediaRecorder(stream);
    recordedChunks = [];
    recording.ondataavailable = e => { if (e.data.size) recordedChunks.push(e.data); };
    recording.onstop = async () => {
      recording = null; $('micBtn').classList.remove('rec');
      const blob = new Blob(recordedChunks, { type: 'audio/webm' });
      stream.getTracks().forEach(t => t.stop());
      await transcribe(blob);
    };
    recording.start();
    $('micBtn').classList.add('rec');
    toast('Recording… click the mic to stop.', '');
  } catch (e) {
    toast('Microphone unavailable: ' + e.message, 'err');
  }
});
async function transcribe(blob) {
  toast('Transcribing…', '');
  try {
    const fd = new FormData();
    fd.append('file', blob, 'recording.webm');
    const r = await fetch('/api/v1/transcribe', { method: 'POST', body: fd });
    const j = await r.json();
    if (!r.ok) throw new Error(errMsg(j, 'Transcription failed'));
    if (j.text) { $('q').value = j.text; toast('Heard: "' + j.text.slice(0, 60) + '"', 'ok'); ask(); }
    else toast('No speech detected.', 'err');
  } catch (e) {
    toast(e.message, 'err');
  }
}

/* ---------------- chat ---------------- */
const messagesEl = $('messages');
function appendMsg(role, html, opts = {}) {
  $('empty').style.display = 'none';
  const wrap = document.createElement('div');
  wrap.className = 'msg ' + role;
  let inner = '<span class="who">' + (role === 'user' ? 'You' : 'Assistant') + '</span>';
  if (opts.images && opts.images.length) {
    inner += opts.images.map(d => `<img class="thumb" src="${d}" alt="attached">`).join('');
  }
  inner += '<div class="bubble">' + html + '</div>';
  if (opts.meta) inner += '<div class="cites">' + opts.meta + '</div>';
  wrap.innerHTML = inner;
  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return wrap;
}
function typing(show) {
  let el = $('typingEl');
  if (show && !el) {
    const wrap = document.createElement('div');
    wrap.className = 'msg assistant'; wrap.id = 'typingEl';
    wrap.innerHTML = '<div class="typing"><i></i><i></i><i></i></div>';
    messagesEl.appendChild(wrap);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  } else if (!show && el) el.remove();
}
function renderSources(srch) {
  const empty = $('srcEmpty');
  if (empty) empty.style.display = 'none';
  if (!srch || !srch.length) {
    $('srcCount').textContent = '0 hits';
    $('sources').innerHTML = '<div class="status info" id="srcEmpty">No relevant passages retrieved — the answer may not be grounded.</div>';
    return;
  }
  $('srcCount').textContent = srch.length + ' hits';
  $('sources').innerHTML = srch.map((s,i) => {
    const fname = (s.metadata && (s.metadata.filename || s.metadata.source))
      ? baseName(s.metadata.filename || s.metadata.source) : '';
    const snippet = String((s && s.text) || '');
    return `
    <div style="padding:9px 10px;background:#f8fafc;border:1px solid var(--border);border-radius:8px;margin-bottom:8px">
      <div style="font-size:11px;color:var(--muted);margin-bottom:4px">
        <b style="color:var(--brand-dark)">score ${Number(s.score).toFixed(3)}</b>
        · ${esc(s.source||'hybrid')}${fname ? ' · <span style="color:var(--text)">' + esc(fname) + '</span>' : ''}${s.metadata && s.metadata.page_number ? ' · p.' + s.metadata.page_number : ''}
      </div>
      <div style="font-size:12px;color:var(--text);white-space:pre-wrap;word-break:break-word">${esc(snippet.length>420 ? snippet.slice(0,420)+'…' : snippet)}</div>
    </div>`;
  }).join('');
}
async function ask() {
  const q = $('q').value.trim();
  if (!q && !pendingImages.length) return;
  const chosen = $('model').value.split('::');
  const provider = chosen[0], model = chosen[1];
  const source = $('scope').value || null;
  const images = pendingImages.slice();
  appendMsg('user', esc(q), { images });
  pendingImages = []; renderAttachments(); $('q').value = '';
  typing(true);
  const btn = $('askBtn'); btn.disabled = true;
  $('chatMeta').textContent = 'working…';
  try {
    const body = { query: q, top_k: 5, provider, model };
    if (images.length) body.images = images;
    if (source) body.source = source;
    const r = await fetch('/api/v1/query', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
    });
    const j = await r.json();
    if (!r.ok) throw new Error(errMsg(j, 'Query failed'));
    typing(false);
    const cites = j.search && j.search.length
      ? j.search.map((s,i) => `<span class="cite" onclick="toggleSrc(${i})">[${i+1}]</span>`).join('')
      : '';
    appendMsg('assistant', esc(j.answer || '(no answer)'), { meta: cites });
    if (j.search) renderSources(j.search);
    $('meta').textContent = 'model: ' + (j.model || '—') + ' · generated: ' + j.generated +
      ' · latency: ' + (j.latency_ms/1000).toFixed(1) + 's · cache: ' + j.cache_hit +
      (source ? ' · scope: ' + source : '');
    $('chatMeta').textContent = 'answered';
  } catch (e) {
    typing(false);
    appendMsg('assistant', '<b>Error:</b> ' + esc(e.message));
    $('chatMeta').textContent = 'error';
  } finally { btn.disabled = false; }
}
function toggleSrc(i) {
  const cards = $('sources').querySelectorAll('div[style*="background:#f8fafc"]');
  if (cards[i]) cards[i].scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/* ---------------- document scope ---------------- */
function updateReviewVisibility() {
  $('reviewBtn').style.display = ($('scope').value ? '' : 'none');
}
function scopeOptions() {
  const sel = $('scope');
  const prev = sel.value;
  sel.innerHTML = '<option value="">All documents</option>';
  const seen = {};
  scopeSources.forEach(s => {
    seen[s.source] = true;
    const o = document.createElement('option');
    o.value = s.source;
    o.textContent = '\u21a6 ' + baseName(s.source);
    sel.appendChild(o);
  });
  if (lastSource && !seen[lastSource]) {
    const o = document.createElement('option');
    o.value = lastSource;
    o.textContent = '\u21a6 ' + baseName(lastSourceLabel || lastSource);
    sel.appendChild(o);
  }
  if (prev && Array.from(sel.options).some(o => o.value === prev)) sel.value = prev;
  else if (!sel.value && sel.options.length === 2) sel.value = sel.options[1].value;
  updateReviewVisibility();
}
async function loadScope() {
  try {
    const r = await fetch('/api/v1/sources');
    const j = await r.json();
    if (!r.ok) throw new Error('sources unavailable');
    scopeSources = Array.isArray(j) ? j : [];
  } catch (e) { scopeSources = []; }
  scopeOptions();
}

/* ---------------- answer engine settings ---------------- */
function syncModelSelect(provider, model) {
  const sel = $('model');
  const key = (provider || 'ollama') + '::' + (model || '');
  const hit = Array.from(sel.options).find(o => o.value === key);
  if (hit) { sel.value = key; return; }
  const o = document.createElement('option');
  o.value = key;
  o.textContent = (provider || 'ollama') + ' \u00b7 ' + (model || '') + ' (active)';
  sel.appendChild(o);
  sel.value = key;
}
async function loadSettings() {
  try {
    const r = await fetch('/api/v1/settings');
    const j = await r.json();
    if (!r.ok) throw new Error('settings unavailable');
    $('setProvider').value = j.provider || 'ollama';
    $('setModel').value = j.model || '';
    $('engineTag').textContent = j.api_key_masked ? 'key set' : 'no API key';
    syncModelSelect(j.provider, j.model);
  } catch (e) { $('engineTag').textContent = 'unavailable'; }
}
$('applySet').addEventListener('click', async () => {
  const btn = $('applySet'); btn.disabled = true;
  $('setStatus').className = 'status info';
  $('setStatus').textContent = 'Applying…';
  try {
    const body = { provider: $('setProvider').value };
    const model = $('setModel').value.trim();
    if (model) body.model = model;
    const key = $('setKey').value.trim();
    if (key) body.api_key = key;
    const r = await fetch('/api/v1/settings', {
      method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
    });
    const j = await r.json();
    if (!r.ok) throw new Error(errMsg(j, 'Update failed'));
    $('setStatus').className = 'status ok';
    $('setStatus').textContent = 'Saved: ' + j.provider + ' \u00b7 ' + j.model;
    $('engineTag').textContent = j.api_key_masked ? 'key set' : 'no API key';
    $('setKey').value = '';
    syncModelSelect(j.provider, j.model);
    toast('Answer engine updated — next question uses ' + j.model, 'ok');
  } catch (e) {
    $('setStatus').className = 'status err';
    $('setStatus').textContent = '\u2717 ' + e.message;
  } finally { btn.disabled = false; }
});
$('reviewBtn').addEventListener('click', () => {
  $('q').value = 'Review this document: give an overview, its strengths, gaps or risks, and concrete recommendations.';
  ask();
});

$('askBtn').addEventListener('click', ask);
$('q').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask(); }
});
$('scope').addEventListener('change', updateReviewVisibility);

loadModels(); loadSettings(); loadScope(); health(); setInterval(health, 15000);
</script>
</body>
</html>
"""


@router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def ui() -> HTMLResponse:
    return HTMLResponse(content=PAGE, headers={"Cache-Control": "no-store"})
