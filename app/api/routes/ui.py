"""Minimal browser UI for the RAG demo (upload + ask)."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["ui"])

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Enterprise RAG - Live Demo</title>
<style>
  :root { --bg:#0f172a; --panel:#1e293b; --accent:#38bdf8; --ok:#4ade80; --err:#f87171; --text:#e2e8f0; --muted:#94a3b8; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background: var(--bg); color: var(--text); }
  header { padding: 20px 28px; border-bottom: 1px solid #334155; display:flex; align-items:center; gap:12px; }
  header h1 { font-size: 18px; margin:0; }
  header .pill { font-size:12px; padding:3px 10px; border-radius:999px; background:#334155; color:var(--muted); }
  main { max-width: 900px; margin: 24px auto; padding: 0 20px; display: grid; gap: 20px; }
  .card { background: var(--panel); border:1px solid #334155; border-radius: 12px; padding: 18px; }
  .card h2 { font-size: 14px; margin: 0 0 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
  .row { display:flex; gap:10px; flex-wrap: wrap; align-items:center; }
  input[type=file] { color: var(--muted); }
  input[type=text] { flex:1; min-width:220px; background:#0f172a; border:1px solid #334155; color:var(--text); border-radius:8px; padding:10px 12px; font-size:14px; }
  button { background: var(--accent); color:#082f49; border:0; border-radius:8px; padding:10px 16px; font-size:14px; font-weight:600; cursor:pointer; }
  button:disabled { opacity:.5; cursor:not-allowed; }
  .status { font-size:13px; margin-top:10px; white-space:pre-wrap; }
  .ok { color: var(--ok); } .err { color: var(--err); }
  .answer { background:#0f172a; border-left:3px solid var(--accent); padding:12px 14px; border-radius:8px; font-size:14px; line-height:1.6; white-space:pre-wrap; }
  .src { margin-top:8px; font-size:12.5px; color:var(--muted); }
  .src div { padding:8px 10px; background:#0f172a; border-radius:6px; margin-top:6px; border:1px solid #1e293b; }
  .meta { font-size:12px; color:var(--muted); margin-top:8px; }
  table.metrics { width:100%; border-collapse:collapse; font-size:13px; }
  table.metrics td { padding:5px 8px; border-bottom:1px solid #1e293b; }
  table.metrics td:first-child { color: var(--muted); }
  .spin { display:inline-block; width:14px; height:14px; border:2px solid #082f49; border-top-color:transparent; border-radius:50%; animation:sp .8s linear infinite; vertical-align:-2px; }
  @keyframes sp { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<header>
  <h1>Enterprise RAG — live demo</h1>
  <span class="pill" id="health">checking…</span>
</header>
<main>

  <section class="card">
    <h2>1. Upload a document</h2>
    <div class="row">
      <input type="file" id="file" accept=".pdf,.docx,.txt,.md,.markdown,.csv,.json">
      <button id="uploadBtn" onclick="upload()">Upload &amp; index</button>
    </div>
    <div class="status" id="uploadStatus"></div>
  </section>

  <section class="card">
    <h2>2. Ask a question</h2>
    <div class="row">
      <input type="text" id="q" placeholder="e.g. What does the document say about X?" onkeydown="if(event.key==='Enter')ask()">
      <button id="askBtn" onclick="ask()">Ask</button>
    </div>
    <div class="status" id="askStatus"></div>
    <div class="answer" id="answer" hidden></div>
    <div class="src" id="sources" hidden></div>
    <div class="meta" id="meta"></div>
  </section>

</main>
<script>
  const $ = id => document.getElementById(id);

  async function health() {
    try {
      const r = await fetch('/health/ready');
      const j = await r.json();
      const ok = j.status === 'ok' && Object.values(j.checks).every(Boolean);
      $('health').textContent = ok ? '● ready' : '● degraded';
      $('health').style.color = ok ? 'var(--ok)' : 'var(--err)';
    } catch (e) {
      $('health').textContent = '● offline';
      $('health').style.color = 'var(--err)';
    }
  }

  function setStatus(el, html, cls) {
    el.className = 'status ' + (cls || '');
    el.innerHTML = html;
  }

  async function upload() {
    const f = $('file').files[0];
    if (!f) return setStatus($('uploadStatus'), 'Choose a file first.', 'err');
    const btn = $('uploadBtn'); btn.disabled = true;
    setStatus($('uploadStatus'), '<span class="spin"></span> Parsing, chunking, embedding & indexing (first run loads models, may take a while)…');
    try {
      const fd = new FormData();
      fd.append('file', f);
      const r = await fetch('/api/v1/upload', { method: 'POST', body: fd });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error ? j.error.message : 'Upload failed');
      const m = j.metrics || {};
      const rows = ['filename','chunks','chroma_count','bm25_count','total_seconds','embed_throughput_chunks_per_min']
        .filter(k => m[k] !== undefined)
        .map(k => `<tr><td>${k}</td><td>${m[k]}</td></tr>`).join('');
      setStatus($('uploadStatus'),
        '<span class="ok">✓ Indexed ' + (m.chunks||'?') + ' chunks</span>' +
        '<table class="metrics">' + rows + '</table>');
    } catch (e) {
      setStatus($('uploadStatus'), '✗ ' + e.message, 'err');
    } finally { btn.disabled = false; }
  }

  async function ask() {
    const q = $('q').value.trim();
    if (!q) return;
    const btn = $('askBtn'); btn.disabled = true;
    setStatus($('askStatus'), '<span class="spin"></span> Retrieving & generating…');
    $('answer').hidden = true; $('sources').hidden = true; $('meta').textContent = '';
    try {
      const r = await fetch('/api/v1/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, top_k: 5 })
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error ? j.error.message : 'Query failed');
      setStatus($('askStatus'), '');
      $('answer').textContent = j.answer || '(no answer)';
      $('answer').hidden = false;
      if (j.search && j.search.length) {
        $('sources').innerHTML = '<b>Sources (' + j.search.length + ')</b>' + j.search
          .map(s => '<div><b>score ' + Number(s.score).toFixed(3) + '</b> — ' + escapeHtml(s.text) + '</div>')
          .join('');
        $('sources').hidden = false;
      }
      $('meta').textContent = 'model: ' + (j.model||'—') + ' · generated: ' + j.generated +
        ' · latency: ' + (j.latency_ms/1000).toFixed(1) + 's · cache: ' + j.cache_hit;
    } catch (e) {
      setStatus($('askStatus'), '✗ ' + e.message, 'err');
    } finally { btn.disabled = false; }
  }

  function escapeHtml(s) { return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

  health();
  setInterval(health, 15000);
</script>
</body>
</html>
"""


@router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def ui() -> str:
    return PAGE
