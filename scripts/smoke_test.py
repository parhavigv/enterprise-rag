"""Live end-to-end smoke test.

Spawns uvicorn as a detached process, waits for readiness, exercises every
API surface, then terminates the server. Requires a running Ollama.

Usage:  python scripts/smoke_test.py
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


HOST, PORT = "127.0.0.1", _free_port()
BASE = f"http://{HOST}:{PORT}"
PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")
if not Path(PYTHON).exists():
    PYTHON = sys.executable

CREATE_NO_WINDOW = 0x08000000


def main() -> int:
    env = dict(os.environ)
    env.setdefault("NLTK_DISABLE_IMPORT_SECURITY", "1")
    env.setdefault("LLM_MODEL", "llama3")
    # Reranker downloads ~430MB on first use; disable for the fast smoke path
    # (it is covered by the mocked unit tests).
    env.setdefault("RERANKER_ENABLED", "false")
    env["LOG_FORMAT"] = "console"

    proc = subprocess.Popen(
        [
            PYTHON,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            HOST,
            "--port",
            str(PORT),
            "--log-level",
            "warning",
        ],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=CREATE_NO_WINDOW,
    )

    failures: list[str] = []
    checks: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        marker = "PASS" if ok else "FAIL"
        checks.append(f"{marker}  {name}  {detail}")
        if not ok:
            failures.append(name)

    try:
        # Wait for readiness
        deadline = time.time() + 90
        ready = False
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            try:
                r = httpx.get(f"{BASE}/health/ready", timeout=5)
                if r.status_code == 200:
                    body = r.json()
                    ready = body.get("status") == "ok"
                    if ready:
                        break
            except Exception:
                pass
            time.sleep(2)
        check("server ready (all probes green)", ready)

        if not ready:
            print("\n".join(checks))
            proc.kill()
            return 1

        # 1. Liveness
        r = httpx.get(f"{BASE}/health", timeout=10)
        check("GET /health -> 200", r.status_code == 200)

        # 2. Stats
        r = httpx.get(f"{BASE}/api/v1/stats", timeout=10)
        check("GET /api/v1/stats -> 200", r.status_code == 200)
        if r.status_code == 200:
            body = r.json()
            check(
                "index has data (chroma>0, bm25>0)",
                body["chroma_count"] > 0 and body["bm25_count"] > 0,
            )

        # 3. Search (retrieval only)
        r = httpx.post(
            f"{BASE}/api/v1/search",
            json={"query": "What does hybrid retrieval fuse?", "top_k": 3, "generate": False},
            timeout=60,
        )
        check("POST /api/v1/search -> 200", r.status_code == 200)
        if r.status_code == 200:
            body = r.json()
            check("search returns documents", len(body["search"]) > 0)
            check(
                "search result has node_id+text+score",
                all(set(s) >= {"node_id", "text", "score"} for s in body["search"]),
            )

        # 4. Full RAG query (LLM generation against Ollama)
        r = httpx.post(
            f"{BASE}/api/v1/query",
            json={"query": "What is the default chunk size?", "top_k": 3},
            timeout=180,
        )
        if r.status_code != 200:
            print(f"\n[query failed] status={r.status_code} body={r.text[:500]}")
        check("POST /api/v1/query -> 200", r.status_code == 200)
        if r.status_code == 200:
            body = r.json()
            check("query generates an answer", bool(body.get("answer")))
            check("query is grounded in sources", len(body.get("search", [])) > 0)
            print("\n--- Generated answer ---")
            print(body["answer"][:600])
            print("------------------------")

        # 5. Caching: repeat identical query should be fast and hit cache
        t0 = time.perf_counter()
        httpx.post(
            f"{BASE}/api/v1/search", json={"query": "cache me please", "top_k": 2}, timeout=60
        )
        first = time.perf_counter() - t0
        t0 = time.perf_counter()
        httpx.post(
            f"{BASE}/api/v1/search", json={"query": "cache me please", "top_k": 2}, timeout=60
        )
        second = time.perf_counter() - t0
        check("repeated query served fast (cache)", second <= first)

        # 6. Validation errors
        r = httpx.post(f"{BASE}/api/v1/query", json={"query": "   "}, timeout=10)
        check("blank query -> 422", r.status_code == 422)
        r = httpx.post(f"{BASE}/api/v1/ingest", json={"path": "x", "format": "bad"}, timeout=10)
        check("bad ingest format -> 422", r.status_code == 422)

        # 7. Request ID header
        r = httpx.get(f"{BASE}/health", headers={"X-Request-ID": "smoke-1"}, timeout=10)
        check("request-id header echoed", r.headers.get("X-Request-ID") == "smoke-1")

        # 8. Unknown route -> 404
        r = httpx.get(f"{BASE}/does-not-exist", timeout=10)
        check("unknown route -> 404", r.status_code == 404)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    print("\n" + "\n".join(checks))
    print(f"\n{len(checks) - len(failures)}/{len(checks)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
