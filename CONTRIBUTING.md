# Contributing

Thanks for contributing to Enterprise RAG. This is a solo-maintained project
with a deliberately strict bar: everything must lint clean, type-check, and
carry tests — especially across the RBAC/governance layer.

## Development setup

```bash
python -m venv .venv
.\.venv\Scripts\activate        # Windows (Linux/macOS: source .venv/bin/activate)
pip install -e ".[dev]"         # runtime + dev/test dependencies
copy .env.example .env          # Windows (Linux/macOS: cp .env.example .env)
```

Run the app and check the matrices:

```bash
python -m pytest tests/ -q                      # full suite (231 tests)
python -m pytest tests/test_rbac_guardrails.py -v
python -m pytest tests/test_auth_hardening.py -v
python -m ruff check .                          # lint
python -m ruff format --check .                 # formatting
```

## Guidelines

- **Security first.** If your change touches retrieval, filtering, or token
  handling, extend the adversarial suites (`test_rbac_guardrails.py`,
  `test_auth_hardening.py`). A regression that lets one protected chunk
  through **must** fail a test.
- **Fail loudly.** Do not silently downgrade sensitive state (e.g. an invalid
  clearance level) to a looser default.
- **No secrets in the repo.** `AUTH_JWT_SECRET`, API keys, and `.env` never
  enter git. Dev user records are hashed PBKDF2 placeholders only.
- **Style.** Black-ish, 100 columns, enforced by ruff (`line-length = 100`).
  Docstrings on public modules/classes; changes must pass `ruff check` and
  `ruff format --check`.
- **Dependencies.** Pin exact versions in `requirements.txt` (runtime) and
  `requirements-dev.txt` (test). Keep prod images lean — no test tooling in
  the runtime requirements.

## Process

1. Open an issue describing the change and its motivation.
2. Branch from `main`, implement, add tests.
3. Run the full matrix locally (see above) and confirm green.
4. Open a pull request. CI runs lint, formatting, the test suite with a
   coverage gate, dependency audit, and a Docker build.

**Note on scope:** this repository is maintained by its author for
demonstration and portfolio purposes; PRs are reviewed on a best-effort basis.