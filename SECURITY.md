# Security Policy

## Reporting a vulnerability

Please **do not** open a public issue for security problems. Email the
maintainer directly (see the repository profile) with:

- a subject line prefixed `[SECURITY]`,
- a description of the vulnerability,
- the affected version / commit,
- a minimal reproducer if possible.

You will receive an acknowledgement within 72 hours and a fix plan as soon as
the impact is understood. We ask that you keep details private until a release
is available.

## Scope

The following are in scope:

- Access-control bypasses: any path where a principal retrieves chunks that
  their clearance level or departments should exclude.
- Authentication / token weaknesses: JWT forgery, claim confusion, replay
  across audiences, weak signing secrets.
- Information disclosure: error or audit payloads leaking secrets, decode
  internals, or document contents beyond the caller's permissions.
- Injection into the retrieval pipeline: metadata or ingested content shaping
  filters, prompts, or queries in unintended ways.

The identity store defaults to a durable SQLite database
(`data/users.db` — schema, PBKDF2 credential hashing, account lockout and a
`login_attempts` audit trail). It is a sound identity source for a
single-instance deployment. It is **not** an IdP: multi-service/SSO systems
must replace the `UserStore` seam with Okta, Entra ID, Keycloak (OIDC) or a
central user table, and provision accounts centrally rather than via the
bundled `scripts/manage_users.py`.

## Deployment hardening checklist

1. Set `ENVIRONMENT=production`. The app will refuse to start without a strong
   `AUTH_JWT_SECRET` (>= 32 random characters).
2. Do **not** rely on the auto-seeded demo accounts or `*-password!`
   passwords: `AUTH_SEED_DEMO_USERS` is only honoured outside production.
   Provision real accounts with `python -m scripts.manage_users` before
   exposing the service.
3. Back up `data/users.db` / its WAL as part of disaster recovery; a lost DB
   means lost accounts.
4. Terminate TLS at the ingress; the API itself speaks HTTP.
5. Run the API behind a distributed rate limiter / WAF in multi-node
   deployments (`AUTH_RATE_LIMIT_ENABLED=false` to hand off to it).
6. Scrub the JWT secret from image layers: pass it via orchestration secrets,
   never bake it into the image or an `.env` file committed to git.
7. Keep audit records (`/audit/access`, structured logs, `login_attempts`)
   out of the public image; stream them to an aggregator (Splunk, ELK,
   CloudWatch).

## Supported versions

Only the current default branch (`main`) receives security fixes. Tags are
advisory.