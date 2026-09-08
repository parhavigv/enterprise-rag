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

Out of scope: the static developer user store (`app/auth/users.json`) is a
dev/test stand-in, not a production identity source. Production deployments
must replace `UserStore` with an IdP or a real identity table.

## Deployment hardening checklist

1. Set `ENVIRONMENT=production`. The app will refuse to start without a strong
   `AUTH_JWT_SECRET` (>= 32 random characters).
2. Do **not** reuse the default dev user records or `*-password!` passwords.
3. Terminate TLS at the ingress; the API itself speaks HTTP.
4. Run the API behind a distributed rate limiter / WAF in multi-node
   deployments (`AUTH_RATE_LIMIT_ENABLED=false` to hand off to it).
5. Scrub the JWT secret from image layers: pass it via orchestration secrets,
   never bake it into the image or an `.env` file committed to git.
6. Keep audit records (`/audit/access` + structured logs) out of the public
   image; stream them to an aggregator (Splunk, ELK, CloudWatch).

## Supported versions

Only the current default branch (`main`) receives security fixes. Tags are
advisory.