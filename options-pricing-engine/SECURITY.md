# Security Guide

This document summarises the security posture and operational controls for the
Options Pricing Engine (OPE).

## Authentication

* **OpenID Connect (OIDC) is the canonical identity mechanism.**
  * `OIDC_ISSUER`, `OIDC_AUDIENCE` and `OIDC_JWKS_URL` must be configured in
    every production deployment. The service refuses to start in production when
    any of them are missing or blank.
  * Access tokens are validated against the issuer's JWKS. Signature, audience,
    issuer, expiry (`exp`), not-before (`nbf`) and issued-at (`iat`) claims are
    required. A ±60 second clock-skew allowance is applied during validation.
  * Signing keys are cached for five minutes and the previous keyset is kept as
    a grace period so that rotations do not trigger authentication failures.
* **Development fallback.** When an OIDC provider is unavailable (for example,
  on an isolated developer laptop), you may provide `DEV_JWT_SECRET` and
  optional rotation values in `DEV_JWT_ADDITIONAL_SECRETS`. Secrets must decode
  to ≥32 bytes (base64url or hexadecimal). Development tokens are required to
  contain the same issuer, audience, subject, issued-at, not-before and expiry
  claims enforced in OIDC mode. HMAC secrets are ignored – and rejected – when
  `OPE_ENVIRONMENT=production`.

## Secret rotation

* **OIDC / JWKS rotations** happen at the identity provider. The application
  automatically refreshes JWKS documents and accepts both the current and
  previously seen keyset during the transition.
* **Development secrets** can be rotated by deploying with the new value in
  `DEV_JWT_SECRET` and the previous entries listed in
  `DEV_JWT_ADDITIONAL_SECRETS`. Once all clients are updated, remove the old
  values and redeploy. Using any development secret in production is treated as
  a deployment misconfiguration and the application will refuse to start.

## Transport security

* The API enforces HSTS (`Strict-Transport-Security`), `X-Content-Type-Options`
  and `X-Frame-Options` headers. Terminate TLS at your edge proxy and configure
  `OPE_ALLOWED_HOSTS` / `OPE_ALLOWED_ORIGINS` to the minimal set of domains
  required per environment. Wildcards are intentionally rejected in production.

## Least privilege & hardening

* Use FastAPI dependencies such as `require_permission` to guard routes with the
  minimum scopes required for each action.
* Place `/metrics` and `/docs` behind authenticated ingress in production.
* Run `pytest`, `ruff` and `mypy` locally (and ensure CI passes) before merging
  code changes. Keep dependencies patched by updating `pyproject.toml` on a
  regular cadence.
