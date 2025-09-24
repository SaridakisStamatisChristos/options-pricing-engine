# Security Guide

This document summarises the security posture and operational controls for the
Options Pricing Engine.

## Authentication

* All endpoints require a valid JWT signed with `HS256`.
* The signing secret is mandatory (`OPE_JWT_SECRET`) and must be at least 32
  bytes of entropy. The service will refuse to start without it.
* Tokens must include the following claims: `sub`, `iat`, `nbf`, `exp`, `aud`
  and `iss`. Missing or malformed claims result in HTTP 401.
* Token expiry defaults to 30 minutes (`OPE_JWT_EXP_MINUTES`). Adjust to your
  risk tolerance and rotate secrets regularly.

## Secret rotation

1. Provision a new secret.
2. Deploy with the new secret in `OPE_JWT_SECRET` and the previous secrets in
   `OPE_JWT_ADDITIONAL_SECRETS`.
3. Once all issuers use the new secret, remove the old values from
   `OPE_JWT_ADDITIONAL_SECRETS` and redeploy.

## Transport security

* The API adds HSTS (`Strict-Transport-Security`) and clickjacking protections
  (`X-Frame-Options`) by default. Terminate TLS at your edge proxy.
* Configure `OPE_ALLOWED_HOSTS` and `OPE_ALLOWED_ORIGINS` to the minimal set of
  domains needed per environment. Wildcards are intentionally not allowed in
  production.

## Least privilege

* The in-memory user store provided is a stub. Integrate with your identity
  provider and enforce scoped permissions via `require_permission`.
* Limit exposure of `/docs` and `/metrics` in production by placing them behind
  authenticated ingress where appropriate.

## Secure development checklist

* Run `pytest`, `ruff` and `mypy` before merging.
* Keep dependencies patched by periodically updating `pyproject.toml`.
* Review new routes for proper authentication dependencies and input
  validation.
