# Security Policy

## Supported versions

Security fixes are applied to the current `2.x` line. The historical `1.x`
layout is unsupported.

## Report a vulnerability

Use GitHub's **Security → Report a vulnerability** private advisory flow for
this repository. Do not open a public issue containing exploit details, tokens,
private keys, or customer data. Include affected versions, reproduction steps,
impact, and any proposed mitigation. You should receive an acknowledgement
within five business days.

## Authentication contract

- Production startup requires `OIDC_ISSUER`, `OIDC_AUDIENCE`, and an HTTPS
  `OIDC_JWKS_URL`.
- JWT signature, issuer, audience, `exp`, `nbf`, `iat`, `sub`, and header `kid`
  are validated with configurable clock skew (60 seconds by default).
- Only allow-listed asymmetric algorithms are accepted. The untrusted token
  header algorithm must exactly match the selected JWK's algorithm; JWK `use`
  and `key_ops` must permit signature verification. RSA verification keys must
  be at least 2048 bits.
- JWKS requests have bounded timeouts, reject redirects, require JSON, and run
  outside the ASGI event loop. Previously validated keys provide a bounded
  availability grace period during provider failure.
- Routes enforce explicit scopes through FastAPI dependencies.
- A development HMAC fallback is available only outside production, requires
  issuer/audience validation, and requires at least 32 decoded secret bytes.
  Any `DEV_JWT_*` value makes production startup fail.

## Network and HTTP controls

- Terminate TLS at a trusted ingress. HSTS, `nosniff`, frame denial, and a
  restrictive referrer policy are returned by the application.
- Configure the ingress to overwrite, not append, forwarding headers. The
  application rate limiter uses the visible client address and must not trust
  arbitrary client-supplied forwarding headers.
- Production requires an explicit `Host` allow-list. Configure exact CORS
  origins; do not combine wildcard origins with credentials.
- Request bodies are counted while streaming and rejected as soon as they cross
  `MAX_BODY_BYTES`, including requests without `Content-Length`.
- Protect or restrict `/metrics`, `/healthz`, and documentation at the ingress
  according to your disclosure requirements.

## Secrets and key rotation

Never commit bearer tokens or JWT keys. Inject secrets from the deployment
platform and rotate at the identity provider:

1. Publish the new JWK while retaining the old JWK.
2. Issue new tokens using the new `kid`.
3. Wait at least the maximum token lifetime plus JWKS cache TTL.
4. Remove the old JWK after authentication metrics are clean.

For local HMAC rotation only, make the new value `DEV_JWT_SECRET` and temporarily
list old values in `DEV_JWT_ADDITIONAL_SECRETS`. Unprefixed values are literal
UTF-8; binary values require an explicit `base64:` or `hex:` prefix. Encoding is
never guessed, and every decoded secret must contain 32–4096 bytes.

## Container and dependency controls

The supplied image:

- installs the locked production dependency graph only;
- uses a multi-stage build with no compiler in the runtime image;
- runs as an unprivileged numeric UID;
- supports a read-only root filesystem, all capabilities dropped, and
  `no-new-privileges` in Compose;
- runs one process per container to avoid inconsistent security/rate-limit state.

CI runs Ruff, MyPy, Bandit, `pip-audit` against both the locked and minimum
supported dependency graphs, CodeQL, and a high/critical Trivy image scan.
Dependabot covers Python, GitHub Actions, and Docker dependencies. A clean scan
reduces known risk but is not proof of absence of vulnerabilities.

## Threat-model boundaries

- This service does not provide durable idempotency or replay storage.
- In-memory rate limiting is per replica and applies one quota across protected
  routes for each visible client. Enforce identity-aware or global quotas at a
  trusted API gateway when required.
- A task timeout cannot terminate a Python thread already executing native or
  Python numerical code.
- Pricing results are not authorization decisions and must not be treated as a
  trusted market data source.
- Logs intentionally avoid raw bearer tokens and full request payloads. Keep
  reverse-proxy access logs under the same rule.

## Local security verification

```bash
uv sync --locked --extra test --extra quality
uv run bandit -q -r src -ll
uv export --quiet --locked --no-dev --no-emit-project --format requirements.txt --output-file /tmp/options-engine-runtime.txt
uv run pip-audit --strict --require-hashes --requirement /tmp/options-engine-runtime.txt
uv run pytest tests/security tests/unit/test_authentication_helpers.py
```
