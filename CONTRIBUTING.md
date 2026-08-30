# Contributing

## Development setup

```bash
git clone https://github.com/SaridakisStamatisChristos/options-pricing-engine.git
cd options-pricing-engine
uv sync --locked --extra dev
```

Create a focused branch and keep numerical changes separate from unrelated
formatting or API changes when possible.

## Required checks

```bash
uv run ruff format .
uv run ruff check .
uv run mypy src
uv run bandit -q -r src -ll
uv run pytest -m "not performance" --cov=options_engine
uv build
```

Performance smoke tests are opt-in:

```bash
RUN_PERFORMANCE_TESTS=1 uv run pytest -m performance
```

## Numerical-change expectations

A pricing or calibration change should include:

1. the market/model assumptions being changed;
2. an independent oracle where available (closed form, high-resolution tree,
   limiting case, parity, or arbitrage inequality);
3. stress cases near expiry, extreme moneyness, low/high volatility, dividends,
   and negative rates where relevant;
4. statistical tolerances based on reported standard errors for Monte Carlo
   tests—never a fixed tolerance selected for one seed;
5. replay/seed behavior and cache-identity tests for stochastic changes;
6. updated method documentation and release notes when public behavior changes.

Do not make tests pass by clipping prices to a benchmark, shrinking confidence
intervals, clamping invalid requests into a different contract, or reading an
oracle inside the estimator under test.

## API and security changes

- Preserve the canonical `/api/v1` schemas unless a versioned migration is
  documented.
- Add a test for body limits, authentication failure mapping, and permission
  checks when changing middleware or routes.
- Never include tokens, secrets, full JWKS documents, or customer payloads in
  fixtures or logs.
- Use GitHub's private vulnerability reporting flow for security issues.

## Pull requests

Explain the problem, the chosen tradeoff, verification evidence, and any
remaining limitation. CI must pass on Python 3.11, 3.12, and 3.13. By
contributing, you agree that your contribution is licensed under Apache-2.0.
