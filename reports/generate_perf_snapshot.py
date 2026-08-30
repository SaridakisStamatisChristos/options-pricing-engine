"""Generate the Monte Carlo/Black-Scholes latency snapshot used for regressions."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from numpy.random import SeedSequence

import options_engine.core.pricing_models as pricing_models
from options_engine.core.models import ExerciseStyle, MarketData, OptionContract, OptionType
from options_engine.core.pricing_models import BlackScholesModel, MonteCarloModel

EXPECTED_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
CONTRACT = OptionContract("PERF", 105.0, 0.5, OptionType.CALL, ExerciseStyle.EUROPEAN)
MARKET = MarketData(spot_price=100.0, risk_free_rate=0.01, dividend_yield=0.0)
VOL = 0.25
PATHS = 8000


def _ensure_vectorized_engine() -> None:
    module_path = Path(pricing_models.__file__).resolve()
    if not module_path.is_relative_to(EXPECTED_SOURCE_ROOT):
        raise SystemExit(
            "Perf snapshot must run against the vectorized engine; "
            f"imported module at {str(module_path)!r}."
        )


def _time(fn: Callable[[], None]) -> float:
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1e3


def _run_monte_carlo(paths: int = PATHS) -> None:
    MonteCarloModel(
        paths=paths,
        antithetic=True,
        use_control_variates=False,
        seed_sequence=SeedSequence(2028),
    ).calculate_price(CONTRACT, MARKET, VOL)


def _snapshot_runs(num_runs: int = 10) -> Sequence[float]:
    _run_monte_carlo()  # warm-up
    runs = sorted(_time(_run_monte_carlo) for _ in range(num_runs))
    return runs


def main() -> int:
    _ensure_vectorized_engine()

    BlackScholesModel().calculate_price(CONTRACT, MARKET, VOL)
    _run_monte_carlo()

    bs_ms = _time(lambda: BlackScholesModel().calculate_price(CONTRACT, MARKET, VOL))
    mc_runs = list(_snapshot_runs())
    payload = {
        "black_scholes_ms": bs_ms,
        "monte_carlo_ms_p50": mc_runs[len(mc_runs) // 2],
        "monte_carlo_ms_runs": mc_runs,
    }
    output_path = Path(__file__).with_name("perf_snapshot.json")
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    resolved = output_path.resolve()
    try:
        display_path = resolved.relative_to(Path.cwd())
    except ValueError:
        display_path = resolved
    print(f"Wrote {display_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
