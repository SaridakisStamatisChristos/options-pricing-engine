"""Deterministic replay dispatch for stochastic pricing capsules."""

from __future__ import annotations

from numbers import Real

from .models import ExerciseStyle, MarketData, OptionContract, OptionType, PricingResult
from .monte_carlo import MonteCarloModel
from .pricing_common import (
    MAX_MONTE_CARLO_PATHS,
    _bounded_integer,
    _require_boolean,
)
from .replay import ReplayCapsule


def replay_pricing_capsule(capsule: ReplayCapsule) -> PricingResult:
    """Re-run a pricing request described by ``capsule``."""
    if not isinstance(capsule, ReplayCapsule):
        raise TypeError("capsule must be a ReplayCapsule")
    if not capsule.verify_integrity():
        raise ValueError("Replay capsule integrity check failed")
    payload = capsule.payload
    unknown_top_level = set(payload) - {"model", "request", "seed", "surface_id"}
    if unknown_top_level:
        raise ValueError(
            f"Replay capsule contains unsupported entries: {sorted(unknown_top_level)}"
        )
    model_info = payload.get("model", {})
    request_info = payload.get("request", {})
    if not isinstance(model_info, dict) or not isinstance(request_info, dict):
        raise ValueError("Replay capsule model and request entries must be mappings")
    model_name = model_info.get("name")
    config = model_info.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("Replay capsule model config must be a mapping")
    if set(model_info) - {"name", "config"}:
        raise ValueError("Replay capsule model entry contains unsupported fields")
    if set(request_info) - {"contract", "market_data", "volatility"}:
        raise ValueError("Replay capsule request entry contains unsupported fields")

    contract_info = request_info.get("contract") or {}
    market_info = request_info.get("market_data") or {}
    if not isinstance(contract_info, dict) or not isinstance(market_info, dict):
        raise ValueError("Replay capsule contract and market data must be mappings")
    volatility_raw = request_info.get("volatility")
    if isinstance(volatility_raw, bool) or not isinstance(volatility_raw, Real):
        raise ValueError("Invalid volatility in replay capsule")
    volatility = float(volatility_raw)

    if set(contract_info) - {
        "symbol",
        "strike_price",
        "time_to_expiry",
        "option_type",
        "exercise_style",
        "contract_id",
    }:
        raise ValueError("Replay capsule contract contains unsupported fields")
    if set(market_info) - {"spot_price", "risk_free_rate", "dividend_yield"}:
        raise ValueError("Replay capsule market data contains unsupported fields")

    symbol = contract_info.get("symbol")
    strike = contract_info.get("strike_price")
    expiry = contract_info.get("time_to_expiry")
    option_type = contract_info.get("option_type")
    exercise_style = contract_info.get("exercise_style")
    contract_id = contract_info.get("contract_id", "")
    if (
        not isinstance(symbol, str)
        or isinstance(strike, bool)
        or not isinstance(strike, Real)
        or isinstance(expiry, bool)
        or not isinstance(expiry, Real)
        or not isinstance(option_type, str)
        or not isinstance(exercise_style, str)
        or not isinstance(contract_id, str)
    ):
        raise ValueError("Invalid contract parameters in replay capsule")

    try:
        contract = OptionContract(
            symbol=symbol,
            strike_price=float(strike),
            time_to_expiry=float(expiry),
            option_type=OptionType(option_type),
            exercise_style=ExerciseStyle(exercise_style),
            contract_id=contract_id,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid contract parameters in replay capsule") from exc

    spot = market_info.get("spot_price")
    rate = market_info.get("risk_free_rate")
    dividend = market_info.get("dividend_yield")

    def _replay_real(value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("Invalid market data in replay capsule")
        return float(value)

    spot_value = _replay_real(spot)
    rate_value = _replay_real(rate)
    dividend_value = _replay_real(dividend)
    try:
        market = MarketData(
            spot_price=spot_value,
            risk_free_rate=rate_value,
            dividend_yield=dividend_value,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid market data in replay capsule") from exc

    seed_sequence = capsule.resolve_seed_sequence()

    if model_name == "monte_carlo":
        if set(config) - {"paths", "antithetic", "use_control_variates"}:
            raise ValueError("Replay capsule model config contains unsupported fields")
        if seed_sequence is None:
            raise ValueError("Monte Carlo replay capsule does not contain a valid seed")
        paths = _bounded_integer(
            "capsule paths",
            config.get("paths", MonteCarloModel.DEFAULT_PATHS),
            minimum=1,
            maximum=MAX_MONTE_CARLO_PATHS,
        )
        antithetic = _require_boolean(
            "capsule antithetic",
            config.get("antithetic", MonteCarloModel.DEFAULT_ANTITHETIC),
        )
        use_control_variates = _require_boolean(
            "capsule use_control_variates",
            config.get("use_control_variates", True),
        )
        model = MonteCarloModel(
            paths=paths,
            antithetic=antithetic,
            use_control_variates=use_control_variates,
        )
        return model.calculate_price(contract, market, volatility, seed_sequence=seed_sequence)

    raise ValueError(f"Replay is not supported for model '{model_name}'")


__all__ = ["replay_pricing_capsule"]
