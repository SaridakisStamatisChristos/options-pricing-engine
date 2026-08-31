"""FastAPI route registrations for the minimal core API."""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, FastAPI, HTTPException, Response
from fastapi import Path as PathParam
from pydantic import ValidationError

from ..core.black_scholes import BlackScholesModel
from ..core.crr import BinomialModel
from ..core.dividends import CashDividend, CashDividendSchedule
from ..core.lsmc import american_lsmc_price
from ..core.models import ExerciseStyle, MarketData, OptionContract, OptionType, PricingResult
from ..core.monte_carlo import MonteCarloModel
from ..core.pricing_common import MAX_LSMC_STEPS, MAX_LSMC_WORK_ITEMS
from ..core.variance_reduction import (
    StrategyConfig,
    VarianceReductionToolkit,
)
from .capsule import (
    CAPSULE_STORE,
    DEFAULT_BUILD_ID,
    IDEMPOTENCY_CACHE,
    LSMC_BATCH_AGGREGATE_WORK_LIMIT,
    MC_BATCH_AGGREGATE_LIMIT,
    MC_MAX_PATHS,
    IdempotencyConflictError,
    build_capsule_record,
    derive_seed_lineage,
    lineage_to_seed_sequence,
)
from .codec import (
    CONFLICT_ERROR,
    COST_GUARD_ERROR,
    IDEMPOTENCY_CONFLICT_ERROR,
    NOT_FOUND_ERROR,
    UNSUPPORTED_ERROR,
    VALIDATION_ERROR,
    canonical_dumps,
    canonical_hash,
    canonical_response,
    http_error,
    safe_float,
)
from .legacy_schemas import (
    BatchRequest,
    BatchResponse,
    BatchResult,
    ConfidenceInterval,
    GreeksOnlyResponse,
    QuoteRequest,
    QuoteResponse,
    ReplayRequest,
    VersionResponse,
)
from .security import require_permission

LOGGER = logging.getLogger(__name__)


def _load_build_id() -> str:
    build_id = os.getenv("OPTIONS_ENGINE_BUILD_ID", DEFAULT_BUILD_ID).strip()
    if not 1 <= len(build_id) <= 128 or any(ord(character) < 32 for character in build_id):
        raise RuntimeError("OPTIONS_ENGINE_BUILD_ID must contain between 1 and 128 characters")
    return build_id


_DEFAULT_BUILD_ID = _load_build_id()
BUILD_ID = _DEFAULT_BUILD_ID


def _parse_env_flag(value: str | None) -> bool:
    if value is None:
        return False
    normalised = value.strip().lower()
    if normalised in {"1", "true", "yes", "on"}:
        return True
    if normalised in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError("MC_ENABLE_QMC must be a boolean")


MC_ENABLE_QMC = _parse_env_flag(os.getenv("MC_ENABLE_QMC"))


def _current_build_id() -> str:
    package = sys.modules.get("options_engine.api.routes")
    if package is not None:
        value = package.__dict__.get("BUILD_ID")
        if (
            isinstance(value, str)
            and 1 <= len(value) <= 128
            and not any(ord(character) < 32 for character in value)
        ):
            return value
    return _DEFAULT_BUILD_ID


OPTION_TYPE_MAP = {"CALL": OptionType.CALL, "PUT": OptionType.PUT}
EXERCISE_STYLE_MAP = {"EUROPEAN": ExerciseStyle.EUROPEAN, "AMERICAN": ExerciseStyle.AMERICAN}


@dataclass(frozen=True)
class MonteCarloPlan:
    """Description of the Monte Carlo configuration derived from a request."""

    paths: int
    antithetic: bool
    use_qmc: bool
    use_cv: bool


BLACK_SCHOLES = BlackScholesModel()


@lru_cache(maxsize=128)
def _get_binomial_model(steps: int) -> BinomialModel:
    return BinomialModel(steps=steps)


def _build_contract(request: QuoteRequest) -> OptionContract:
    payload = request.contract
    option_type = OPTION_TYPE_MAP[payload.option_type]
    exercise_style = EXERCISE_STYLE_MAP[payload.exercise_style]
    return OptionContract(
        symbol=payload.symbol,
        strike_price=payload.strike_price,
        time_to_expiry=payload.time_to_expiry,
        option_type=option_type,
        exercise_style=exercise_style,
    )


def _build_market(request: QuoteRequest) -> MarketData:
    market = request.market
    return MarketData(
        spot_price=market.spot_price,
        risk_free_rate=market.risk_free_rate,
        dividend_yield=market.dividend_yield,
        cash_dividends=CashDividendSchedule(
            tuple(
                CashDividend(ex_time=dividend.ex_time, amount=dividend.amount)
                for dividend in market.cash_dividends
            )
        ),
    )


def _greeks_filter(
    keys: Iterable[tuple[str, float | None]], request: QuoteRequest
) -> dict[str, float]:
    greeks_request = request.greeks
    include_all = greeks_request is None
    result: dict[str, float] = {}
    for name, value in keys:
        if value is None:
            continue
        if include_all or getattr(greeks_request, name, False):
            result[name] = safe_float(value)
    return result


def _plan_monte_carlo(request: QuoteRequest) -> MonteCarloPlan:
    params = request.model.params
    paths = params.paths if params and params.paths is not None else 20_000
    default_use_qmc = MC_ENABLE_QMC
    default_antithetic = not default_use_qmc

    antithetic = (
        default_antithetic
        if params is None or params.antithetic is None
        else bool(params.antithetic)
    )
    use_qmc = default_use_qmc if params is None or params.use_qmc is None else bool(params.use_qmc)
    use_cv = True if params is None or params.use_cv is None else bool(params.use_cv)

    if use_qmc:
        antithetic = False

    max_paths = (
        request.precision.max_paths
        if request.precision and request.precision.max_paths
        else MC_MAX_PATHS
    )
    if max_paths > MC_MAX_PATHS:
        raise http_error(COST_GUARD_ERROR, headers={"Retry-After": "1"})
    paths = min(paths, max_paths)
    if paths > MC_MAX_PATHS:
        raise http_error(COST_GUARD_ERROR, headers={"Retry-After": "1"})

    return MonteCarloPlan(paths=int(paths), antithetic=antithetic, use_qmc=use_qmc, use_cv=use_cv)


def _describe_vr_pipeline(plan: MonteCarloPlan) -> str:
    stages: list[str] = []
    if plan.use_qmc:
        stages.append("rqmc")
    elif plan.antithetic:
        stages.append("antithetic")
    if plan.use_cv:
        stages.append("cv")
    if not stages:
        return "baseline"
    return "+".join(stages)


def _confidence_interval(
    result: PricingResult,
    paths_used: int,
    *,
    vr_pipeline: str,
) -> ConfidenceInterval | None:
    if result.standard_error is None:
        return None
    diagnostics = result.estimate_diagnostics or {}
    raw_interval = diagnostics.get("raw_confidence_interval")
    if (
        isinstance(raw_interval, (tuple, list))
        and len(raw_interval) == 2
        and all(isinstance(value, (int, float)) for value in raw_interval)
    ):
        half_width_abs = 0.5 * abs(float(raw_interval[1]) - float(raw_interval[0]))
    else:  # Compatibility fallback for third-party PricingResult producers.
        half_width_abs = 1.96 * result.standard_error
    denominator = max(abs(result.theoretical_price), 1e-6)
    half_width_bps = 10_000.0 * half_width_abs / denominator
    interval = result.confidence_interval
    degrees_value = diagnostics.get("degrees_of_freedom")
    units_value = diagnostics.get("independent_units")
    return ConfidenceInterval(
        half_width_abs=safe_float(half_width_abs),
        half_width_bps=safe_float(half_width_bps),
        paths_used=int(paths_used),
        vr_pipeline=vr_pipeline,
        lower_bound=safe_float(interval[0]) if interval is not None else None,
        upper_bound=safe_float(interval[1]) if interval is not None else None,
        method=(
            str(diagnostics["interval_method"])
            if diagnostics.get("interval_method") is not None
            else None
        ),
        degrees_of_freedom=(
            int(degrees_value)
            if isinstance(degrees_value, (int, float)) and not isinstance(degrees_value, bool)
            else None
        ),
        independent_units=(
            int(units_value)
            if isinstance(units_value, (int, float)) and not isinstance(units_value, bool)
            else None
        ),
    )


def _log_request(
    *,
    start_time: float,
    request: QuoteRequest,
    model_family: str,
    capsule_id: str,
    ci: ConfidenceInterval | None,
) -> None:
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    spot = request.market.spot_price
    strike = request.contract.strike_price
    tau = request.contract.time_to_expiry
    moneyness = spot / strike if strike else float("inf")
    if moneyness < 0.8:
        money_bucket = "deep_otm"
    elif moneyness > 1.2:
        money_bucket = "deep_itm"
    else:
        money_bucket = "near_atm"
    if tau < 0.25:
        tau_bucket = "short"
    elif tau < 1.0:
        tau_bucket = "medium"
    else:
        tau_bucket = "long"
    ci_bps = ci.half_width_bps if ci else 0.0
    paths_used = ci.paths_used if ci else 0
    vr_pipeline = ci.vr_pipeline if ci else "none"
    LOGGER.info(
        "pricing.complete latency_ms=%.2f model_family=%s "
        "moneyness_bucket=%s tau_bucket=%s vr_pipeline=%s paths_used=%d "
        "ci_bps=%.4f capsule_id=%s",
        latency_ms,
        model_family,
        money_bucket,
        tau_bucket,
        vr_pipeline,
        paths_used,
        ci_bps,
        capsule_id,
    )


def _execute_quote(
    request: QuoteRequest,
    *,
    index: int,
    idempotency_key: str | None = None,
) -> tuple[dict[str, Any], str]:
    request_payload = request.model_dump(exclude_none=True)
    request_payload.pop("idempotency_key", None)
    request_hash = canonical_hash(request_payload)

    if idempotency_key:
        try:
            cached = IDEMPOTENCY_CACHE.get(idempotency_key, request_hash)
        except IdempotencyConflictError as exc:
            raise http_error(IDEMPOTENCY_CONFLICT_ERROR) from exc
        if cached is not None:
            return json_response_from_body(cached)

    start = time.perf_counter()
    try:
        contract = _build_contract(request)
        market = _build_market(request)
    except (TypeError, ValueError) as exc:  # pragma: no cover - guarded in tests
        raise http_error(VALIDATION_ERROR) from exc

    family = request.model.family
    params = request.model.params
    model_used: dict[str, Any] = {"family": family, "params": {}}
    seed_prefix = params.seed_prefix if params else None
    seed_lineage = derive_seed_lineage(seed_prefix=seed_prefix, base_hash=request_hash, index=index)

    if family == "black_scholes":
        if contract.exercise_style is not ExerciseStyle.EUROPEAN:
            raise http_error(VALIDATION_ERROR)
        result = BLACK_SCHOLES.calculate_price(contract, market, request.volatility)
        ci = None
    elif family == "binomial":
        steps = params.steps if params and params.steps is not None else 200
        model_used["params"]["steps"] = steps
        binomial_model = _get_binomial_model(int(steps))
        result = binomial_model.calculate_price(contract, market, request.volatility)
        ci = None
    elif family == "monte_carlo":
        plan = _plan_monte_carlo(request)
        target_ci_bps = request.precision.target_ci_bps if request.precision else None
        precision_max_paths = (
            request.precision.max_paths
            if request.precision and request.precision.max_paths is not None
            else MC_MAX_PATHS
        )
        if contract.exercise_style is ExerciseStyle.AMERICAN:
            if params is not None and params.use_qmc is True:
                raise http_error(UNSUPPORTED_ERROR)
            steps = params.steps if params and params.steps is not None else 64
            american_plan = MonteCarloPlan(
                paths=plan.paths,
                antithetic=(
                    True if params is None or params.antithetic is None else params.antithetic
                ),
                use_qmc=False,
                use_cv=plan.use_cv,
            )
            max_work_paths = MAX_LSMC_WORK_ITEMS // int(steps)
            if american_plan.antithetic and max_work_paths % 2:
                max_work_paths -= 1
            path_ceiling = min(int(precision_max_paths), max_work_paths)
            paths = american_plan.paths
            effective_paths = paths + (paths % 2) if american_plan.antithetic else paths
            if (
                steps > MAX_LSMC_STEPS
                or path_ceiling < 1
                or steps * effective_paths > MAX_LSMC_WORK_ITEMS
            ):
                raise http_error(COST_GUARD_ERROR, headers={"Retry-After": "1"})
            if params and params.seed_prefix:
                seed_material = f"{params.seed_prefix}:{seed_lineage}"
            else:
                seed_material = seed_lineage
            seed = int.from_bytes(
                hashlib.blake2b(seed_material.encode("utf-8"), digest_size=8).digest(), "big"
            )
            target_met = False
            while True:
                lsmc_result = american_lsmc_price(
                    spot=market.spot_price,
                    strike=contract.strike_price,
                    tau=contract.time_to_expiry,
                    sigma=request.volatility,
                    r=market.risk_free_rate,
                    q=market.dividend_yield,
                    option_type="call" if contract.option_type is OptionType.CALL else "put",
                    steps=int(steps),
                    paths=int(paths),
                    seed=int(seed),
                    antithetic=american_plan.antithetic,
                    use_cv=american_plan.use_cv,
                )
                price = safe_float(lsmc_result.price)
                half_width_abs = safe_float(lsmc_result.ci_half_width)
                denominator = max(abs(price), 1e-6)
                half_width_bps = safe_float(10_000.0 * half_width_abs / denominator)
                target_met = target_ci_bps is None or half_width_bps <= target_ci_bps
                if target_met or paths >= path_ceiling:
                    break
                next_paths = min(path_ceiling, paths * 2)
                if next_paths <= paths:
                    break
                paths = next_paths

            effective_paths = paths + (paths % 2) if american_plan.antithetic else paths
            resolved_plan = MonteCarloPlan(
                paths=effective_paths,
                antithetic=american_plan.antithetic,
                use_qmc=False,
                use_cv=american_plan.use_cv,
            )
            lsmc_diagnostics = lsmc_result.estimate_diagnostics or {}
            degrees_value = lsmc_diagnostics.get("degrees_of_freedom")
            units_value = lsmc_diagnostics.get("independent_units")
            ci = ConfidenceInterval(
                half_width_abs=half_width_abs,
                half_width_bps=half_width_bps,
                paths_used=effective_paths,
                vr_pipeline=_describe_vr_pipeline(resolved_plan),
                lower_bound=(
                    safe_float(lsmc_result.confidence_interval[0])
                    if lsmc_result.confidence_interval is not None
                    else None
                ),
                upper_bound=(
                    safe_float(lsmc_result.confidence_interval[1])
                    if lsmc_result.confidence_interval is not None
                    else None
                ),
                method=(
                    str(lsmc_diagnostics["interval_method"])
                    if lsmc_diagnostics.get("interval_method") is not None
                    else None
                ),
                degrees_of_freedom=(
                    int(degrees_value)
                    if isinstance(degrees_value, (int, float))
                    and not isinstance(degrees_value, bool)
                    else None
                ),
                independent_units=(
                    int(units_value)
                    if isinstance(units_value, (int, float)) and not isinstance(units_value, bool)
                    else None
                ),
            )
            result = PricingResult(
                contract_id=contract.contract_id,
                theoretical_price=price,
                standard_error=lsmc_result.standard_error,
                confidence_interval=lsmc_result.confidence_interval,
                estimate_diagnostics=lsmc_result.estimate_diagnostics,
                model_used="american_lsmc",
            )
            model_used["params"].update(
                {
                    "paths": int(paths),
                    "steps": int(steps),
                    "antithetic": american_plan.antithetic,
                    "use_qmc": False,
                    "use_cv": american_plan.use_cv,
                }
            )
            model_used["meta"] = lsmc_result.meta
            if target_ci_bps is not None:
                model_used["precision"] = {
                    "target_ci_bps": target_ci_bps,
                    "target_met": target_met,
                    "max_paths": path_ceiling,
                }
        else:
            seed_sequence = lineage_to_seed_sequence(seed_lineage)
            paths = plan.paths
            target_met = False
            while True:
                resolved_plan = MonteCarloPlan(
                    paths=paths,
                    antithetic=plan.antithetic,
                    use_qmc=plan.use_qmc,
                    use_cv=plan.use_cv,
                )
                if plan.use_qmc:
                    strategy = StrategyConfig(
                        "requested_rqmc",
                        antithetic=False,
                        control_variate=plan.use_cv,
                        qmc=True,
                    )
                    baseline = MonteCarloModel(
                        paths=paths,
                        antithetic=False,
                        seed_sequence=seed_sequence,
                        use_control_variates=False,
                    )
                    toolkit = VarianceReductionToolkit(
                        baseline_model=baseline,
                        seed_sequence=seed_sequence,
                        strategies={strategy.name: strategy},
                    )
                    report = toolkit.run_strategy(
                        strategy.name,
                        paths,
                        contract,
                        market,
                        request.volatility,
                    )
                    result = report.pricing_result
                    paths_used = report.diagnostics.used_paths
                else:
                    monte_carlo_model = MonteCarloModel(
                        paths=paths,
                        antithetic=plan.antithetic,
                        seed_sequence=seed_sequence,
                        use_control_variates=plan.use_cv,
                    )
                    result = monte_carlo_model.calculate_price(
                        contract,
                        market,
                        request.volatility,
                        seed_sequence=seed_sequence,
                    )
                    paths_used = paths + (paths % 2) if plan.antithetic else paths
                ci = _confidence_interval(
                    result,
                    paths_used=paths_used,
                    vr_pipeline=_describe_vr_pipeline(resolved_plan),
                )
                target_met = (
                    target_ci_bps is None or ci is None or ci.half_width_bps <= target_ci_bps
                )
                if target_met or paths >= precision_max_paths:
                    break
                next_paths = min(int(precision_max_paths), paths * 2)
                if next_paths <= paths:
                    break
                paths = next_paths

            model_used["params"].update(
                {
                    "paths": paths_used,
                    "antithetic": plan.antithetic,
                    "use_qmc": plan.use_qmc,
                    "use_cv": plan.use_cv,
                }
            )
            if target_ci_bps is not None:
                model_used["precision"] = {
                    "target_ci_bps": target_ci_bps,
                    "target_met": target_met,
                    "max_paths": int(precision_max_paths),
                }
    else:  # pragma: no cover - defensive programming
        raise http_error(VALIDATION_ERROR)

    greeks = _greeks_filter(
        (
            ("delta", result.delta),
            ("gamma", result.gamma),
            ("vega", result.vega),
            ("theta", result.theta),
            ("rho", result.rho),
        ),
        request,
    )

    response_payload: dict[str, Any] = {
        "theoretical_price": safe_float(result.theoretical_price),
        "model_used": model_used,
        "capsule_id": "",
    }
    if request.surface is not None:
        response_payload["surface_id"] = request.surface.resolved_id()
    if greeks:
        response_payload["greeks"] = greeks
    if ci is not None:
        response_payload["ci"] = ci.model_dump()
    if result.estimate_diagnostics is not None:
        response_payload["estimate_diagnostics"] = result.estimate_diagnostics
    response_payload["seed_lineage"] = seed_lineage

    capsule_record = build_capsule_record(
        request_payload=request_payload,
        response_payload=response_payload,
        model_used=model_used,
        seed_lineage=seed_lineage,
        build_id=_current_build_id(),
    )
    response_payload["capsule_id"] = capsule_record.capsule_id
    CAPSULE_STORE.save(capsule_record)

    _log_request(
        start_time=start,
        request=request,
        model_family=family,
        capsule_id=capsule_record.capsule_id,
        ci=ci,
    )

    body = canonical_dumps(response_payload)
    if idempotency_key:
        try:
            IDEMPOTENCY_CACHE.put(idempotency_key, request_hash, body)
        except IdempotencyConflictError as exc:
            raise http_error(IDEMPOTENCY_CONFLICT_ERROR) from exc

    return response_payload, body


def json_response_from_body(body: str) -> tuple[dict[str, Any], str]:
    payload = QuoteResponse.model_validate_json(body).model_dump()
    return payload, body


def register_routes(app: FastAPI) -> None:
    router = APIRouter()

    @router.post(
        "/quote",
        dependencies=[Depends(require_permission("pricing:read"))],
    )
    def quote_endpoint(request: QuoteRequest) -> Any:
        _, body = _execute_quote(request, index=0, idempotency_key=request.idempotency_key)
        return Response(content=body, media_type="application/json")

    @router.post(
        "/batch",
        dependencies=[Depends(require_permission("pricing:read"))],
    )
    def batch_endpoint(request: BatchRequest) -> Any:
        if len(request.items) > 100:
            raise http_error(COST_GUARD_ERROR, headers={"Retry-After": "1"})

        greeks_default = request.greeks_default.model_dump() if request.greeks_default else None
        parsed_items: list[QuoteRequest | None] = []

        for raw_item in request.items:
            payload = dict(raw_item)
            if greeks_default and "greeks" not in payload:
                payload["greeks"] = greeks_default
            try:
                parsed_item = QuoteRequest.model_validate(payload)
            except ValidationError:
                parsed_items.append(None)
            else:
                parsed_items.append(parsed_item)

        aggregate_paths = 0
        aggregate_lsmc_work = 0
        for candidate in parsed_items:
            if candidate is not None and candidate.model.family == "monte_carlo":
                plan = _plan_monte_carlo(candidate)
                has_precision_target = bool(
                    candidate.precision and candidate.precision.target_ci_bps is not None
                )
                potential_paths = (
                    candidate.precision.max_paths
                    if has_precision_target
                    and candidate.precision
                    and candidate.precision.max_paths is not None
                    else MC_MAX_PATHS
                    if has_precision_target
                    else plan.paths
                )
                aggregate_paths += int(potential_paths)
                if candidate.contract.exercise_style == "AMERICAN":
                    params = candidate.model.params
                    steps = params.steps if params and params.steps is not None else 64
                    work_paths = min(
                        int(potential_paths),
                        MAX_LSMC_WORK_ITEMS // int(steps),
                    )
                    antithetic = (
                        True if params is None or params.antithetic is None else params.antithetic
                    )
                    effective_paths = work_paths + (work_paths % 2) if antithetic else work_paths
                    aggregate_lsmc_work += int(steps) * effective_paths
        if (
            aggregate_paths > MC_BATCH_AGGREGATE_LIMIT
            or aggregate_lsmc_work > LSMC_BATCH_AGGREGATE_WORK_LIMIT
        ):
            raise http_error(COST_GUARD_ERROR, headers={"Retry-After": "1"})

        results: list[BatchResult] = []
        capsule_ids: list[str] = []
        for index, batch_item in enumerate(parsed_items):
            if batch_item is None:
                results.append(BatchResult(index=index, ok=False, error=VALIDATION_ERROR.detail))
                continue
            try:
                payload, _ = _execute_quote(batch_item, index=index)
                results.append(BatchResult(index=index, ok=True, value=QuoteResponse(**payload)))
                capsule_ids.append(payload["capsule_id"])
            except HTTPException as exc:
                results.append(BatchResult(index=index, ok=False, error=str(exc.detail)))

        response_payload = BatchResponse(results=results, capsule_ids=capsule_ids).model_dump()
        return canonical_response(response_payload)

    @router.post(
        "/greeks",
        dependencies=[Depends(require_permission("pricing:read"))],
    )
    def greeks_endpoint(request: QuoteRequest) -> Any:
        request.greeks = request.greeks or None
        payload, _body = _execute_quote(request, index=0)
        greeks = payload.get("greeks") or {}
        response = GreeksOnlyResponse(
            greeks=greeks,
            capsule_id=payload["capsule_id"],
            model_used=payload["model_used"],
        ).model_dump()
        return canonical_response(response)

    @router.get("/version")
    def version_endpoint() -> Any:
        import numpy
        import scipy

        payload = VersionResponse(
            build_id=_current_build_id(),
            library_versions={"numpy": numpy.__version__, "scipy": scipy.__version__},
            flags={"randomized_qmc": True, "stratified": True, "cv": True},
        ).model_dump()
        return canonical_response(payload)

    @router.post(
        "/replay/{capsule_id}",
        dependencies=[Depends(require_permission("pricing:read"))],
    )
    def replay_endpoint(
        capsule_id: Annotated[
            str,
            PathParam(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"),
        ],
        request: Annotated[ReplayRequest | None, Body()] = None,
    ) -> Any:
        record = CAPSULE_STORE.get(capsule_id)
        if record is None:
            raise http_error(NOT_FOUND_ERROR)
        strict = request.strict_build if request else False
        if strict and record.build_id != _current_build_id():
            raise http_error(CONFLICT_ERROR)
        payload = dict(record.response_payload)
        payload["capsule_id"] = record.capsule_id
        payload["replayed"] = True
        return canonical_response(payload)

    app.include_router(router)
