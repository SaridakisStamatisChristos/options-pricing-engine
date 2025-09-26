"""FastAPI route registrations for the minimal core API."""

from __future__ import annotations

import hashlib
import logging
import os
import time
import sys
from dataclasses import dataclass
from importlib import util
from pathlib import Path as SystemPath
from typing import Any, Dict, Iterable, List, Optional, Tuple

from fastapi import APIRouter, Body, Depends, FastAPI, HTTPException, Path as PathParam, Response
from pydantic import ValidationError

from ..core.models import ExerciseStyle, MarketData, OptionContract, OptionType, PricingResult
from ..core.pricing_models import (
    BinomialModel,
    BlackScholesModel,
    MonteCarloModel,
    american_lsmc_price,
)
from .capsule import (
    CAPSULE_STORE,
    DEFAULT_BUILD_ID,
    IDEMPOTENCY_CACHE,
    MC_BATCH_AGGREGATE_LIMIT,
    MC_MAX_PATHS,
    build_capsule_record,
    derive_seed_lineage,
    lineage_to_seed_sequence,
)
from .codec import (
    CONFLICT_ERROR,
    COST_GUARD_ERROR,
    NOT_FOUND_ERROR,
    VALIDATION_ERROR,
    canonical_dumps,
    canonical_hash,
    canonical_response,
    http_error,
    safe_float,
)
from .security import require_permission


def _load_schemas_module():
    module_path = SystemPath(__file__).with_name("schemas.py")
    spec = util.spec_from_file_location("options_engine.api._minimal_schemas", module_path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError("Failed to load schemas module")
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_schemas = _load_schemas_module()
BatchRequest = _schemas.BatchRequest
BatchResponse = _schemas.BatchResponse
BatchResult = _schemas.BatchResult
ConfidenceInterval = _schemas.ConfidenceInterval
GreeksOnlyResponse = _schemas.GreeksOnlyResponse
QuoteRequest = _schemas.QuoteRequest
QuoteResponse = _schemas.QuoteResponse
ReplayRequest = _schemas.ReplayRequest
VersionResponse = _schemas.VersionResponse


LOGGER = logging.getLogger(__name__)

_DEFAULT_BUILD_ID = os.getenv("OPTIONS_ENGINE_BUILD_ID", DEFAULT_BUILD_ID)
BUILD_ID = _DEFAULT_BUILD_ID


def _parse_env_flag(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


MC_ENABLE_QMC = _parse_env_flag(os.getenv("MC_ENABLE_QMC"))


def _current_build_id() -> str:
    package = sys.modules.get("options_engine.api.routes")
    if package is not None:
        value = package.__dict__.get("BUILD_ID")
        if isinstance(value, str):
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
BINOMIAL_CACHE: Dict[int, BinomialModel] = {}


def _get_binomial_model(steps: int) -> BinomialModel:
    model = BINOMIAL_CACHE.get(steps)
    if model is None:
        model = BinomialModel(steps=steps)
        BINOMIAL_CACHE[steps] = model
    return model


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
    )


def _greeks_filter(keys: Iterable[Tuple[str, Optional[float]]], request: QuoteRequest) -> Dict[str, float]:
    greeks_request = request.greeks
    include_all = greeks_request is None
    result: Dict[str, float] = {}
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
    default_antithetic = False if default_use_qmc else True

    antithetic = (
        default_antithetic if params is None or params.antithetic is None else bool(params.antithetic)
    )
    use_qmc = (
        default_use_qmc if params is None or params.use_qmc is None else bool(params.use_qmc)
    )
    use_cv = True if params is None or params.use_cv is None else bool(params.use_cv)

    if use_qmc:
        antithetic = False

    max_paths = request.precision.max_paths if request.precision and request.precision.max_paths else MC_MAX_PATHS
    if max_paths > MC_MAX_PATHS:
        raise http_error(COST_GUARD_ERROR, headers={"Retry-After": "1"})
    paths = min(paths, max_paths)
    if paths > MC_MAX_PATHS:
        raise http_error(COST_GUARD_ERROR, headers={"Retry-After": "1"})

    return MonteCarloPlan(paths=int(paths), antithetic=antithetic, use_qmc=use_qmc, use_cv=use_cv)


def _describe_vr_pipeline(plan: MonteCarloPlan) -> str:
    stages: List[str] = []
    if plan.use_qmc:
        stages.append("qmc")
    elif plan.antithetic:
        stages.append("antithetic")
    if plan.use_cv:
        stages.append("cv")
    if not stages:
        return "baseline"
    return "+".join(stages)


def _confidence_interval(price: float, standard_error: Optional[float], paths_used: int, *, vr_pipeline: str) -> ConfidenceInterval | None:
    if standard_error is None:
        return None
    half_width_abs = 1.96 * standard_error
    denominator = max(abs(price), 1e-6)
    half_width_bps = 10_000.0 * half_width_abs / denominator
    return ConfidenceInterval(
        half_width_abs=safe_float(half_width_abs),
        half_width_bps=safe_float(half_width_bps),
        paths_used=int(paths_used),
        vr_pipeline=vr_pipeline,
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
    message = (
        f"latency_ms={latency_ms:.2f} model_family={model_family} "
        f"moneyness_bucket={money_bucket} tau_bucket={tau_bucket} "
        f"vr_pipeline={vr_pipeline} paths_used={paths_used} ci_bps={ci_bps:.4f} "
        f"capsule_id={capsule_id}"
    )
    print(message)


def _execute_quote(
    request: QuoteRequest,
    *,
    index: int,
    idempotency_key: Optional[str] = None,
) -> Tuple[Dict[str, Any], str]:
    request_payload = request.model_dump(exclude_none=True)
    request_payload.pop("idempotency_key", None)
    request_hash = canonical_hash(request_payload)

    if idempotency_key:
        cached = IDEMPOTENCY_CACHE.get(idempotency_key, request_hash)
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
    model_used: Dict[str, Any] = {"family": family, "params": {}}
    seed_prefix = params.seed_prefix if params else None
    seed_lineage = derive_seed_lineage(seed_prefix=seed_prefix, base_hash=request_hash, index=index)

    if family == "black_scholes":
        result = BLACK_SCHOLES.calculate_price(contract, market, request.volatility)
        ci = None
    elif family == "binomial":
        steps = params.steps if params and params.steps is not None else 200
        model_used["params"]["steps"] = steps
        model = _get_binomial_model(int(steps))
        result = model.calculate_price(contract, market, request.volatility)
        ci = None
    elif family == "monte_carlo":
        plan = _plan_monte_carlo(request)
        if contract.exercise_style is ExerciseStyle.AMERICAN:
            steps = params.steps if params and params.steps is not None else 64
            paths = plan.paths
            if params and params.seed_prefix:
                seed_material = f"{params.seed_prefix}:{seed_lineage}"
            else:
                seed_material = seed_lineage
            seed = int.from_bytes(hashlib.blake2b(seed_material.encode("utf-8"), digest_size=8).digest(), "big")
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
                antithetic=True,
                use_cv=True,
            )
            price = safe_float(lsmc_result.price)
            half_width_abs = safe_float(lsmc_result.ci_half_width)
            denominator = max(abs(price), 1e-6)
            half_width_bps = safe_float(10_000.0 * half_width_abs / denominator)
            ci = ConfidenceInterval(
                half_width_abs=half_width_abs,
                half_width_bps=half_width_bps,
                paths_used=int(paths),
                vr_pipeline=_describe_vr_pipeline(
                    MonteCarloPlan(paths=int(paths), antithetic=True, use_qmc=False, use_cv=True)
                ),
            )
            result = PricingResult(
                contract_id=contract.contract_id,
                theoretical_price=price,
                standard_error=lsmc_result.standard_error,
                model_used="american_lsmc",
            )
            model_used["params"].update(
                {
                    "paths": int(paths),
                    "steps": int(steps),
                    "antithetic": True,
                    "use_qmc": False,
                    "use_cv": True,
                }
            )
            model_used["meta"] = lsmc_result.meta
        else:
            model_used["params"].update(
                {
                    "paths": plan.paths,
                    "antithetic": plan.antithetic,
                    "use_qmc": plan.use_qmc,
                    "use_cv": plan.use_cv,
                }
            )
            seed_sequence = lineage_to_seed_sequence(seed_lineage)
            model = MonteCarloModel(
                paths=plan.paths,
                antithetic=plan.antithetic,
                seed_sequence=seed_sequence,
                use_control_variates=plan.use_cv,
            )
            result = model.calculate_price(
                contract,
                market,
                request.volatility,
                seed_sequence=seed_sequence,
            )
            ci = _confidence_interval(
                result.theoretical_price,
                result.standard_error,
                paths_used=plan.paths,
                vr_pipeline=_describe_vr_pipeline(plan),
            )
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

    response_payload: Dict[str, Any] = {
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
        IDEMPOTENCY_CACHE.put(idempotency_key, request_hash, body)

    return response_payload, body


def json_response_from_body(body: str) -> Tuple[Dict[str, Any], str]:
    payload = QuoteResponse.model_validate_json(body).model_dump()
    return payload, body


def register_routes(app: FastAPI) -> None:
    router = APIRouter()

    @router.get("/healthz")
    def healthz() -> Any:
        """Simple unauthenticated health endpoint."""

        return canonical_response({"status": "ok"})

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
        parsed_items: List[QuoteRequest | None] = []

        for raw_item in request.items:
            payload = dict(raw_item)
            if greeks_default and "greeks" not in payload:
                payload["greeks"] = greeks_default
            try:
                item = QuoteRequest.model_validate(payload)
            except ValidationError:
                parsed_items.append(None)
            else:
                parsed_items.append(item)

        aggregate_paths = 0
        for item in parsed_items:
            if item is not None and item.model.family == "monte_carlo":
                plan = _plan_monte_carlo(item)
                aggregate_paths += plan.paths
        if aggregate_paths > MC_BATCH_AGGREGATE_LIMIT:
            raise http_error(COST_GUARD_ERROR, headers={"Retry-After": "1"})

        results: List[BatchResult] = []
        capsule_ids: List[str] = []
        for index, item in enumerate(parsed_items):
            if item is None:
                results.append(BatchResult(index=index, ok=False, error=VALIDATION_ERROR.detail))
                continue
            try:
                payload, _ = _execute_quote(item, index=index)
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
        payload, body = _execute_quote(request, index=0)
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
            flags={"qmc": False, "stratified": False, "cv": True},
        ).model_dump()
        return canonical_response(payload)

    @router.post(
        "/replay/{capsule_id}",
        dependencies=[Depends(require_permission("pricing:read"))],
    )
    def replay_endpoint(
        capsule_id: str = PathParam(...), request: ReplayRequest = Body(default=ReplayRequest())
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

