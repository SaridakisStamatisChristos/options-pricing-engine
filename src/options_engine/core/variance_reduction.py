"""Variance-reduction utilities for Monte Carlo pricing."""

from __future__ import annotations

import hashlib
import math
import threading
import time
from dataclasses import dataclass, field
from numbers import Integral, Real

import numpy as np
from numpy.random import Generator, SeedSequence
from scipy import stats
from scipy.stats import norm, qmc

from ..utils.validation import validate_pricing_parameters
from .black_scholes import BlackScholesModel
from .models import ExerciseStyle, MarketData, OptionContract, OptionType, PricingResult
from .monte_carlo import MonteCarloModel
from .pricing_common import (
    MAX_MONTE_CARLO_PATHS,
    _antithetic_units,
    _apply_pathwise_control_variates,
)
from .statistical_inference import estimate_mean


def _bounded_integer(name: str, value: object, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalised = int(value)
    if not minimum <= normalised <= maximum:
        raise ValueError(f"{name} must be within [{minimum}, {maximum}]")
    return normalised


def _positive_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalised = float(value)
    if not math.isfinite(normalised) or normalised <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return normalised


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    """Configuration describing a single variance-reduction strategy."""

    name: str
    antithetic: bool = True
    control_variate: bool = True
    stratified: bool = False
    qmc: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("strategy name must be a string")
        normalized_name = self.name.strip()
        if (
            not normalized_name
            or len(normalized_name) > 64
            or any(ord(character) < 32 or ord(character) == 127 for character in normalized_name)
        ):
            raise ValueError("strategy name must contain between 1 and 64 characters")
        object.__setattr__(self, "name", normalized_name)
        for field_name in ("antithetic", "control_variate", "stratified", "qmc"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a boolean")
        if self.qmc and self.stratified:
            raise ValueError("a strategy cannot be both QMC and stratified")
        if (self.qmc or self.stratified) and self.antithetic:
            raise ValueError("randomized-replicate strategies do not support antithetic pairing")


@dataclass(slots=True)
class VarianceReductionDiagnostics:
    """Diagnostics summarising the impact of a variance-reduction run."""

    strategy: str
    used_paths: int
    ci_half_width: float
    baseline_paths: int
    baseline_half_width: float
    path_reduction: float
    bias_pvalue: float


@dataclass(slots=True)
class VarianceReductionReport:
    """Container tying the pricing result with variance-reduction diagnostics."""

    pricing_result: PricingResult
    diagnostics: VarianceReductionDiagnostics


@dataclass(slots=True)
class _SimulationOutcome:
    """Internal helper capturing a single simulation run."""

    result: PricingResult
    strategy: StrategyConfig
    paths: int
    ci_half_width: float
    bias_pvalue: float


@dataclass(slots=True)
class VarianceReductionToolkit:
    """Toolkit combining multiple variance-reduction techniques."""

    baseline_model: MonteCarloModel = field(default_factory=MonteCarloModel)
    black_scholes_model: BlackScholesModel = field(default_factory=BlackScholesModel)
    seed_sequence: SeedSequence | None = None
    use_common_random_numbers: bool = True
    min_paths: int = 256
    strategies: dict[str, StrategyConfig] | None = None
    _baseline_strategy: StrategyConfig = field(init=False)
    _root_seed: SeedSequence = field(init=False)
    _seed_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.baseline_model, MonteCarloModel):
            raise TypeError("baseline_model must be a MonteCarloModel")
        if not isinstance(self.black_scholes_model, BlackScholesModel):
            raise TypeError("black_scholes_model must be a BlackScholesModel")
        if self.seed_sequence is None:
            self.seed_sequence = self.baseline_model.seed_sequence or SeedSequence(0)
        elif not isinstance(self.seed_sequence, SeedSequence):
            raise TypeError("seed_sequence must be a numpy.random.SeedSequence or None")
        if not isinstance(self.use_common_random_numbers, bool):
            raise TypeError("use_common_random_numbers must be a boolean")
        self.min_paths = _bounded_integer(
            "min_paths",
            self.min_paths,
            minimum=32,
            maximum=MAX_MONTE_CARLO_PATHS,
        )
        self._root_seed = self.seed_sequence
        if self.strategies is None:
            self.strategies = {
                "control_variate": StrategyConfig("control_variate"),
                "stratified_control": StrategyConfig(
                    "stratified_control", antithetic=False, stratified=True
                ),
                "sobol_control": StrategyConfig("sobol_control", antithetic=False, qmc=True),
            }
        elif not isinstance(self.strategies, dict) or not self.strategies:
            raise ValueError("strategies must be a non-empty dictionary")
        if len(self.strategies) > 32:
            raise ValueError("at most 32 variance-reduction strategies are supported")
        for name, strategy in self.strategies.items():
            if not isinstance(name, str) or not isinstance(strategy, StrategyConfig):
                raise TypeError("strategy entries must map strings to StrategyConfig values")
            if name != strategy.name:
                raise ValueError("strategy dictionary keys must match StrategyConfig names")
        self.strategies = dict(self.strategies)
        self._baseline_strategy = StrategyConfig(
            name="baseline",
            antithetic=self.baseline_model.antithetic,
            control_variate=False,
            stratified=False,
            qmc=False,
        )

    def _configured_strategies(self) -> dict[str, StrategyConfig]:
        if self.strategies is None:  # pragma: no cover - initialized above
            raise RuntimeError("variance-reduction strategies were not initialized")
        return self.strategies

    def price(
        self,
        contract: OptionContract,
        market_data: MarketData,
        volatility: float,
        *,
        target_ci_half_width: float,
        initial_paths: int | None = None,
        max_paths: int = 262_144,
    ) -> PricingResult:
        """Return the variance-reduced price matching the requested confidence interval."""

        report = self.price_with_diagnostics(
            contract,
            market_data,
            volatility,
            target_ci_half_width=target_ci_half_width,
            initial_paths=initial_paths,
            max_paths=max_paths,
        )
        return report.pricing_result

    def price_with_diagnostics(
        self,
        contract: OptionContract,
        market_data: MarketData,
        volatility: float,
        *,
        target_ci_half_width: float,
        initial_paths: int | None = None,
        max_paths: int = 262_144,
    ) -> VarianceReductionReport:
        """Auto-select the most efficient strategy meeting the target half-width."""

        validate_pricing_parameters(contract, market_data, volatility)
        if contract.exercise_style is not ExerciseStyle.EUROPEAN:
            raise ValueError(
                "Variance-reduction terminal simulation supports European exercise only"
            )
        target_ci_half_width = _positive_finite("target_ci_half_width", target_ci_half_width)
        max_paths = _bounded_integer(
            "max_paths", max_paths, minimum=self.min_paths, maximum=MAX_MONTE_CARLO_PATHS
        )
        if initial_paths is not None:
            initial_paths = _bounded_integer(
                "initial_paths",
                initial_paths,
                minimum=1,
                maximum=max_paths,
            )

        start_paths = min(
            max_paths,
            max(self.min_paths, initial_paths or self.baseline_model.paths),
        )
        baseline_report = self._find_paths(
            contract,
            market_data,
            volatility,
            self._baseline_strategy,
            start_paths,
            target_ci_half_width,
            max_paths,
            baseline_reference=None,
        )

        best_report: VarianceReductionReport | None = None
        baseline_paths = baseline_report.diagnostics.used_paths

        for strategy in self._configured_strategies().values():
            start = max(self.min_paths, baseline_paths // 8)
            candidate = self._find_paths(
                contract,
                market_data,
                volatility,
                strategy,
                start,
                target_ci_half_width,
                max_paths,
                baseline_reference=baseline_report,
            )
            if candidate.diagnostics.ci_half_width <= target_ci_half_width and (
                candidate.diagnostics.bias_pvalue >= 0.05
                and (
                    best_report is None
                    or candidate.diagnostics.used_paths < best_report.diagnostics.used_paths
                )
            ):
                best_report = candidate

        if best_report is not None:
            return best_report

        return baseline_report

    def run_strategy(
        self,
        strategy_name: str,
        paths: int,
        contract: OptionContract,
        market_data: MarketData,
        volatility: float,
    ) -> VarianceReductionReport:
        """Execute a specific variance-reduction strategy for inspection."""

        strategies = self._configured_strategies()
        if strategy_name not in strategies:
            raise KeyError(f"Unknown strategy '{strategy_name}'")

        validate_pricing_parameters(contract, market_data, volatility)
        if contract.exercise_style is not ExerciseStyle.EUROPEAN:
            raise ValueError(
                "Variance-reduction terminal simulation supports European exercise only"
            )

        target_paths = max(
            self.min_paths,
            _bounded_integer("paths", paths, minimum=1, maximum=MAX_MONTE_CARLO_PATHS),
        )

        baseline_outcome = self._run_simulation(
            contract,
            market_data,
            volatility,
            self._baseline_strategy,
            target_paths,
        )
        outcome = self._run_simulation(
            contract,
            market_data,
            volatility,
            strategies[strategy_name],
            target_paths,
        )

        diagnostics = self._build_diagnostics(outcome, baseline_outcome)
        return VarianceReductionReport(outcome.result, diagnostics)

    def _find_paths(
        self,
        contract: OptionContract,
        market_data: MarketData,
        volatility: float,
        strategy: StrategyConfig,
        start_paths: int,
        target_ci_half_width: float,
        max_paths: int,
        baseline_reference: VarianceReductionReport | None,
    ) -> VarianceReductionReport:
        paths = max(self.min_paths, start_paths)

        while True:
            outcome = self._run_simulation(contract, market_data, volatility, strategy, paths)
            if outcome.ci_half_width <= target_ci_half_width or paths >= max_paths:
                diagnostics = self._build_diagnostics(outcome, baseline_reference or outcome)
                return VarianceReductionReport(outcome.result, diagnostics)
            paths = min(max_paths, paths * 2)

    def _run_simulation(
        self,
        contract: OptionContract,
        market_data: MarketData,
        volatility: float,
        strategy: StrategyConfig,
        paths: int,
    ) -> _SimulationOutcome:
        start = time.perf_counter()
        rng = self._rng(strategy, paths)
        replicate_payoffs: np.ndarray | None = None
        replicate_raw_payoffs: np.ndarray | None = None
        randomized_replicates = strategy.qmc or strategy.stratified
        if randomized_replicates:
            replicate_count = 16
            requested_per_replicate = max(2, int(paths) // replicate_count)
            if strategy.qmc:
                exponent = max(1, math.floor(math.log2(requested_per_replicate)))
                points_per_replicate = 2**exponent
            else:
                exponent = 0
                points_per_replicate = requested_per_replicate
            uniforms: list[np.ndarray] = []
            for _ in range(replicate_count):
                if strategy.qmc:
                    scramble_seed = int(rng.integers(0, 2**32 - 1))
                    engine = qmc.Sobol(d=1, scramble=True, seed=scramble_seed)
                    uniforms.append(engine.random_base2(exponent).reshape(-1))
                else:
                    uniforms.append(
                        (np.arange(points_per_replicate) + rng.random(points_per_replicate))
                        / points_per_replicate
                    )
            uniform_matrix = np.vstack(uniforms)
            draws_matrix = norm.ppf(np.clip(uniform_matrix, 1e-12, 1.0 - 1e-12))
            draws = draws_matrix.reshape(-1)
        else:
            draws = self._generate_draws(paths, rng, strategy, None)
            replicate_count = 1
            points_per_replicate = draws.size
        actual_paths = int(draws.size)

        time_sqrt = math.sqrt(max(0.0, contract.time_to_expiry))
        drift = (
            market_data.risk_free_rate - market_data.dividend_yield - 0.5 * volatility**2
        ) * contract.time_to_expiry
        diffusion = volatility * time_sqrt * draws
        with np.errstate(over="ignore", invalid="ignore"):
            terminal_prices = market_data.spot_price * np.exp(drift + diffusion)
        if not np.isfinite(terminal_prices).all():
            raise ValueError("variance-reduction simulation exceeds the floating-point range")

        if contract.option_type is OptionType.CALL:
            payoff = np.maximum(terminal_prices - contract.strike_price, 0.0)
        else:
            payoff = np.maximum(contract.strike_price - terminal_prices, 0.0)

        discount_factor = math.exp(-market_data.risk_free_rate * contract.time_to_expiry)
        discounted_payoffs = discount_factor * payoff

        adjusted_payoffs = discounted_payoffs
        cv_report: dict[str, object] | None = None
        if randomized_replicates:
            raw_matrix = discounted_payoffs.reshape(replicate_count, points_per_replicate)
            terminal_matrix = terminal_prices.reshape(replicate_count, points_per_replicate)
            adjusted_rows: list[np.ndarray] = []
            reports: list[dict[str, object]] = []
            for raw_row, terminal_row in zip(raw_matrix, terminal_matrix, strict=False):
                if strategy.control_variate:
                    adjusted_row, row_report = _apply_pathwise_control_variates(
                        raw_row,
                        terminal_row,
                        contract=contract,
                        market_data=market_data,
                        volatility=volatility,
                        antithetic=False,
                    )
                    reports.append(row_report)
                else:
                    adjusted_row = raw_row
                adjusted_rows.append(adjusted_row)
            adjusted_matrix = np.vstack(adjusted_rows)
            adjusted_payoffs = adjusted_matrix.reshape(-1)
            replicate_payoffs = np.mean(adjusted_matrix, axis=1)
            replicate_raw_payoffs = np.mean(raw_matrix, axis=1)
            if reports:
                residual_variances = [
                    float(value)
                    for report in reports
                    if isinstance((value := report.get("residual_var")), (int, float))
                ]
                raw_variances = [
                    float(value)
                    for report in reports
                    if isinstance((value := report.get("raw_var")), (int, float))
                ]
                correlations = [
                    float(value)
                    for report in reports
                    if isinstance((value := report.get("rho")), (int, float))
                ]
                betas = [report.get("beta") for report in reports if report.get("beta")]
                cv_report = {
                    "cv_used": any(bool(report.get("cv_used")) for report in reports),
                    "rho": float(np.median(correlations)) if correlations else None,
                    "beta": betas[0] if betas else None,
                    "raw_var": float(np.median(raw_variances)) if raw_variances else None,
                    "residual_var": (
                        float(np.median(residual_variances)) if residual_variances else None
                    ),
                }
        elif strategy.control_variate:
            adjusted_payoffs, cv_report = _apply_pathwise_control_variates(
                discounted_payoffs,
                terminal_prices,
                contract=contract,
                market_data=market_data,
                volatility=volatility,
                antithetic=strategy.antithetic,
            )
        else:
            adjusted_payoffs = _antithetic_units(discounted_payoffs, antithetic=strategy.antithetic)

        if randomized_replicates and replicate_payoffs is not None:
            inference_sample = replicate_payoffs
            estimator_name = "independent_randomized_qmc_replicates"
        else:
            inference_sample = adjusted_payoffs
            estimator_name = "terminal_payoff_sample_mean"

        inference = estimate_mean(inference_sample, lower_bound=0.0)
        price = inference.bounded_estimate
        standard_error = inference.standard_error
        confidence_interval = inference.confidence_interval
        raw_half_width = inference.raw_half_width
        ci_half_width = raw_half_width if raw_half_width is not None else float("inf")

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if (
            randomized_replicates
            and replicate_payoffs is not None
            and replicate_raw_payoffs is not None
        ):
            _, p_value = stats.ttest_rel(replicate_payoffs, replicate_raw_payoffs)
            if not math.isfinite(p_value):
                p_value = 1.0
        elif adjusted_payoffs.size > 1:
            baseline = _antithetic_units(discounted_payoffs, antithetic=strategy.antithetic)
            _, p_value = stats.ttest_rel(adjusted_payoffs, baseline)
            if not math.isfinite(p_value):
                p_value = 1.0
        else:
            p_value = 1.0

        components = [strategy.name]
        if strategy.qmc:
            components.append("sobol")
        if strategy.stratified:
            components.append("stratified")
        if strategy.control_variate:
            components.append("cv")
        if strategy.antithetic:
            components.append("antithetic")
        model_used = "vr_" + "_".join(components)

        result = PricingResult(
            contract_id=contract.contract_id,
            theoretical_price=price,
            computation_time_ms=elapsed_ms,
            model_used=f"{model_used}_{actual_paths}",
            implied_volatility=volatility,
            standard_error=standard_error,
            confidence_interval=confidence_interval,
            control_variate_report=cv_report,
            estimate_diagnostics=inference.diagnostics(estimator=estimator_name),
        )

        return _SimulationOutcome(
            result=result,
            strategy=strategy,
            paths=actual_paths,
            ci_half_width=ci_half_width,
            bias_pvalue=float(p_value),
        )

    def _rng(self, strategy: StrategyConfig, paths: int) -> Generator:
        if self.use_common_random_numbers:
            # Derive rather than spawn so results depend only on the root seed
            # and workload, never on prior call order or cache hits.
            state = self._root_seed.generate_state(8, dtype=np.uint32).tobytes()
            digest = hashlib.blake2b(
                state + paths.to_bytes(8, "big", signed=False),
                digest_size=8,
            )
            seed = int.from_bytes(digest.digest(), "big")
        else:
            with self._seed_lock:
                seed = int(self._root_seed.spawn(1)[0].generate_state(1)[0])
        return np.random.default_rng(seed)

    def _generate_draws(
        self,
        paths: int,
        rng: Generator,
        strategy: StrategyConfig,
        sobol_engine: qmc.Sobol | None,
    ) -> np.ndarray:
        count = _bounded_integer("paths", paths, minimum=1, maximum=MAX_MONTE_CARLO_PATHS)
        if strategy.antithetic:
            count = max(2, count + (count % 2))
            base = count // 2
        else:
            base = count

        normals = self._base_normals(base, rng, strategy, sobol_engine)
        if strategy.antithetic:
            normals = np.concatenate([normals, -normals])

        return normals.astype(float, copy=False)

    def _base_normals(
        self,
        count: int,
        rng: Generator,
        strategy: StrategyConfig,
        sobol_engine: qmc.Sobol | None,
    ) -> np.ndarray:
        if sobol_engine is not None:
            uniforms = sobol_engine.random(count).reshape(-1)
        else:
            if strategy.stratified:
                edges = np.linspace(0.0, 1.0, count + 1)
                widths = edges[1:] - edges[:-1]
                uniforms = edges[:-1] + widths * rng.random(count)
            else:
                uniforms = rng.random(count)
        clipped = np.clip(uniforms, 1e-12, 1 - 1e-12)
        return np.asarray(norm.ppf(clipped), dtype=float)

    def _build_diagnostics(
        self,
        outcome: _SimulationOutcome,
        baseline: _SimulationOutcome | VarianceReductionReport,
    ) -> VarianceReductionDiagnostics:
        if isinstance(baseline, VarianceReductionReport):
            baseline_paths = baseline.diagnostics.used_paths
            baseline_half_width = baseline.diagnostics.ci_half_width
        else:
            baseline_paths = baseline.paths
            baseline_half_width = baseline.ci_half_width

        path_reduction = baseline_paths / outcome.paths if outcome.paths else float("inf")

        return VarianceReductionDiagnostics(
            strategy=outcome.strategy.name,
            used_paths=outcome.paths,
            ci_half_width=outcome.ci_half_width,
            baseline_paths=baseline_paths,
            baseline_half_width=baseline_half_width,
            path_reduction=path_reduction,
            bias_pvalue=outcome.bias_pvalue,
        )
