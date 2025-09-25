"""Variance-reduction utilities for Monte Carlo pricing."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.random import Generator, SeedSequence
from scipy import stats
from scipy.stats import norm, qmc

from .models import MarketData, OptionContract, OptionType, PricingResult
from .pricing_models import (
    BlackScholesModel,
    MonteCarloModel,
    _apply_pathwise_control_variates,
)
from ..utils.validation import validate_pricing_parameters


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    """Configuration describing a single variance-reduction strategy."""

    name: str
    antithetic: bool = True
    control_variate: bool = True
    stratified: bool = False
    qmc: bool = False


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
    seed_sequence: Optional[SeedSequence] = None
    use_common_random_numbers: bool = True
    min_paths: int = 256
    strategies: Optional[Dict[str, StrategyConfig]] = None
    _seed_cache: Dict[Tuple[str, int], int] = field(default_factory=dict, init=False)
    _baseline_strategy: StrategyConfig = field(init=False)
    _root_seed: SeedSequence = field(init=False)

    def __post_init__(self) -> None:
        if self.seed_sequence is None:
            self.seed_sequence = SeedSequence(0)
        self._root_seed = self.seed_sequence
        if self.strategies is None:
            self.strategies = {
                "control_variate": StrategyConfig("control_variate"),
                "stratified_control": StrategyConfig(
                    "stratified_control", stratified=True
                ),
                "sobol_control": StrategyConfig("sobol_control", antithetic=False, qmc=True),
                "sobol_stratified_control": StrategyConfig(
                    "sobol_stratified_control", stratified=True, qmc=True
                ),
            }
        self._baseline_strategy = StrategyConfig(
            name="baseline",
            antithetic=self.baseline_model.antithetic,
            control_variate=False,
            stratified=False,
            qmc=False,
        )

    def price(
        self,
        contract: OptionContract,
        market_data: MarketData,
        volatility: float,
        *,
        target_ci_half_width: float,
        initial_paths: Optional[int] = None,
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
        initial_paths: Optional[int] = None,
        max_paths: int = 262_144,
    ) -> VarianceReductionReport:
        """Auto-select the most efficient strategy meeting the target half-width."""

        validate_pricing_parameters(contract, market_data, volatility)

        start_paths = int(max(self.min_paths, initial_paths or self.baseline_model.paths))
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

        best_report: Optional[VarianceReductionReport] = None
        baseline_paths = baseline_report.diagnostics.used_paths

        for strategy in self.strategies.values():
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
            if candidate.diagnostics.ci_half_width <= target_ci_half_width:
                if best_report is None or (
                    candidate.diagnostics.used_paths < best_report.diagnostics.used_paths
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

        if strategy_name not in self.strategies:
            raise KeyError(f"Unknown strategy '{strategy_name}'")

        validate_pricing_parameters(contract, market_data, volatility)

        target_paths = max(self.min_paths, int(paths))

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
            self.strategies[strategy_name],
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
        baseline_reference: Optional[VarianceReductionReport],
    ) -> VarianceReductionReport:
        paths = max(self.min_paths, int(start_paths))
        last_outcome: Optional[_SimulationOutcome] = None

        while paths <= max_paths:
            outcome = self._run_simulation(contract, market_data, volatility, strategy, paths)
            last_outcome = outcome
            if outcome.ci_half_width <= target_ci_half_width:
                diagnostics = self._build_diagnostics(outcome, baseline_reference or outcome)
                return VarianceReductionReport(outcome.result, diagnostics)
            paths *= 2

        if last_outcome is None:
            raise RuntimeError("Simulation did not execute")

        diagnostics = self._build_diagnostics(last_outcome, baseline_reference or last_outcome)
        return VarianceReductionReport(last_outcome.result, diagnostics)

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
        sobol_engine = self._sobol_engine(strategy, paths, rng)

        draws = self._generate_draws(paths, rng, strategy, sobol_engine)
        actual_paths = draws.size

        time_sqrt = math.sqrt(max(0.0, contract.time_to_expiry))
        drift = (
            market_data.risk_free_rate
            - market_data.dividend_yield
            - 0.5 * volatility**2
        ) * contract.time_to_expiry
        diffusion = volatility * time_sqrt * draws
        terminal_prices = market_data.spot_price * np.exp(drift + diffusion)

        if contract.option_type is OptionType.CALL:
            payoff = np.maximum(terminal_prices - contract.strike_price, 0.0)
        else:
            payoff = np.maximum(contract.strike_price - terminal_prices, 0.0)

        discount_factor = math.exp(-market_data.risk_free_rate * contract.time_to_expiry)
        discounted_payoffs = discount_factor * payoff

        adjusted_payoffs = discounted_payoffs
        cv_report: dict[str, float | bool | None] | None = None
        if strategy.control_variate:
            adjusted_payoffs, cv_report = _apply_pathwise_control_variates(
                discounted_payoffs,
                terminal_prices,
                contract=contract,
                market_data=market_data,
                volatility=volatility,
            )

        price = float(np.mean(adjusted_payoffs))
        standard_error: Optional[float]
        ci_half_width: float
        confidence_interval: Optional[Tuple[float, float]]

        if actual_paths > 1:
            sample_std = float(np.std(adjusted_payoffs, ddof=1))
            standard_error = sample_std / math.sqrt(actual_paths)
            z_score = norm.ppf(0.975)
            ci_half_width = z_score * standard_error
            confidence_interval = (
                price - ci_half_width,
                price + ci_half_width,
            )
        else:
            standard_error = None
            ci_half_width = float("inf")
            confidence_interval = None

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if adjusted_payoffs.size > 1:
            baseline = discounted_payoffs
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
            theoretical_price=max(0.0, price),
            computation_time_ms=elapsed_ms,
            model_used=f"{model_used}_{actual_paths}",
            implied_volatility=volatility,
            standard_error=standard_error,
            confidence_interval=confidence_interval,
            control_variate_report=cv_report,
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
            seed = self._seed_cache.setdefault(
                ("common", paths),
                int(self._root_seed.spawn(1)[0].generate_state(1)[0]),
            )
        else:
            seed = int(self._root_seed.spawn(1)[0].generate_state(1)[0])
        return np.random.default_rng(seed)

    def _sobol_engine(
        self, strategy: StrategyConfig, paths: int, rng: Generator
    ) -> Optional[qmc.Sobol]:
        if not strategy.qmc:
            return None
        if self.use_common_random_numbers:
            seed = self._seed_cache.setdefault(
                ("sobol", paths),
                int(self._root_seed.spawn(1)[0].generate_state(1)[0]),
            )
        else:
            seed = int(rng.integers(0, 2**32 - 1))
        return qmc.Sobol(d=1, scramble=True, seed=seed)

    def _generate_draws(
        self,
        paths: int,
        rng: Generator,
        strategy: StrategyConfig,
        sobol_engine: Optional[qmc.Sobol],
    ) -> np.ndarray:
        count = max(1, int(paths))
        if strategy.antithetic:
            count = max(2, count + (count % 2))
            base = count // 2
        else:
            base = count

        normals = self._base_normals(base, rng, strategy, sobol_engine)
        if strategy.antithetic:
            normals = np.concatenate([normals, -normals])

        if strategy.qmc:
            normals = normals - float(np.mean(normals))
            std = float(np.std(normals, ddof=0))
            if std > 0.0:
                normals = normals / std

        return normals.astype(float, copy=False)

    def _base_normals(
        self,
        count: int,
        rng: Generator,
        strategy: StrategyConfig,
        sobol_engine: Optional[qmc.Sobol],
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
        return norm.ppf(clipped)

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

        path_reduction = (
            baseline_paths / outcome.paths if outcome.paths else float("inf")
        )

        return VarianceReductionDiagnostics(
            strategy=outcome.strategy.name,
            used_paths=outcome.paths,
            ci_half_width=outcome.ci_half_width,
            baseline_paths=baseline_paths,
            baseline_half_width=baseline_half_width,
            path_reduction=path_reduction,
            bias_pvalue=outcome.bias_pvalue,
        )

