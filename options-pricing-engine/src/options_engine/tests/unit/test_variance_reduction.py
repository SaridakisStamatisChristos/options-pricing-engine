import math

from numpy.random import SeedSequence

from options_engine.core.models import MarketData, OptionContract, OptionType
from options_engine.core.pricing_models import MonteCarloModel
from options_engine.core.variance_reduction import VarianceReductionToolkit


def _build_contract(option_type: OptionType = OptionType.CALL) -> OptionContract:
    return OptionContract(
        symbol="VR",
        strike_price=100.0,
        time_to_expiry=1.0,
        option_type=option_type,
    )


def _build_market() -> MarketData:
    return MarketData(spot_price=105.0, risk_free_rate=0.02, dividend_yield=0.01)


def test_control_variates_reduce_half_width_against_baseline() -> None:
    root_seed = SeedSequence(202701)
    baseline_seed, toolkit_seed = root_seed.spawn(2)

    baseline_model = MonteCarloModel(paths=16_384, seed_sequence=baseline_seed)
    toolkit = VarianceReductionToolkit(baseline_model=baseline_model, seed_sequence=toolkit_seed)

    contract = _build_contract()
    market = _build_market()

    report = toolkit.run_strategy("control_variate", 4_096, contract, market, 0.25)

    assert report.diagnostics.ci_half_width < report.diagnostics.baseline_half_width
    assert report.diagnostics.bias_pvalue >= 0.05


def test_qmc_strategy_reproducible_with_common_random_numbers() -> None:
    root_seed = SeedSequence(998877)
    baseline_seed, toolkit_seed = root_seed.spawn(2)

    baseline_model = MonteCarloModel(paths=8_192, seed_sequence=baseline_seed)
    toolkit = VarianceReductionToolkit(baseline_model=baseline_model, seed_sequence=toolkit_seed)

    contract = _build_contract(option_type=OptionType.PUT)
    market = _build_market()

    first = toolkit.run_strategy("sobol_stratified_control", 2_048, contract, market, 0.35)
    second = toolkit.run_strategy("sobol_stratified_control", 2_048, contract, market, 0.35)

    assert math.isclose(
        first.pricing_result.theoretical_price,
        second.pricing_result.theoretical_price,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        first.diagnostics.ci_half_width,
        second.diagnostics.ci_half_width,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_variance_reduction_auto_meets_target_with_fewer_paths() -> None:
    root_seed = SeedSequence(424242)
    baseline_seed, toolkit_seed = root_seed.spawn(2)

    baseline_model = MonteCarloModel(paths=65_536, seed_sequence=baseline_seed)
    toolkit = VarianceReductionToolkit(baseline_model=baseline_model, seed_sequence=toolkit_seed)

    contract = _build_contract()
    market = _build_market()

    target_half_width = 0.12
    report = toolkit.price_with_diagnostics(
        contract,
        market,
        0.2,
        target_ci_half_width=target_half_width,
        max_paths=262_144,
    )

    assert report.diagnostics.ci_half_width <= target_half_width
    assert report.diagnostics.path_reduction >= 4.0
    assert report.diagnostics.bias_pvalue >= 0.05
