"""Backward-compatible façade for pricing model implementations.

New code may import the focused modules directly. Existing imports from
``options_engine.core.pricing_models`` remain stable.
"""

from .black_scholes import BlackScholesModel, _black_scholes_greeks, black_scholes_price
from .crr import BinomialModel, binomial_price
from .lsmc import (
    BasisMetrics,
    ExercisePolicyStep,
    LongstaffSchwartzModel,
    LSMCAnalysis,
    american_lsmc_price,
)
from .monte_carlo import MonteCarloModel
from .pricing_common import (
    MAX_BINOMIAL_STEPS,
    MAX_CV_FOLDS,
    MAX_LSMC_PATHS,
    MAX_LSMC_STEPS,
    MAX_LSMC_WORK_ITEMS,
    MAX_MONTE_CARLO_PATHS,
    MAX_RANDOM_SEED,
    PriceResult,
    _antithetic_units,
    _apply_pathwise_control_variates,
    _bounded_integer,
    _cross_fitted_control_variate,
    _require_boolean,
)
from .replay_pricing import replay_pricing_capsule

__all__ = [
    "MAX_BINOMIAL_STEPS",
    "MAX_CV_FOLDS",
    "MAX_LSMC_PATHS",
    "MAX_LSMC_STEPS",
    "MAX_LSMC_WORK_ITEMS",
    "MAX_MONTE_CARLO_PATHS",
    "MAX_RANDOM_SEED",
    "BasisMetrics",
    "BinomialModel",
    "BlackScholesModel",
    "ExercisePolicyStep",
    "LSMCAnalysis",
    "LongstaffSchwartzModel",
    "MonteCarloModel",
    "PriceResult",
    "_antithetic_units",
    "_apply_pathwise_control_variates",
    "_black_scholes_greeks",
    "_bounded_integer",
    "_cross_fitted_control_variate",
    "_require_boolean",
    "american_lsmc_price",
    "binomial_price",
    "black_scholes_price",
    "replay_pricing_capsule",
]
