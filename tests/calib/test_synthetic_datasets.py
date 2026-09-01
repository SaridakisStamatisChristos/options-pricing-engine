"""Deterministic recovery and adversarial board fixtures."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from options_engine.calib.datasets import (
    AdversarialBoard,
    SyntheticBoard,
    adversarial_board,
    heston_recovery_board,
    sabr_recovery_board,
    ssvi_recovery_board,
)


@pytest.mark.parametrize(
    "factory", [sabr_recovery_board, heston_recovery_board, ssvi_recovery_board]
)
def test_recovery_boards_are_byte_for_byte_deterministic(
    factory: Callable[..., SyntheticBoard],
) -> None:
    first = factory(noise_amplitude=0.001)
    second = factory(noise_amplitude=0.001)
    assert first.board.to_records() == second.board.to_records()
    assert first.parameters == second.parameters
    assert first.noise_amplitude == 0.001


def test_heston_fixture_documents_cross_family_generation() -> None:
    fixture = heston_recovery_board()
    assert fixture.generator == "Fang-Oosterlee COS"
    assert fixture.board.quotes.groupby("tenor").size().tolist() == [11, 11, 11]
    assert np.isfinite(fixture.board.quotes["mid_iv"]).all()


def test_ssvi_fixture_has_increasing_atm_total_variance() -> None:
    quotes = ssvi_recovery_board().board.quotes
    atm = (
        quotes.iloc[(np.log(quotes["strike"] / quotes["forward"])).abs().argsort()]
        .groupby("tenor", sort=True)
        .head(1)
        .sort_values("tenor")
    )
    total_variance = (atm["mid_iv"] ** 2 * atm["tenor"]).to_numpy()
    assert np.all(np.diff(total_variance) >= 0.0)


@pytest.mark.parametrize("case", list(AdversarialBoard))
def test_adversarial_boards_are_finite_and_labeled(case: AdversarialBoard) -> None:
    first = adversarial_board(case)
    second = adversarial_board(case)
    assert first.qc["adversarial"] == case.value
    assert first.to_records() == second.to_records()
    assert np.isfinite(
        first.quotes[["tenor", "strike", "forward", "mid_iv"]].to_numpy(dtype=float)
    ).all()


def test_noise_amplitude_is_bounded() -> None:
    with pytest.raises(ValueError, match="noise_amplitude"):
        sabr_recovery_board(noise_amplitude=0.1)
