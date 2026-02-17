import os

import numpy as np
import pytest

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


def _make_seq(seed, **extra_opts):
    data = load_data("SEQdata")
    return SEQuential(
        data,
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="tx_init",
        outcome_col="outcome",
        time_varying_cols=["N", "L", "P"],
        fixed_cols=["sex"],
        method="ITT",
        parameters=SEQopts(seed=seed, **extra_opts),
    )


def test_hazard_reproducible_with_seed():
    results = []
    for _ in range(2):
        s = _make_seq(seed=42, hazard_estimate=True)
        s.expand()
        s.fit()
        s.hazard()
        results.append(s.hazard_ratio)

    assert results[0]["Hazard ratio"][0] == results[1]["Hazard ratio"][0]


def test_hazard_bootstrap_se_reproducible_with_seed():
    results = []
    for _ in range(2):
        s = _make_seq(seed=42, hazard_estimate=True, bootstrap_nboot=3)
        s.expand()
        s.bootstrap()
        s.fit()
        s.hazard()
        results.append(s.hazard_ratio)

    assert results[0]["Hazard ratio"][0] == results[1]["Hazard ratio"][0]
    assert results[0]["LCI"][0] == results[1]["LCI"][0]
    assert results[0]["UCI"][0] == results[1]["UCI"][0]


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Bootstrap reproducibility test hangs in CI"
)
def test_hazard_bootstrap_percentile_reproducible_with_seed():
    results = []
    for _ in range(2):
        s = _make_seq(
            seed=42,
            hazard_estimate=True,
            bootstrap_nboot=3,
            bootstrap_CI_method="percentile",
        )
        s.expand()
        s.bootstrap()
        s.fit()
        s.hazard()
        results.append(s.hazard_ratio)

    assert results[0]["Hazard ratio"][0] == results[1]["Hazard ratio"][0]
    assert results[0]["LCI"][0] == results[1]["LCI"][0]
    assert results[0]["UCI"][0] == results[1]["UCI"][0]


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Reproducibility test hangs in CI"
)
def test_survival_reproducible_with_seed():
    results = []
    for _ in range(2):
        s = _make_seq(seed=42, km_curves=True)
        s.expand()
        s.fit()
        s.survival()
        results.append(s.km_data)

    np.testing.assert_allclose(
        results[0]["pred"].to_numpy(), results[1]["pred"].to_numpy(), atol=1e-14
    )


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Bootstrap reproducibility test hangs in CI"
)
def test_survival_bootstrap_reproducible_with_seed():
    results = []
    for _ in range(2):
        s = _make_seq(seed=42, km_curves=True, bootstrap_nboot=3)
        s.expand()
        s.bootstrap()
        s.fit()
        s.survival()
        results.append(s.km_data)

    for col in ["pred", "SE", "LCI", "UCI"]:
        np.testing.assert_allclose(
            results[0][col].to_numpy(), results[1][col].to_numpy(), atol=1e-14
        )
