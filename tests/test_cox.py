import pytest
from pytest import approx

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


def _hazard(cox_package, nboot=0, **opts):
    data = load_data("SEQdata")
    # seed always set: the hazard step simulates outcomes, so a fixed seed is
    # required to isolate the backend difference from the random simulation.
    params = dict(hazard_estimate=True, cox_package=cox_package, seed=42, **opts)
    if nboot:
        params["bootstrap_nboot"] = nboot
    s = SEQuential(
        data,
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="tx_init",
        outcome_col="outcome",
        time_varying_cols=["N", "L", "P"],
        fixed_cols=["sex"],
        method="ITT",
        parameters=SEQopts(**params),
    )
    s.expand()
    if nboot:
        s.bootstrap()
    s.fit()
    s.hazard()
    return s


def test_cox_package_invalid_raises():
    with pytest.raises(ValueError, match="cox_package"):
        SEQopts(cox_package="survival")


def test_sksurv_matches_lifelines_ITT():
    # Both backends fit the same univariate Cox partial likelihood with Efron
    # tie handling, so the point hazard ratio should agree closely.
    ll = _hazard("lifelines").hazard_ratio
    sk = _hazard("scikit-survival").hazard_ratio
    assert sk["Hazard ratio"][0] == approx(ll["Hazard ratio"][0], rel=1e-3, abs=1e-3)


def test_sksurv_matches_lifelines_bootstrap():
    ll = _hazard("lifelines", nboot=5).hazard_ratio
    sk = _hazard("scikit-survival", nboot=5).hazard_ratio
    for col in ("Hazard ratio", "LCI", "UCI"):
        assert sk[col][0] == approx(ll[col][0], rel=1e-3, abs=1e-3)


def test_sksurv_subgroup_hazard_runs():
    data = load_data("SEQdata")
    s = SEQuential(
        data,
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="tx_init",
        outcome_col="outcome",
        time_varying_cols=["N", "L", "P"],
        fixed_cols=["sex"],
        method="ITT",
        parameters=SEQopts(
            hazard_estimate=True,
            cox_package="scikit-survival",
            subgroup_colname="sex",
        ),
    )
    s.expand()
    s.fit()
    s.hazard()
    assert s.hazard_ratio["Hazard ratio"].is_finite().all()
