import importlib

import numpy as np
import pytest

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data

# the package __init__ rebinds the name "SEQuential" to the class, so import
# the module object explicitly to monkeypatch its module-level _outcome_fit.
seqmod = importlib.import_module("pySEQTarget.SEQuential")


def test_ITT_hazard():
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
        parameters=SEQopts(hazard_estimate=True),
    )
    s.expand()
    s.fit()
    s.hazard()


def test_bootstrap_hazard():
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
        parameters=SEQopts(hazard_estimate=True, bootstrap_nboot=2, seed=42),
    )
    s.expand()
    s.bootstrap()
    s.fit()
    s.hazard()


def test_subgroup_hazard():
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
        parameters=SEQopts(hazard_estimate=True, subgroup_colname="sex"),
    )
    s.expand()
    s.bootstrap()
    s.fit()
    s.hazard()


def test_hazard_survives_skipped_bootstrap_replicate(monkeypatch):
    # When a bootstrap replicate fails to fit (singular matrix -> LinAlgError),
    # fit() skips it, so outcome_model has fewer entries than _boot_samples.
    # hazard() must still pair each resample with its own model and not crash
    # with an IndexError. Regression for the glum-backend short-course failure.
    data = load_data("SEQdata")

    real_outcome_fit = seqmod._outcome_fit
    fail_on = 2  # _current_boot_idx of the replicate to fail (sample index 1)

    def flaky_outcome_fit(seq_self, *args, **kwargs):
        if getattr(seq_self, "_current_boot_idx", None) == fail_on:
            raise np.linalg.LinAlgError("A singular matrix detected: injected for test")
        return real_outcome_fit(seq_self, *args, **kwargs)

    monkeypatch.setattr(seqmod, "_outcome_fit", flaky_outcome_fit)

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
        parameters=SEQopts(bootstrap_nboot=4, seed=42, hazard_estimate=True),
    )
    s.expand()
    s.bootstrap()
    with pytest.warns(UserWarning):
        s.fit()

    # One replicate skipped: effective nboot drops and the failed sample index
    # (fail_on - 1) is absent from the success map.
    assert s.bootstrap_nboot == 3
    assert len(s.outcome_model) == 4  # main fit + 3 successful replicates
    assert s._boot_sample_idx == [0, 2, 3]

    # Must not raise IndexError and must produce a finite HR with CI.
    s.hazard()
    hr = s.hazard_ratio
    assert hr["Hazard ratio"][0] is not None and np.isfinite(hr["Hazard ratio"][0])
    assert hr["LCI"][0] is not None
    assert hr["UCI"][0] is not None
