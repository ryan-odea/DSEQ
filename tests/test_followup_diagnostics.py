import polars as pl

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


def _build(method, **opts):
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
        method=method,
        parameters=SEQopts(**opts),
    )
    s.expand()
    return s


def test_followup_tables_present_and_accessible():
    s = _build("ITT")
    for key in ("unique_followup", "nonunique_followup"):
        assert key in s.diagnostics

    s.fit()
    out = s.collect()
    for key in ("unique_followup", "nonunique_followup"):
        tbl = out.retrieve_data(key)
        assert "tx_init_bas" in tbl.columns
        assert "len" in tbl.columns


def test_nonunique_followup_counts_outcome_fit_rows_per_arm():
    # Non-unique follow-up == follow-up intervals (rows) the outcome model is
    # fit on, grouped by baseline treatment.
    s = _build("ITT")
    expected = s.DT.group_by("tx_init_bas").len().sort("tx_init_bas")
    assert s.diagnostics["nonunique_followup"].equals(expected)


def test_censoring_followup_excludes_switched_rows():
    # Under method="censoring" the outcome model drops switch == 1 rows, so the
    # follow-up tables must too.
    s = _build("censoring", weighted=True, weight_preexpansion=True)
    expected = (
        s.DT.filter(pl.col("switch") != 1)
        .group_by("tx_init_bas")
        .len()
        .sort("tx_init_bas")
    )
    assert s.diagnostics["nonunique_followup"].equals(expected)


def test_unique_followup_counts_distinct_subjects_per_arm():
    s = _build("ITT")
    expected = (
        s.DT.group_by("tx_init_bas")
        .agg(pl.col("ID").n_unique().alias("len"))
        .sort("tx_init_bas")
    )
    assert s.diagnostics["unique_followup"].equals(expected)
    # never more unique subjects than follow-up intervals
    u = s.diagnostics["unique_followup"].sort("tx_init_bas")
    nn = s.diagnostics["nonunique_followup"].sort("tx_init_bas")
    assert (u["len"] <= nn["len"]).all()
