import numpy as np
import polars as pl
import pytest

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


def _data_with_compevent(p=0.05, seed=42):
    rng = np.random.default_rng(seed)
    data = load_data("SEQdata")
    return data.with_columns(
        pl.Series("compevent", (rng.random(data.height) < p).astype(int))
    )


def _build(with_ce=True, **opts):
    data = _data_with_compevent() if with_ce else load_data("SEQdata")
    params = dict(**opts)
    if with_ce:
        params["compevent_colname"] = "compevent"
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
    return s


def test_compevent_tables_present_and_accessible():
    s = _build()
    for key in ("unique_compevent", "nonunique_compevent"):
        assert key in s.diagnostics

    s.fit()
    out = s.collect()
    for key in ("unique_compevent", "nonunique_compevent"):
        tbl = out.retrieve_data(key)
        assert "tx_init_bas" in tbl.columns
        assert "len" in tbl.columns


def test_nonunique_compevent_counts_intervals_per_arm():
    # Non-unique counts intervals (rows) with compevent == 1, per baseline arm.
    s = _build()
    expected = (
        s.DT.filter(pl.col("compevent") == 1)
        .group_by("tx_init_bas")
        .len()
        .sort("tx_init_bas")
    )
    assert s.diagnostics["nonunique_compevent"].equals(expected)


def test_unique_compevent_counts_distinct_subjects_per_arm():
    s = _build()
    expected = (
        s.DT.filter(pl.col("compevent") == 1)
        .group_by("tx_init_bas")
        .agg(pl.col("ID").n_unique().alias("len"))
        .sort("tx_init_bas")
    )
    assert s.diagnostics["unique_compevent"].equals(expected)
    u = s.diagnostics["unique_compevent"].sort("tx_init_bas")
    nn = s.diagnostics["nonunique_compevent"].sort("tx_init_bas")
    assert (u["len"] <= nn["len"]).all()


def test_compevent_tables_absent_when_no_compevent_configured():
    # When compevent_colname is None the diagnostics dict has no compevent keys
    # and retrieve_data raises (data is None -> ValueError).
    s = _build(with_ce=False)
    assert "unique_compevent" not in s.diagnostics
    assert "nonunique_compevent" not in s.diagnostics

    s.fit()
    out = s.collect()
    with pytest.raises(ValueError):
        out.retrieve_data("unique_compevent")
    with pytest.raises(ValueError):
        out.retrieve_data("nonunique_compevent")
