import polars as pl
from polars.testing import assert_frame_equal

from pySEQTarget import SEQopts, SEQuential
from pySEQTarget.data import load_data


def _make_model(data):
    return SEQuential(
        data,
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="treatment",
        outcome_col="outcome",
        time_varying_cols=[],
        fixed_cols=[],
    )


def test_expansion_truncates_at_first_outcome():
    """Expansion should truncate each (id, trial) at and including the first
    outcome=1 row. Rows from later periods must not appear."""
    data = pl.DataFrame(
        {
            "ID": [1, 1, 1, 1, 1],
            "time": [0, 1, 2, 3, 4],
            "eligible": [1, 0, 0, 0, 0],
            # Both treatment values required by default treatment_level=[0,1]
            "treatment": [0, 1, 0, 1, 0],
            # outcome=1 at time=2, but original data continues with outcome=0
            "outcome": [0, 0, 1, 0, 0],
        }
    )

    model = _make_model(data)
    model.expand()

    # Only trial 0 exists (eligible only at time=0).
    # Should have followup 0, 1, 2 — the outcome=1 row is included but not beyond.
    followups = sorted(model.DT["followup"].to_list())
    assert followups == [0, 1, 2]

    # The outcome=1 row must be present
    outcome_row = model.DT.filter(pl.col("outcome") == 1)
    assert len(outcome_row) == 1
    assert int(outcome_row["followup"][0]) == 2


def test_expansion_does_not_truncate_without_outcome():
    """Subjects who never experience the outcome should retain all expanded rows."""
    data = pl.DataFrame(
        {
            "ID": [1, 1, 1, 1, 1],
            "time": [0, 1, 2, 3, 4],
            "eligible": [1, 0, 0, 0, 0],
            "treatment": [0, 1, 0, 1, 0],
            "outcome": [0, 0, 0, 0, 0],
        }
    )

    model = _make_model(data)
    model.expand()

    # All 5 followup periods should be present
    followups = sorted(model.DT["followup"].to_list())
    assert followups == [0, 1, 2, 3, 4]


def test_expansion_truncates_each_trial_independently():
    """Truncation must apply per (id, trial), not globally. A subject enrolled in
    multiple trials should have each trial truncated at its own first outcome."""
    data = pl.DataFrame(
        {
            "ID": [1, 1, 1, 1, 1],
            "time": [0, 1, 2, 3, 4],
            "eligible": [1, 1, 0, 0, 0],
            "treatment": [0, 1, 0, 1, 0],
            # outcome=1 at time=3: trial 0 sees it at followup=3, trial 1 at followup=2
            "outcome": [0, 0, 0, 1, 0],
        }
    )

    model = _make_model(data)
    model.expand()

    trial_0 = model.DT.filter(pl.col("trial") == 0)
    trial_1 = model.DT.filter(pl.col("trial") == 1)

    # Trial 0 starts at time=0, outcome at time=3 → followup 0,1,2,3
    assert sorted(trial_0["followup"].to_list()) == [0, 1, 2, 3]

    # Trial 1 starts at time=1, outcome at time=3 → followup 0,1,2
    assert sorted(trial_1["followup"].to_list()) == [0, 1, 2]


def test_expand_only_returns_expanded_dataframe():
    """expand_only=True should return the expanded DataFrame directly and the
    return value should equal self.DT from a standard expand() call."""
    data = pl.DataFrame(
        {
            "ID": [1, 1, 1, 1, 1],
            "time": [0, 1, 2, 3, 4],
            "eligible": [1, 0, 0, 0, 0],
            "treatment": [0, 1, 0, 1, 0],
            "outcome": [0, 0, 0, 0, 0],
        }
    )

    model_only = SEQuential(
        data,
        id_col="ID",
        time_col="time",
        eligible_col="eligible",
        treatment_col="treatment",
        outcome_col="outcome",
        time_varying_cols=[],
        fixed_cols=[],
        parameters=SEQopts(expand_only=True),
    )
    result = model_only.expand()

    assert isinstance(result, pl.DataFrame)
    assert_frame_equal(result, model_only.DT)

    model_full = _make_model(data)
    model_full.expand()

    assert_frame_equal(result, model_full.DT)


def _make_verbose_model(verbose, **extra_opts):
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
        parameters=SEQopts(verbose=verbose, **extra_opts),
    )


def test_verbose_expand(capsys):
    s = _make_verbose_model(verbose=True)
    s.expand()
    out = capsys.readouterr().out
    assert "Full dataset:" in out
    assert "Eligible observations:" in out
    assert "Expanded dataset:" in out
    assert "Final analysis dataset:" in out
    assert "Sampled expanded dataset:" not in out
    assert "observations" in out
    assert "variables" in out


def test_verbose_expand_with_sampling(capsys):
    s = _make_verbose_model(verbose=True, selection_random=True, selection_sample=0.5)
    s.expand()
    out = capsys.readouterr().out
    assert "Sampled expanded dataset:" in out


def test_verbose_bootstrap(capsys):
    s = _make_verbose_model(verbose=True, bootstrap_nboot=10)
    s.expand()
    capsys.readouterr()
    s.bootstrap()
    out = capsys.readouterr().out
    assert "Bootstrapping" in out
    assert "subjects" in out
    assert "observations per resample" in out
    assert "10 times" in out


def test_verbose_false_no_output(capsys):
    s = _make_verbose_model(verbose=False, bootstrap_nboot=5)
    s.expand()
    s.bootstrap()
    out = capsys.readouterr().out
    assert out == ""
