import unittest.mock as mock

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import pytest

matplotlib.use("Agg")  # non-interactive backend — no windows opened

from pySEQTarget import SEQopts, SEQuential  # noqa: E402
from pySEQTarget.data import load_data  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def base_seq():
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
        parameters=SEQopts(km_curves=True),
    )
    s.expand()
    s.fit()
    s.survival()
    return s


def test_sequential_plot_returns_figure(base_seq):
    """SEQuential.plot() should store a Figure in km_graph, not print its repr."""
    base_seq.plot()
    assert isinstance(base_seq.km_graph, matplotlib.figure.Figure)


def test_sequential_plot_calls_show(base_seq):
    """SEQuential.plot() must call plt.show() to actually display the figure."""
    with mock.patch("matplotlib.pyplot.show") as mock_show:
        base_seq.plot()
        mock_show.assert_called_once()


def test_seqoutput_plot_shows_figure(base_seq):
    """SEQoutput.plot() should display the figure without raising."""
    base_seq.plot()
    result = base_seq.collect()
    result.plot()  # must not raise; previously printed Figure repr instead


def test_seqoutput_plot_raises_without_km_graph():
    """SEQoutput.plot() raises ValueError when no figure was generated."""
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
        parameters=SEQopts(km_curves=False),
    )
    s.expand()
    s.fit()
    result = s.collect()
    with pytest.raises(ValueError):
        result.plot()


def test_sequential_plot_risk(base_seq):
    base_seq.plot(plot_type="risk")
    assert isinstance(base_seq.km_graph, matplotlib.figure.Figure)


def test_sequential_plot_survival():
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
        parameters=SEQopts(km_curves=True, plot_type="survival"),
    )
    s.expand()
    s.fit()
    s.survival()
    s.plot()
    assert isinstance(s.km_graph, matplotlib.figure.Figure)


def test_sequential_plot_subgroups():
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
        parameters=SEQopts(km_curves=True, subgroup_colname="sex"),
    )
    s.expand()
    s.fit()
    s.survival()
    s.plot()
    assert isinstance(s.km_graph, matplotlib.figure.Figure)
