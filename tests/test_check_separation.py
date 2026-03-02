import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from pySEQTarget.error._check_separation import _check_separation


def _mock_model(params):
    """Create a minimal mock model with a .params attribute."""
    return SimpleNamespace(params=np.array(params))


def test_no_warning_for_normal_coefficients():
    model = _mock_model([0.5, -1.2, 3.0, -0.01])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _check_separation(model)  # should not raise


def test_warns_for_large_positive_coefficient():
    model = _mock_model([0.5, 26.0])
    with pytest.warns(UserWarning, match="separation"):
        _check_separation(model)


def test_warns_for_large_negative_coefficient():
    model = _mock_model([0.5, -30.0])
    with pytest.warns(UserWarning, match="separation"):
        _check_separation(model)


def test_warns_for_inf_coefficient():
    model = _mock_model([1.0, np.inf])
    with pytest.warns(UserWarning, match="separation"):
        _check_separation(model)


def test_warns_for_neg_inf_coefficient():
    model = _mock_model([1.0, -np.inf])
    with pytest.warns(UserWarning, match="separation"):
        _check_separation(model)


def test_warns_for_nan_coefficient():
    model = _mock_model([1.0, np.nan])
    with pytest.warns(UserWarning, match="separation"):
        _check_separation(model)


def test_boundary_coefficient_does_not_warn():
    # Exactly 25 should not trigger (threshold is strictly > 25)
    model = _mock_model([25.0, -25.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _check_separation(model)


def test_label_appears_in_warning():
    model = _mock_model([100.0])
    with pytest.warns(UserWarning, match="censoring numerator"):
        _check_separation(model, label="censoring numerator")


def test_default_label_appears_in_warning():
    model = _mock_model([np.inf])
    with pytest.warns(UserWarning, match="model"):
        _check_separation(model)
