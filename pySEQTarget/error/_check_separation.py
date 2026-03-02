import warnings

import numpy as np


def _check_separation(model_fit, label="model"):
    """
    Check for perfect or quasi-complete separation in a fitted logistic regression model.
    Issues a warning if large (|coef| > 25) or non-finite coefficients are detected,
    as these are reliable indicators of separation in logistic regression.
    """
    params = np.array(model_fit.params).flatten()

    has_large = np.any(np.abs(params) > 25)
    has_nonfinite = np.any(~np.isfinite(params))

    if has_large or has_nonfinite:
        warnings.warn(
            f"Possible perfect or quasi-complete separation detected in {label}. "
            "The resulting weights may be unreliable.",
            UserWarning,
            stacklevel=2,
        )
