import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pySEQTarget.helpers._fix_categories import _fix_categories_for_predict


def test_fix_categories_preserves_integer_dtype():
    """Regression test: integer categorical columns must not be coerced to strings.
    The old code did .astype(str).astype('category') which caused category-level
    mismatches and NaN predictions."""
    # Mirror actual usage: column is pre-cast to categorical dtype, plain name in formula
    df = pd.DataFrame(
        {
            "y": [0, 1, 0, 1, 0, 1, 0, 1],
            "tx": pd.Categorical([0, 0, 0, 0, 1, 1, 1, 1]),
        }
    )
    model = smf.glm("y ~ tx", data=df, family=sm.families.Binomial()).fit(disp=False)

    # Plain integers — what Polars→pandas produces before _cast_categories runs
    newdata = pd.DataFrame({"tx": [0, 1, 0, 1]})
    fixed = _fix_categories_for_predict(model, newdata)

    # Categories must remain integer-typed, not strings
    assert isinstance(fixed["tx"].dtype, pd.CategoricalDtype)
    assert fixed["tx"].dtype.categories.dtype == df["tx"].cat.categories.dtype

    # No NaNs in predictions — this would fail on the old code
    probs = model.predict(fixed)
    assert not np.any(np.isnan(probs))
