import numpy as np
import pandas as pd
import patsy
from glum import GeneralizedLinearRegressor


class _GlumFit:
    """
    Wraps a fitted glum model exposing the statsmodels interface the rest of
    the codebase expects:
      .params (Series), .model.exog_names, .model.data.design_info,
      .predict(df) and .predict(X_numpy, transform=False).
    """

    class _Data:
        pass

    def __init__(self, glum_model, design_info, feature_names):
        self._glum = glum_model
        self._design_info = design_info

        # .model.data.design_info — used by _survival_pred and _fix_categories
        _d = self._Data()
        _d.design_info = design_info
        self._data = _d

        # statsmodels convention: intercept first
        all_coefs = np.concatenate([[glum_model.intercept_], glum_model.coef_])
        self.params = pd.Series(all_coefs, index=feature_names)

    @property
    def model(self):
        # makes .model.exog_names and .model.data.design_info work
        return self

    @property
    def data(self):
        return self._data

    @property
    def exog_names(self):
        return list(self.params.index)

    def predict(self, data, transform=True):
        if transform:
            # data is a pandas DataFrame — build design matrix via stored patsy info
            X = patsy.build_design_matrices(
                [self._design_info], data, return_type="dataframe"
            )[0]
            X_arr = X.drop(columns=["Intercept"], errors="ignore").values
        else:
            # data is a pre-built numpy design matrix (includes intercept col — drop it)
            X_arr = np.asarray(data)[:, 1:]
        return self._glum.predict(X_arr)


def _fit_glum(formula, data, var_weights=None):
    """Fit a binomial GLM with glum and return a _GlumFit wrapper."""
    y_mat, X_mat = patsy.dmatrices(formula, data, return_type="dataframe")
    y_arr = y_mat.values.ravel()
    design_info = X_mat.design_info
    feature_names = list(X_mat.columns)  # "Intercept" first, then predictors
    X_arr = X_mat.drop(columns=["Intercept"]).values

    glm = GeneralizedLinearRegressor(family="binomial", fit_intercept=True)

    fit_kwargs = {}
    if var_weights is not None:
        fit_kwargs["sample_weight"] = np.asarray(var_weights)

    glm.fit(X_arr, y_arr, **fit_kwargs)
    return _GlumFit(glm, design_info, feature_names)
