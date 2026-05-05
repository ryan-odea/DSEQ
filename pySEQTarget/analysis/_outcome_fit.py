import re

import numpy as np
import polars as pl
import statsmodels.api as sm
import statsmodels.formula.api as smf


def _compute_spline_knots(followup_arr, df=3):
    lower = float(np.min(followup_arr))
    upper = float(np.max(followup_arr))
    n_inner = df - 2
    if n_inner == 0:
        inner_knots = []
    else:
        # Replicate patsy's knot placement: percentiles of unique values in [lower, upper]
        x = np.unique(followup_arr[(lower <= followup_arr) & (followup_arr <= upper)])
        q = np.linspace(0, 100, n_inner + 2)[1:-1]
        inner_knots = np.percentile(x, q.tolist()).tolist()
    return inner_knots, lower, upper


def _apply_spline_formula(formula, indicator_squared, spline_knots):
    inner_knots, lower, upper = spline_knots
    spline = f"cr(followup, knots={inner_knots}, lower_bound={lower}, upper_bound={upper})"

    formula = re.sub(r"(\w+)\s*\*\s*followup\b", rf"\1*{spline}", formula)
    formula = re.sub(r"\bfollowup\s*\*\s*(\w+)", rf"{spline}*\1", formula)
    formula = re.sub(rf"\bfollowup{re.escape(indicator_squared)}\b", "", formula)
    formula = re.sub(r"\bfollowup\b", "", formula)

    formula = re.sub(r"\s+", " ", formula)
    formula = re.sub(r"\+\s*\+", "+", formula)
    formula = re.sub(r"^\s*\+\s*|\s*\+\s*$", "", formula).strip()

    if formula:
        return f"{formula} + I({spline}**2)"
    return f"I({spline}**2)"


def _cast_categories(self, df_pd):
    if self.treatment_col in df_pd.columns:
        df_pd[self.treatment_col] = df_pd[self.treatment_col].astype("category")
    tx_bas = f"{self.treatment_col}{self.indicator_baseline}"
    if tx_bas in df_pd.columns:
        df_pd[tx_bas] = df_pd[tx_bas].astype("category")

    if self.followup_class and not self.followup_spline:
        df_pd["followup"] = df_pd["followup"].astype("category")
        squared_col = f"followup{self.indicator_squared}"
        if squared_col in df_pd.columns:
            df_pd[squared_col] = df_pd[squared_col].astype("category")

    if self.fixed_cols:
        for col in self.fixed_cols:
            if col in df_pd.columns:
                df_pd[col] = df_pd[col].astype("category")

    return df_pd


def _outcome_fit(
    self,
    df: pl.DataFrame,
    outcome: str,
    formula: str,
    weighted: bool = False,
    weight_col: str = "weight",
):
    if weighted:
        df = df.with_columns(
            pl.col(weight_col).clip(
                lower_bound=self.weight_min, upper_bound=self.weight_max
            )
        )

    if self.method == "censoring":
        df = df.filter(pl.col("switch") != 1)

    df_pd = _cast_categories(self, df.to_pandas())

    if self.followup_spline:
        if getattr(self, "_current_boot_idx", None) is None:
            self._followup_spline_knots = _compute_spline_knots(
                self.DT["followup"].to_numpy()
            )
        formula = _apply_spline_formula(
            formula, self.indicator_squared, self._followup_spline_knots
        )

    full_formula = f"{outcome} ~ {formula}"

    glm_kwargs = {
        "formula": full_formula,
        "data": df_pd,
        "family": sm.families.Binomial(),
    }

    if weighted:
        glm_kwargs["var_weights"] = df_pd[weight_col]

    model = smf.glm(**glm_kwargs)
    model_fit = model.fit()
    return model_fit
