import polars as pl
from scipy import stats


def _compute_rd_rr(comp, has_bootstrap, z=None, group_cols=None):
    """
    Compute Risk Difference and Risk Ratio from a comparison dataframe.
    Consolidates the repeated calculation logic.
    """
    if group_cols is None:
        group_cols = []

    if has_bootstrap:
        rd_se = (pl.col("se_x").pow(2) + pl.col("se_y").pow(2)).sqrt()
        rd_comp = comp.with_columns(
            [
                (pl.col("risk_x") - pl.col("risk_y")).alias("Risk Difference"),
                (pl.col("risk_x") - pl.col("risk_y") - z * rd_se).alias("RD 95% LCI"),
                (pl.col("risk_x") - pl.col("risk_y") + z * rd_se).alias("RD 95% UCI"),
            ]
        )
        rd_comp = rd_comp.drop(["risk_x", "risk_y", "se_x", "se_y"])
        col_order = group_cols + [
            "A_x",
            "A_y",
            "Risk Difference",
            "RD 95% LCI",
            "RD 95% UCI",
        ]
        rd_comp = rd_comp.select([c for c in col_order if c in rd_comp.columns])

        rr_log_se = (
            (pl.col("se_x") / pl.col("risk_x")).pow(2)
            + (pl.col("se_y") / pl.col("risk_y")).pow(2)
        ).sqrt()
        rr_comp = comp.with_columns(
            [
                (pl.col("risk_x") / pl.col("risk_y")).alias("Risk Ratio"),
                (
                    (pl.col("risk_x") / pl.col("risk_y")) * (-z * rr_log_se).exp()
                ).alias("RR 95% LCI"),
                (
                    (pl.col("risk_x") / pl.col("risk_y")) * (z * rr_log_se).exp()
                ).alias("RR 95% UCI"),
            ]
        )
        rr_comp = rr_comp.drop(["risk_x", "risk_y", "se_x", "se_y"])
        col_order = group_cols + ["A_x", "A_y", "Risk Ratio", "RR 95% LCI", "RR 95% UCI"]
        rr_comp = rr_comp.select([c for c in col_order if c in rr_comp.columns])
    else:
        rd_comp = comp.with_columns(
            (pl.col("risk_x") - pl.col("risk_y")).alias("Risk Difference")
        )
        rd_comp = rd_comp.drop(["risk_x", "risk_y"])
        col_order = group_cols + ["A_x", "A_y", "Risk Difference"]
        rd_comp = rd_comp.select([c for c in col_order if c in rd_comp.columns])

        rr_comp = comp.with_columns(
            (pl.col("risk_x") / pl.col("risk_y")).alias("Risk Ratio")
        )
        rr_comp = rr_comp.drop(["risk_x", "risk_y"])
        col_order = group_cols + ["A_x", "A_y", "Risk Ratio"]
        rr_comp = rr_comp.select([c for c in col_order if c in rr_comp.columns])

    return rd_comp, rr_comp


def _risk_estimates(self):
    last_followup = self.km_data["followup"].max()
    risk = self.km_data.filter(
        (pl.col("followup") == last_followup) & (pl.col("estimate") == "risk")
    )

    group_cols = [self.subgroup_colname] if self.subgroup_colname else []
    has_bootstrap = self.bootstrap_nboot > 0

    if has_bootstrap:
        alpha = 1 - self.bootstrap_CI
        z = stats.norm.ppf(1 - alpha / 2)
    else:
        z = None

    # Pre-extract data for each treatment level once (avoid repeated filtering)
    risk_by_level = {}
    for tx in self.treatment_level:
        level_data = risk.filter(pl.col(self.treatment_col) == tx)
        risk_by_level[tx] = {
            "pred": level_data.select(group_cols + ["pred"]),
        }
        if has_bootstrap:
            risk_by_level[tx]["SE"] = level_data.select(group_cols + ["SE"])

    rd_comparisons = []
    rr_comparisons = []

    for tx_x in self.treatment_level:
        for tx_y in self.treatment_level:
            if tx_x == tx_y:
                continue

            # Use pre-extracted data instead of filtering again
            risk_x = risk_by_level[tx_x]["pred"].rename({"pred": "risk_x"})
            risk_y = risk_by_level[tx_y]["pred"].rename({"pred": "risk_y"})

            if group_cols:
                comp = risk_x.join(risk_y, on=group_cols, how="left")
            else:
                comp = risk_x.join(risk_y, how="cross")

            comp = comp.with_columns(
                [pl.lit(tx_x).alias("A_x"), pl.lit(tx_y).alias("A_y")]
            )

            if has_bootstrap:
                se_x = risk_by_level[tx_x]["SE"].rename({"SE": "se_x"})
                se_y = risk_by_level[tx_y]["SE"].rename({"SE": "se_y"})

                if group_cols:
                    comp = comp.join(se_x, on=group_cols, how="left")
                    comp = comp.join(se_y, on=group_cols, how="left")
                else:
                    comp = comp.join(se_x, how="cross")
                    comp = comp.join(se_y, how="cross")

            rd_comp, rr_comp = _compute_rd_rr(comp, has_bootstrap, z, group_cols)
            rd_comparisons.append(rd_comp)
            rr_comparisons.append(rr_comp)

    risk_difference = pl.concat(rd_comparisons) if rd_comparisons else pl.DataFrame()
    risk_ratio = pl.concat(rr_comparisons) if rr_comparisons else pl.DataFrame()

    return {"risk_difference": risk_difference, "risk_ratio": risk_ratio}
