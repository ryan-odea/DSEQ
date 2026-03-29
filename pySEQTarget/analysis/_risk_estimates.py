import math

import polars as pl
from scipy import stats


def _compute_rd_rr(comp, has_bootstrap, z=None, group_cols=None):
    """
    Compute Risk Difference and Risk Ratio from a comparison dataframe.
    Fallback used when paired bootstrap data is unavailable (e.g. subgroups).
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
                ((pl.col("risk_x") / pl.col("risk_y")) * (-z * rr_log_se).exp()).alias(
                    "RR 95% LCI"
                ),
                ((pl.col("risk_x") / pl.col("risk_y")) * (z * rr_log_se).exp()).alias(
                    "RR 95% UCI"
                ),
            ]
        )
        rr_comp = rr_comp.drop(["risk_x", "risk_y", "se_x", "se_y"])
        col_order = group_cols + [
            "A_x",
            "A_y",
            "Risk Ratio",
            "RR 95% LCI",
            "RR 95% UCI",
        ]
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

    # Paired approach: compute RD_i and RR_i per bootstrap iteration then take
    # SE or percentile of those — correct because arms share the same bootstrap
    # samples, so pairing captures their correlation. Falls back to the
    # independent delta method for subgroups (not yet supported) or when
    # _boot_risks is unavailable.
    use_paired = (
        has_bootstrap
        and not group_cols
        and hasattr(self, "_boot_risks")
        and all(tx in self._boot_risks for tx in self.treatment_level)
    )

    if has_bootstrap:
        alpha = 1 - self.bootstrap_CI
        z = stats.norm.ppf(1 - alpha / 2)
    else:
        z = None
        alpha = None

    risk_by_level = {}
    for tx in self.treatment_level:
        level_data = risk.filter(pl.col(self.treatment_col) == tx)
        risk_by_level[tx] = {"pred": level_data.select(group_cols + ["pred"])}
        if has_bootstrap and not use_paired:
            risk_by_level[tx]["SE"] = level_data.select(group_cols + ["SE"])

    rd_comparisons = []
    rr_comparisons = []

    for tx_x in self.treatment_level:
        for tx_y in self.treatment_level:
            if tx_x == tx_y:
                continue

            if use_paired:
                boot_x = (
                    self._boot_risks[tx_x]
                    .filter(pl.col("followup") == last_followup)
                    .select(["boot_idx", pl.col("risk").alias("risk_x")])
                )
                boot_y = (
                    self._boot_risks[tx_y]
                    .filter(pl.col("followup") == last_followup)
                    .select(["boot_idx", pl.col("risk").alias("risk_y")])
                )
                paired = boot_x.join(boot_y, on="boot_idx").with_columns(
                    (pl.col("risk_x") - pl.col("risk_y")).alias("RD")
                )

                risk_x_val = float(risk_by_level[tx_x]["pred"]["pred"][0])
                risk_y_val = float(risk_by_level[tx_y]["pred"]["pred"][0])
                rd_point = risk_x_val - risk_y_val
                rr_point = risk_x_val / risk_y_val if risk_y_val != 0 else float("inf")

                # Filter degenerate RR bootstrap values (risk_y == 0 or negative)
                valid_rr = paired.filter(
                    (pl.col("risk_y") > 0) & (pl.col("risk_x") >= 0)
                ).with_columns((pl.col("risk_x") / pl.col("risk_y")).alias("RR"))

                n_valid_rr = len(valid_rr)

                if self.bootstrap_CI_method == "percentile":
                    rd_lci = float(paired["RD"].quantile(alpha / 2))
                    rd_uci = float(paired["RD"].quantile(1 - alpha / 2))
                    if n_valid_rr >= 2:
                        rr_lci = float(valid_rr["RR"].quantile(alpha / 2))
                        rr_uci = float(valid_rr["RR"].quantile(1 - alpha / 2))
                    else:
                        rr_lci = float("nan")
                        rr_uci = float("nan")
                else:
                    rd_se = float(paired["RD"].std())
                    rd_lci = rd_point - z * rd_se
                    rd_uci = rd_point + z * rd_se
                    if n_valid_rr >= 2 and rr_point > 0:
                        log_rr_se = float(valid_rr["RR"].log().std())
                        rr_lci = math.exp(math.log(rr_point) - z * log_rr_se)
                        rr_uci = math.exp(math.log(rr_point) + z * log_rr_se)
                    else:
                        rr_lci = float("nan")
                        rr_uci = float("nan")

                rd_comp = pl.DataFrame(
                    {
                        "A_x": [tx_x],
                        "A_y": [tx_y],
                        "Risk Difference": [rd_point],
                        "RD 95% LCI": [rd_lci],
                        "RD 95% UCI": [rd_uci],
                    }
                )
                rr_comp = pl.DataFrame(
                    {
                        "A_x": [tx_x],
                        "A_y": [tx_y],
                        "Risk Ratio": [rr_point],
                        "RR 95% LCI": [rr_lci],
                        "RR 95% UCI": [rr_uci],
                    }
                )
            else:
                # Fall back to independent delta method
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
