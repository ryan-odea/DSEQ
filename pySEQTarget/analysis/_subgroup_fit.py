import polars as pl

from ._outcome_fit import _outcome_fit


def _subgroup_fit(self, start_params=None):
    subgroups = sorted(self.DT[self.subgroup_colname].unique().to_list())
    self._unique_subgroups = subgroups

    models_list = []
    for val in subgroups:
        subDT = self.DT.filter(pl.col(self.subgroup_colname) == val)
        sg_start = (start_params or {}).get(val, {}) or {}

        models = {
            "outcome": _outcome_fit(
                self,
                subDT,
                self.outcome_col,
                self.covariates,
                self.weighted,
                "weight",
                start_params=sg_start.get("outcome"),
            )
        }

        if self.compevent_colname is not None:
            models["compevent"] = _outcome_fit(
                self,
                subDT,
                self.compevent_colname,
                self.covariates,
                self.weighted,
                "weight",
                start_params=sg_start.get("compevent"),
            )
        models_list.append(models)
    return models_list
