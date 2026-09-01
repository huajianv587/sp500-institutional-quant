from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelSelection:
    elastic_params: dict[str, float]
    tree_params: dict[str, float | int]
    elastic_validation_mae: float | None
    tree_validation_mae: float | None


class WalkForwardModel:
    """Monthly cross-sectional ElasticNet and gradient-boosting ensemble."""

    def __init__(self, min_training_months: int = 24, validation_months: int = 6):
        self.min_training_months = min_training_months
        self.validation_months = validation_months

    @staticmethod
    def _elastic(params: dict[str, float]) -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    ElasticNet(
                        alpha=float(params["alpha"]),
                        l1_ratio=float(params["l1_ratio"]),
                        max_iter=5000,
                        random_state=7,
                    ),
                ),
            ]
        )

    @staticmethod
    def _tree(params: dict[str, float | int]) -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=float(params["learning_rate"]),
                        max_leaf_nodes=int(params["max_leaf_nodes"]),
                        max_iter=150,
                        l2_regularization=0.1,
                        random_state=7,
                    ),
                ),
            ]
        )

    def _select(self, train: pd.DataFrame, features: list[str], target: str) -> ModelSelection:
        months = sorted(pd.to_datetime(train["as_of_date"]).dt.date.unique())
        if len(months) <= self.validation_months:
            return ModelSelection(
                {"alpha": 0.01, "l1_ratio": 0.5},
                {"learning_rate": 0.05, "max_leaf_nodes": 15},
                None,
                None,
            )
        validation_set = set(months[-self.validation_months :])
        core = train.loc[~pd.to_datetime(train["as_of_date"]).dt.date.isin(validation_set)]
        validation = train.loc[pd.to_datetime(train["as_of_date"]).dt.date.isin(validation_set)]
        x_core, y_core = core[features], core[target]
        x_validation, y_validation = validation[features], validation[target]

        elastic_candidates = [
            {"alpha": alpha, "l1_ratio": ratio}
            for alpha in (0.001, 0.01, 0.1)
            for ratio in (0.1, 0.5, 0.9)
        ]
        tree_candidates = [
            {"learning_rate": rate, "max_leaf_nodes": leaves}
            for rate in (0.03, 0.08)
            for leaves in (7, 15)
        ]

        def best(candidates, factory):
            scores = []
            for params in candidates:
                model = factory(params)
                model.fit(x_core, y_core)
                scores.append(
                    (mean_absolute_error(y_validation, model.predict(x_validation)), params)
                )
            return min(scores, key=lambda item: item[0])

        elastic_mae, elastic_params = best(elastic_candidates, self._elastic)
        tree_mae, tree_params = best(tree_candidates, self._tree)
        return ModelSelection(elastic_params, tree_params, elastic_mae, tree_mae)

    def predict(
        self,
        history: pd.DataFrame,
        current: pd.DataFrame,
        feature_columns: list[str],
        target_column: str = "next_month_excess_return",
    ) -> tuple[pd.DataFrame, ModelSelection | None]:
        output = current.copy()
        eligible = history.dropna(subset=[target_column]).copy()
        if not eligible.empty:
            history_max = pd.to_datetime(eligible["as_of_date"]).max()
            current_min = pd.to_datetime(current["as_of_date"]).min()
            if history_max >= current_min:
                raise ValueError(
                    "Walk-forward training rows must strictly precede the prediction month"
                )
        months = pd.to_datetime(eligible["as_of_date"]).dt.to_period("M").nunique()
        if months < self.min_training_months or len(eligible) < 100:
            output["elastic_score"] = 0.0
            output["tree_score"] = 0.0
            output["ml_score"] = 0.0
            output["ensemble_score"] = output["factor_score"]
            return output, None

        selection = self._select(eligible, feature_columns, target_column)
        elastic = TransformedTargetRegressor(
            regressor=self._elastic(selection.elastic_params), transformer=StandardScaler()
        )
        tree = self._tree(selection.tree_params)
        elastic.fit(eligible[feature_columns], eligible[target_column])
        tree.fit(eligible[feature_columns], eligible[target_column])
        output["elastic_score"] = elastic.predict(output[feature_columns])
        output["tree_score"] = tree.predict(output[feature_columns])
        output["ml_score"] = (
            output["elastic_score"].rank(pct=True) + output["tree_score"].rank(pct=True)
        ) / 2
        factor_rank = output["factor_score"].rank(pct=True)
        output["ensemble_score"] = 0.5 * factor_rank + 0.5 * output["ml_score"]
        return output, selection
