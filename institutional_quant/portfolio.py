from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .schemas import PortfolioPosition, PortfolioRecommendation


class PortfolioOptimizer:
    def __init__(
        self,
        max_weight: float = 0.05,
        sector_active_limit: float = 0.08,
        target_volatility: float = 0.12,
        monthly_turnover_limit: float = 0.20,
    ):
        self.max_weight = max_weight
        self.sector_active_limit = sector_active_limit
        self.target_volatility = target_volatility
        self.monthly_turnover_limit = monthly_turnover_limit

    def optimize(
        self,
        candidates: pd.DataFrame,
        returns_history: pd.DataFrame,
        as_of_date: date,
        *,
        score_column: str = "ensemble_score",
        current_weights: dict[str, float] | None = None,
        benchmark_sector_weights: dict[str, float] | None = None,
        min_positions: int = 20,
        max_positions: int = 30,
    ) -> PortfolioRecommendation:
        current_weights = current_weights or {}
        ranked = candidates.sort_values(score_column, ascending=False).copy()
        if benchmark_sector_weights:
            selected_indices: list[int] = []
            for sector, target in benchmark_sector_weights.items():
                minimum_weight = max(0.0, float(target) - self.sector_active_limit)
                required_names = int(np.ceil(minimum_weight / self.max_weight))
                selected_indices.extend(
                    ranked.loc[ranked["sector"] == sector].head(required_names).index.tolist()
                )
            selected_indices = list(dict.fromkeys(selected_indices))
            remaining = [index for index in ranked.index if index not in selected_indices]
            selected_indices.extend(remaining[: max(0, max_positions - len(selected_indices))])
            selected = ranked.loc[selected_indices[:max_positions]].copy().reset_index(drop=True)
        else:
            selected = ranked.head(max(max_positions, 20)).reset_index(drop=True)
        if len(selected) < min_positions:
            raise ValueError(f"At least {min_positions} candidates are required")

        tickers = selected["ticker"].astype(str).tolist()
        sectors = selected["sector"].fillna("Unknown").astype(str).tolist()
        score = selected[score_column].astype(float).to_numpy()
        score = (score - np.nanmean(score)) / (np.nanstd(score) or 1.0)
        returns = returns_history.reindex(columns=tickers).fillna(0.0)
        covariance = (
            returns.cov().to_numpy() * 252 if len(returns) >= 20 else np.eye(len(tickers)) * 0.04
        )
        covariance = np.nan_to_num(covariance, nan=0.0)
        covariance += np.eye(len(tickers)) * 1e-6

        previous = np.array([current_weights.get(ticker, 0.0) for ticker in tickers])
        initial = np.repeat(1 / len(tickers), len(tickers))
        initial = np.minimum(initial, self.max_weight)
        initial /= initial.sum()

        if benchmark_sector_weights is None:
            counts = pd.Series(sectors).value_counts(normalize=True)
            benchmark_sector_weights = counts.to_dict()

        def objective(weights: np.ndarray) -> float:
            expected = float(weights @ score)
            variance = float(weights @ covariance @ weights)
            volatility = np.sqrt(max(variance, 0.0))
            volatility_penalty = max(0.0, volatility - self.target_volatility) ** 2
            turnover_penalty = float(np.abs(weights - previous).sum()) if current_weights else 0.0
            return -expected + 8.0 * variance + 30.0 * volatility_penalty + 0.15 * turnover_penalty

        constraints: list[dict] = [{"type": "eq", "fun": lambda weights: weights.sum() - 1.0}]
        for sector in sorted(set(sectors)):
            indices = np.array([index for index, value in enumerate(sectors) if value == sector])
            target = float(benchmark_sector_weights.get(sector, 0.0))
            lower = max(0.0, target - self.sector_active_limit)
            upper = min(1.0, target + self.sector_active_limit)
            constraints.extend(
                [
                    {
                        "type": "ineq",
                        "fun": lambda weights, idx=indices, value=lower: weights[idx].sum() - value,
                    },
                    {
                        "type": "ineq",
                        "fun": lambda weights, idx=indices, value=upper: value - weights[idx].sum(),
                    },
                ]
            )
        if current_weights:
            outside_sales = sum(
                weight for ticker, weight in current_weights.items() if ticker not in tickers
            )
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda weights: (
                        self.monthly_turnover_limit
                        - 0.5 * (np.abs(weights - previous).sum() + outside_sales)
                    ),
                }
            )

        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=[(0.0, self.max_weight)] * len(tickers),
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10},
        )
        warnings: list[str] = []
        if not result.success:
            if current_weights:
                return PortfolioRecommendation(
                    as_of_date=as_of_date,
                    one_way_turnover=0.0,
                    positions=[
                        PortfolioPosition(
                            company_id=str(
                                selected.loc[selected["ticker"] == ticker, "company_id"].iloc[0]
                                if ticker in tickers
                                else ticker
                            ),
                            ticker=ticker,
                            sector=str(
                                selected.loc[selected["ticker"] == ticker, "sector"].iloc[0]
                                if ticker in tickers
                                else "Unknown"
                            ),
                            weight=weight,
                            score=float(
                                selected.loc[selected["ticker"] == ticker, score_column].iloc[0]
                                if ticker in tickers
                                else 0.0
                            ),
                        )
                        for ticker, weight in current_weights.items()
                        if weight > 1e-8 and weight <= self.max_weight + 1e-8
                    ],
                    status="held",
                    warnings=[
                        f"Optimizer infeasible; existing portfolio retained: {result.message}"
                    ],
                )
            raise ValueError(f"Initial portfolio is infeasible: {result.message}")

        weights = np.where(result.x < 1e-6, 0.0, result.x)
        variance = float(weights @ covariance @ weights)
        one_way_turnover = (
            1.0
            if not current_weights
            else 0.5
            * (
                np.abs(weights - previous).sum()
                + sum(weight for ticker, weight in current_weights.items() if ticker not in tickers)
            )
        )
        positions = [
            PortfolioPosition(
                company_id=str(selected.iloc[index]["company_id"]),
                ticker=tickers[index],
                sector=sectors[index],
                weight=float(weight),
                score=float(selected.iloc[index][score_column]),
            )
            for index, weight in enumerate(weights)
            if weight > 1e-6
        ]
        if len(positions) < min_positions:
            warnings.append(
                f"Optimizer produced {len(positions)} positions; required minimum is {min_positions}"
            )
        return PortfolioRecommendation(
            as_of_date=as_of_date,
            target_volatility=self.target_volatility,
            expected_volatility=float(np.sqrt(max(variance, 0.0))),
            one_way_turnover=float(one_way_turnover),
            positions=positions,
            warnings=warnings,
        )
