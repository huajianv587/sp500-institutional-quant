from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .factors import FACTOR_FAMILIES, FactorEngine
from .ingestion import certify_point_in_time
from .ml import WalkForwardModel
from .portfolio import PortfolioOptimizer
from .schemas import BacktestResult, BacktestSpec, StrategyMetrics
from .storage import Store


@dataclass
class MonthFrame:
    signal_date: date
    return_date: date
    frame: pd.DataFrame


def apply_transaction_cost(gross_return: float, one_way_turnover: float, bps: float) -> float:
    return float(gross_return - one_way_turnover * bps / 10_000)


def _metrics(
    name: str, returns: pd.Series, benchmark: pd.Series, turnover: pd.Series
) -> StrategyMetrics:
    aligned = pd.concat(
        [returns.rename("portfolio"), benchmark.rename("benchmark")], axis=1
    ).dropna()
    values = aligned["portfolio"]
    if values.empty:
        return StrategyMetrics(
            strategy=name,
            cagr=0,
            annualized_volatility=0,
            sharpe_zero_rf=0,
            sortino_zero_rf=0,
            max_drawdown=0,
            beta=0,
            information_ratio=0,
            average_one_way_turnover=0,
            monthly_hit_rate=0,
            observations=0,
        )
    wealth = (1 + values).cumprod()
    years = len(values) / 12
    cagr = float(wealth.iloc[-1] ** (1 / years) - 1) if years else 0.0
    volatility = float(values.std(ddof=0) * np.sqrt(12))
    downside = float(values.where(values < 0, 0).std(ddof=0) * np.sqrt(12))
    excess = values - aligned["benchmark"]
    tracking_error = float(excess.std(ddof=0) * np.sqrt(12))
    benchmark_variance = float(aligned["benchmark"].var(ddof=0))
    beta = (
        float(aligned.cov(ddof=0).loc["portfolio", "benchmark"] / benchmark_variance)
        if benchmark_variance > 0
        else 0.0
    )

    return StrategyMetrics(
        strategy=name,
        cagr=cagr,
        annualized_volatility=volatility,
        sharpe_zero_rf=float(values.mean() * 12 / volatility) if volatility else 0.0,
        sortino_zero_rf=float(values.mean() * 12 / downside) if downside else 0.0,
        max_drawdown=float((wealth / wealth.cummax() - 1).min()),
        beta=beta,
        information_ratio=float(excess.mean() * 12 / tracking_error) if tracking_error else 0.0,
        average_one_way_turnover=float(turnover.reindex(values.index).fillna(0).mean()),
        monthly_hit_rate=float((values > 0).mean()),
        observations=len(values),
    )


def derive_cost_sensitivity(base: BacktestResult, bps: float) -> BacktestResult:
    """Reprice identical holdings/turnover without retraining on the test period."""
    frame = pd.DataFrame(base.monthly_returns)
    old_bps = base.spec.transaction_cost_bps
    strategy_map = {
        "Equal-weight historical universe": "equal_weight_universe",
        "factor_only": "factor_only",
        "ml_only": "ml_only",
        "factor_ml_ensemble": "factor_ml_ensemble",
        "ensemble_agent_overlay": "ensemble_agent_overlay",
    }
    for column in strategy_map.values():
        turnover_column = f"{column}_turnover"
        if column in frame and turnover_column in frame:
            frame[column] = frame[column] + frame[turnover_column] * (old_bps - bps) / 10_000
    frame.index = pd.to_datetime(frame["date"])
    metrics = [
        _metrics(
            "SPY buy-and-hold",
            frame["spy"],
            frame["spy"],
            pd.Series(0, index=frame.index),
        )
    ]
    for metric in base.metrics:
        if metric.strategy == "SPY buy-and-hold":
            continue
        column = strategy_map[metric.strategy]
        metrics.append(
            _metrics(
                metric.strategy,
                frame[column],
                frame["spy"],
                frame[f"{column}_turnover"],
            )
        )
    statistical_tests: list[dict[str, object]] = []
    benchmark_values = frame["spy"].astype(float).to_numpy()
    for metric in metrics:
        column = {
            "SPY buy-and-hold": "spy",
            "Equal-weight historical universe": "equal_weight_universe",
        }.get(metric.strategy, metric.strategy)
        values = frame[column].astype(float).to_numpy()
        design = np.column_stack([np.ones(len(values)), benchmark_values])
        coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
        residuals = values - design @ coefficients
        covariance = np.linalg.pinv(design.T @ design) * (
            residuals @ residuals / max(1, len(values) - 2)
        )
        alpha_se = float(np.sqrt(max(covariance[0, 0], 0.0)))
        statistical_tests.append(
            {
                "strategy": metric.strategy,
                "annualized_alpha": float(coefficients[0] * 12),
                "alpha_t_stat": (
                    float(coefficients[0] / alpha_se)
                    if alpha_se and abs(coefficients[0]) >= 1e-12
                    else 0.0
                ),
                "observations": len(values),
            }
        )
    statistical_tests.extend(item for item in base.statistical_tests if "factor" in item)
    return BacktestResult(
        spec=base.spec.model_copy(update={"transaction_cost_bps": bps}),
        certified_point_in_time=base.certified_point_in_time,
        certification_notes=base.certification_notes
        + [f"Cost sensitivity derived from frozen holdings at {bps:.0f} bps one way."],
        metrics=metrics,
        monthly_returns=frame.replace({np.nan: None}).to_dict(orient="records"),
        factor_ic=base.factor_ic,
        statistical_tests=statistical_tests,
        sector_exposures=base.sector_exposures,
        factor_diagnostics=base.factor_diagnostics,
    )


class BacktestEngine:
    """Point-in-time monthly walk-forward engine; all orders execute next session."""

    base_strategy_columns = {
        "factor_only": "factor_score",
        "ml_only": "ml_score",
        "factor_ml_ensemble": "ensemble_score",
    }

    def __init__(self, store: Store):
        self.store = store
        self.factors = FactorEngine(store)
        self.ml = WalkForwardModel()
        self.optimizer = PortfolioOptimizer()

    def _prices(self, start: date, end: date) -> pd.DataFrame:
        prices = self.store.load_prices(start, end)
        if prices.empty:
            return pd.DataFrame(
                columns=["company_id", "ticker", "price_date", "open", "adjusted_close"]
            )
        cutoff = pd.Timestamp(datetime.combine(end, time.max))
        prices = prices.loc[pd.to_datetime(prices["effective_at"]) <= cutoff]
        return prices[["company_id", "ticker", "price_date", "open", "adjusted_close"]].sort_values(
            ["price_date", "ticker"]
        )

    @staticmethod
    def _month_ends(prices: pd.DataFrame, start: date, end: date) -> list[date]:
        dates = pd.to_datetime(prices["price_date"])
        calendar = pd.DataFrame({"date": dates.drop_duplicates().sort_values()})
        calendar["month"] = calendar["date"].dt.to_period("M")
        month_ends = calendar.groupby("month")["date"].max().dt.date.tolist()
        history_start = (pd.Timestamp(start) - pd.DateOffset(months=37)).date()
        return [value for value in month_ends if history_start <= value <= end]

    @staticmethod
    def _forward_returns(
        prices: pd.DataFrame, signals: list[date], benchmark: str
    ) -> dict[date, pd.DataFrame]:
        output: dict[date, pd.DataFrame] = {}
        trading_dates = sorted(pd.to_datetime(prices["price_date"]).dt.date.unique())
        for index, signal in enumerate(signals[:-1]):
            later = [value for value in trading_dates if value > signal]
            if not later:
                continue
            entry = later[0]
            next_signal = signals[index + 1]
            next_dates = [value for value in trading_dates if value > next_signal]
            if not next_dates:
                continue
            exit_date = next_dates[0]
            entry_frame = prices.loc[
                pd.to_datetime(prices["price_date"]).dt.date == entry,
                ["company_id", "ticker", "open"],
            ]
            exit_frame = prices.loc[
                pd.to_datetime(prices["price_date"]).dt.date == exit_date,
                ["company_id", "ticker", "open"],
            ]
            merged = entry_frame.merge(
                exit_frame, on=["company_id", "ticker"], suffixes=("_entry", "_exit")
            )
            merged["next_return"] = merged["open_exit"] / merged["open_entry"] - 1
            benchmark_rows = merged.loc[merged["ticker"] == benchmark, "next_return"]
            benchmark_return = float(benchmark_rows.iloc[0]) if not benchmark_rows.empty else np.nan
            merged["benchmark_return"] = benchmark_return
            merged["next_month_excess_return"] = merged["next_return"] - benchmark_return
            merged["return_date"] = entry
            output[signal] = merged
        return output

    def _returns_history(
        self, prices: pd.DataFrame, signal: date, tickers: list[str]
    ) -> pd.DataFrame:
        cutoff = prices.loc[pd.to_datetime(prices["price_date"]).dt.date <= signal].copy()
        cutoff = cutoff.loc[cutoff["ticker"].isin(tickers)]
        pivot = cutoff.pivot_table(
            index="price_date", columns="ticker", values="adjusted_close", aggfunc="last"
        )
        return pivot.sort_index().pct_change().tail(252)

    def run(self, spec: BacktestSpec) -> BacktestResult:
        certified, notes = certify_point_in_time(self.store, spec.start_date, spec.end_date)
        if not certified:
            raise ValueError("Backtest certification failed: " + "; ".join(notes))
        load_start = (pd.Timestamp(spec.start_date) - pd.DateOffset(months=38, days=450)).date()
        load_end = spec.end_date + timedelta(days=40)
        prices = self._prices(load_start, load_end)
        if prices.empty:
            raise ValueError("No adjusted price history is available")
        signals = self._month_ends(prices, spec.start_date, spec.end_date)
        forward = self._forward_returns(prices, signals, spec.benchmark)
        feature_columns = sorted(
            {feature for family in FACTOR_FAMILIES.values() for feature in family}
            | {f"factor_{family}" for family in FACTOR_FAMILIES}
        )
        history: list[pd.DataFrame] = []
        month_frames: list[MonthFrame] = []
        for signal in signals:
            if signal not in forward:
                continue
            snapshot = self.factors.snapshot(signal)
            labelled = snapshot.merge(
                forward[signal][
                    [
                        "company_id",
                        "next_return",
                        "next_month_excess_return",
                        "benchmark_return",
                        "return_date",
                    ]
                ],
                on="company_id",
                how="inner",
            )
            training = pd.concat(history, ignore_index=True) if history else pd.DataFrame()
            if training.empty:
                predicted = labelled.copy()
                predicted["ml_score"] = 0.0
                predicted["ensemble_score"] = predicted["factor_score"].rank(pct=True)
            else:
                predicted, _ = self.ml.predict(training, labelled, feature_columns)
            self.factors.persist(predicted)
            month_frames.append(
                MonthFrame(signal, forward[signal]["return_date"].iloc[0], predicted)
            )
            history.append(labelled)

        records: list[dict[str, object]] = []
        ic_records: list[dict[str, object]] = []
        sector_records: list[dict[str, object]] = []
        quantile_records: list[dict[str, object]] = []
        factor_correlations: list[pd.DataFrame] = []
        strategy_columns = dict(self.base_strategy_columns)
        if spec.agent_overlay:
            strategy_columns["ensemble_agent_overlay"] = "agent_adjusted_score"
        previous: dict[str, dict[str, float]] = {name: {} for name in strategy_columns}
        previous_equal: dict[str, float] = {}
        overlay_months: set[date] = set()
        for month in month_frames:
            frame = month.frame
            if month.return_date < spec.start_date or month.return_date > spec.end_date:
                continue
            benchmark_return = float(frame["benchmark_return"].iloc[0])
            universe = frame.loc[frame["ticker"] != spec.benchmark].copy()
            universe["agent_adjusted_score"] = universe["ensemble_score"]
            if spec.agent_overlay:
                decisions = self.store.query_df(
                    """
                    SELECT company_id, decision_json FROM agent_study_decisions
                    WHERE as_of_date = ? AND variant = ?
                    """,
                    [month.signal_date, spec.agent_variant],
                )
                if not decisions.empty:
                    import json

                    adjustments = {}
                    for decision in decisions.itertuples():
                        payload = decision.decision_json
                        if isinstance(payload, str):
                            payload = json.loads(payload)
                        adjustments[str(decision.company_id)] = float(payload["score_adjustment"])
                    universe["agent_adjusted_score"] = universe.apply(
                        lambda row, values=adjustments: (
                            float(row["ensemble_score"]) + values.get(str(row["company_id"]), 0.0)
                        ),
                        axis=1,
                    )
                    overlay_months.add(month.signal_date)
            equal_weights = {str(ticker): 1 / len(universe) for ticker in universe["ticker"]}
            equal_turnover = (
                1.0
                if not previous_equal
                else 0.5
                * (
                    sum(
                        abs(weight - previous_equal.get(ticker, 0.0))
                        for ticker, weight in equal_weights.items()
                    )
                    + sum(
                        weight
                        for ticker, weight in previous_equal.items()
                        if ticker not in equal_weights
                    )
                )
            )
            equal_gross = sum(
                equal_weights[str(row.ticker)] * float(row.next_return)
                for row in universe.itertuples()
            )
            record: dict[str, object] = {
                "date": month.return_date.isoformat(),
                "spy": benchmark_return,
                "equal_weight_universe": apply_transaction_cost(
                    equal_gross, equal_turnover, spec.transaction_cost_bps
                ),
                "equal_weight_universe_turnover": equal_turnover,
            }
            previous_equal = equal_weights
            for factor in ["factor_score", *[f"factor_{name}" for name in FACTOR_FAMILIES]]:
                valid = universe[[factor, "next_month_excess_return"]].dropna()
                ic = (
                    float(spearmanr(valid[factor], valid["next_month_excess_return"]).statistic)
                    if len(valid) >= 5
                    else np.nan
                )
                ic_records.append(
                    {
                        "date": month.return_date.isoformat(),
                        "factor": factor,
                        "rank_ic": None if np.isnan(ic) else ic,
                    }
                )
            valid_quantiles = universe[["factor_score", "next_month_excess_return"]].dropna()
            if len(valid_quantiles) >= 10:
                valid_quantiles["quantile"] = pd.qcut(
                    valid_quantiles["factor_score"], 5, labels=False, duplicates="drop"
                )
                for quantile, values in valid_quantiles.groupby("quantile"):
                    quantile_records.append(
                        {
                            "date": month.return_date.isoformat(),
                            "quantile": int(quantile) + 1,
                            "mean_excess_return": float(values["next_month_excess_return"].mean()),
                        }
                    )
            family_columns = [f"factor_{name}" for name in FACTOR_FAMILIES]
            factor_correlations.append(universe[family_columns].corr(method="spearman"))
            sector_weights = universe["sector"].value_counts(normalize=True).to_dict()
            for name, column in strategy_columns.items():
                returns_history = self._returns_history(
                    prices, month.signal_date, universe["ticker"].tolist()
                )
                recommendation = self.optimizer.optimize(
                    universe,
                    returns_history,
                    month.signal_date,
                    score_column=column,
                    current_weights=previous[name],
                    benchmark_sector_weights=sector_weights,
                    min_positions=spec.min_positions,
                    max_positions=spec.max_positions,
                )
                weights = {
                    position.ticker: position.weight for position in recommendation.positions
                }
                gross = sum(
                    weights.get(row.ticker, 0.0) * float(row.next_return)
                    for row in universe.itertuples()
                )
                record[name] = apply_transaction_cost(
                    gross, recommendation.one_way_turnover, spec.transaction_cost_bps
                )
                record[f"{name}_turnover"] = recommendation.one_way_turnover
                previous[name] = weights
                portfolio_sector: dict[str, float] = {}
                for position in recommendation.positions:
                    portfolio_sector[position.sector] = (
                        portfolio_sector.get(position.sector, 0.0) + position.weight
                    )
                for sector in sorted(set(sector_weights) | set(portfolio_sector)):
                    benchmark_weight = float(sector_weights.get(sector, 0.0))
                    portfolio_weight = float(portfolio_sector.get(sector, 0.0))
                    sector_records.append(
                        {
                            "date": month.return_date.isoformat(),
                            "strategy": name,
                            "sector": sector,
                            "portfolio_weight": portfolio_weight,
                            "benchmark_weight": benchmark_weight,
                            "active_weight": portfolio_weight - benchmark_weight,
                        }
                    )
            records.append(record)

        monthly = pd.DataFrame(records)
        if monthly.empty:
            raise ValueError("No complete out-of-sample months were produced")
        monthly.index = pd.to_datetime(monthly["date"])
        metrics = [
            _metrics(
                "SPY buy-and-hold",
                monthly["spy"],
                monthly["spy"],
                pd.Series(0, index=monthly.index),
            ),
            _metrics(
                "Equal-weight historical universe",
                monthly["equal_weight_universe"],
                monthly["spy"],
                monthly["equal_weight_universe_turnover"],
            ),
        ]
        if spec.agent_overlay and len(overlay_months) < 24:
            raise ValueError(
                f"Agent overlay requires 24 monthly decision sets; found {len(overlay_months)}"
            )
        for name in strategy_columns:
            metrics.append(
                _metrics(name, monthly[name], monthly["spy"], monthly[f"{name}_turnover"])
            )
        statistical_tests: list[dict[str, object]] = []
        benchmark_values = monthly["spy"].astype(float).to_numpy()
        for metric in metrics:
            strategy_column = {
                "SPY buy-and-hold": "spy",
                "Equal-weight historical universe": "equal_weight_universe",
            }.get(metric.strategy, metric.strategy)
            values = monthly[strategy_column].astype(float).to_numpy()
            design = np.column_stack([np.ones(len(values)), benchmark_values])
            coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
            residuals = values - design @ coefficients
            covariance = np.linalg.pinv(design.T @ design) * (
                residuals @ residuals / max(1, len(values) - 2)
            )
            alpha_se = float(np.sqrt(max(covariance[0, 0], 0.0)))
            statistical_tests.append(
                {
                    "strategy": metric.strategy,
                    "annualized_alpha": float(coefficients[0] * 12),
                    "alpha_t_stat": (
                        float(coefficients[0] / alpha_se)
                        if alpha_se and abs(coefficients[0]) >= 1e-12
                        else 0.0
                    ),
                    "observations": len(values),
                }
            )
        ic_frame = pd.DataFrame(ic_records).dropna(subset=["rank_ic"])
        for factor, values in ic_frame.groupby("factor"):
            series = values["rank_ic"].astype(float)
            standard_error = series.std(ddof=1) / np.sqrt(len(series)) if len(series) > 1 else 0.0
            statistical_tests.append(
                {
                    "factor": factor,
                    "mean_rank_ic": float(series.mean()),
                    "ic_t_stat": (float(series.mean() / standard_error) if standard_error else 0.0),
                    "ic_positive_rate": float((series > 0).mean()),
                    "observations": len(series),
                }
            )
        average_correlation = (
            sum(factor_correlations) / len(factor_correlations)
            if factor_correlations
            else pd.DataFrame()
        )
        result = BacktestResult(
            spec=spec,
            certified_point_in_time=True,
            certification_notes=notes
            + [
                "Signals use data effective by month-end; orders use the next available session open."
            ],
            metrics=metrics,
            monthly_returns=monthly.replace({np.nan: None}).to_dict(orient="records"),
            factor_ic=ic_records,
            statistical_tests=statistical_tests,
            sector_exposures=sector_records,
            factor_diagnostics={
                "quantile_returns": quantile_records,
                "average_factor_correlation": (
                    average_correlation.where(pd.notna(average_correlation), None).to_dict()
                    if not average_correlation.empty
                    else {}
                ),
            },
        )
        self.store.save_backtest(result)
        return result
