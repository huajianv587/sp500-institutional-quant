"""Profile one current research snapshot without printing licensed row-level values."""

from __future__ import annotations

import argparse
import json
from datetime import date

from dotenv import load_dotenv

from institutional_quant.config import Settings
from institutional_quant.factors import FACTOR_FAMILIES, FactorEngine
from institutional_quant.storage import create_store


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", required=True, type=date.fromisoformat)
    args = parser.parse_args()

    load_dotenv()
    store = create_store(Settings.from_env())
    store.initialize()
    as_of = args.as_of

    fundamental_summary = store.query_df(
        """
        SELECT COUNT(*) AS rows,
               COUNT(DISTINCT company_id) AS companies,
               COUNT(DISTINCT metric) AS metrics,
               SUM(CASE WHEN effective_at >= as_of_date + INTERVAL '1 day' THEN 1 ELSE 0 END)
                   AS future_effective,
               SUM(CASE WHEN period_end > as_of_date THEN 1 ELSE 0 END) AS future_period_end
        FROM fundamentals WHERE as_of_date = ?
        """,
        [as_of],
    ).iloc[0]
    fundamental_duplicates = store.query_df(
        """
        SELECT COUNT(*) AS duplicate_groups FROM (
          SELECT company_id, metric, period_type, period_end, effective_at, as_of_date
          FROM fundamentals WHERE as_of_date = ?
          GROUP BY 1, 2, 3, 4, 5, 6 HAVING COUNT(*) > 1
        ) duplicate_grain
        """,
        [as_of],
    ).iloc[0]["duplicate_groups"]
    fundamental_coverage = store.query_df(
        """
        SELECT metric, COUNT(DISTINCT company_id) AS companies
        FROM fundamentals WHERE as_of_date = ?
        GROUP BY metric ORDER BY metric
        """,
        [as_of],
    )
    estimate_summary = store.query_df(
        """
        SELECT COUNT(*) AS rows,
               COUNT(DISTINCT company_id) AS companies,
               COUNT(DISTINCT metric) AS metrics,
               SUM(CASE WHEN effective_at >= as_of_date + INTERVAL '1 day' THEN 1 ELSE 0 END)
                   AS future_effective,
               SUM(CASE WHEN fiscal_period IS NULL THEN 1 ELSE 0 END) AS missing_target_period
        FROM estimates WHERE as_of_date = ?
        """,
        [as_of],
    ).iloc[0]
    estimate_duplicates = store.query_df(
        """
        SELECT COUNT(*) AS duplicate_groups FROM (
          SELECT company_id, metric, fiscal_period, effective_at, as_of_date
          FROM estimates WHERE as_of_date = ?
          GROUP BY 1, 2, 3, 4, 5 HAVING COUNT(*) > 1
        ) duplicate_grain
        """,
        [as_of],
    ).iloc[0]["duplicate_groups"]
    estimate_coverage = store.query_df(
        """
        SELECT metric, COUNT(DISTINCT company_id) AS companies
        FROM estimates WHERE as_of_date = ?
        GROUP BY metric ORDER BY metric
        """,
        [as_of],
    )
    snapshot = FactorEngine(store).snapshot(as_of)
    family_coverage = {
        family: int(snapshot[f"factor_{family}"].notna().sum()) for family in FACTOR_FAMILIES
    }
    feature_ranges = {}
    for feature in sorted({name for family in FACTOR_FAMILIES.values() for name in family}):
        numeric = snapshot[feature].dropna()
        feature_ranges[feature] = {
            "coverage": int(len(numeric)),
            "p01": None if numeric.empty else float(numeric.quantile(0.01)),
            "median": None if numeric.empty else float(numeric.median()),
            "p99": None if numeric.empty else float(numeric.quantile(0.99)),
        }

    result = {
        "as_of_date": as_of.isoformat(),
        "grain": "one company, metric, period, effective timestamp and snapshot date",
        "fundamentals": {
            "rows": int(fundamental_summary["rows"]),
            "companies": int(fundamental_summary["companies"]),
            "metrics": int(fundamental_summary["metrics"]),
            "duplicate_groups": int(fundamental_duplicates),
            "future_effective": int(fundamental_summary["future_effective"]),
            "future_period_end": int(fundamental_summary["future_period_end"]),
            "coverage_by_metric": {
                str(row.metric): int(row.companies) for row in fundamental_coverage.itertuples()
            },
        },
        "estimates": {
            "rows": int(estimate_summary["rows"]),
            "companies": int(estimate_summary["companies"]),
            "metrics": int(estimate_summary["metrics"]),
            "duplicate_groups": int(estimate_duplicates),
            "future_effective": int(estimate_summary["future_effective"]),
            "missing_target_period": int(estimate_summary["missing_target_period"]),
            "coverage_by_metric": {
                str(row.metric): int(row.companies) for row in estimate_coverage.itertuples()
            },
        },
        "factor_snapshot": {
            "companies": int(len(snapshot)),
            "investable": int(snapshot["factor_score"].notna().sum()),
            "family_coverage": family_coverage,
            "feature_ranges": feature_ranges,
        },
    }
    print(json.dumps(result, indent=2, allow_nan=False))
    critical = any(
        int(value) > 0
        for value in (
            fundamental_duplicates,
            fundamental_summary["future_effective"],
            fundamental_summary["future_period_end"],
            estimate_duplicates,
            estimate_summary["future_effective"],
            estimate_summary["missing_target_period"],
        )
    )
    return 1 if critical else 0


if __name__ == "__main__":
    raise SystemExit(main())
