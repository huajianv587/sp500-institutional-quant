# Capital IQ to portfolio operating runbook

This runbook turns licensed S&P Capital IQ Pro exports into a reproducible
research result. It is intentionally explicit about what is verified, what is a
substitute, and what still blocks certification.

The project's main question is: **can institutional S&P company information
produce explainable, stable cross-sectional factors and better research
decisions?** Yahoo and Alpaca supply only market-return labels, market-derived
features and paper execution. They are supporting infrastructure, not the
research thesis or a replacement fundamental dataset.

## 1. Study clock

| Stage | Date range | Purpose |
|---|---|---|
| Price warm-up | 2017-04-01 to 2018-07-31 | 12-1 momentum, 252-day volatility/beta and clean next-open execution |
| Model training warm-up | 2018-08-01 to 2021-08-31 | expanding walk-forward training and validation only |
| Frozen out-of-sample study | 2021-09-01 to 2026-08-31 | reported five-year performance |
| Agent overlay | most recent 24 completed rebalance months | debate/no-debate ablation |

No observation may enter a signal before its `effective_at`. `as_of_date` is
the source snapshot date; it is not a replacement for filing or availability
time.

## 2. Export checklist

Create a local staging directory such as `data/exports/ciq/`. It is ignored by
Git. Never upload licensed workbooks to GitHub.

### Current live-research loop

For day-to-day company research, start with the two verified Capital IQ saved
screens:

- `TRAINING_V3_SP500_INSTITUTIONAL_FACTORS_CURRENT`
- `TRAINING_V3_SP500_ESTIMATES_REVISIONS_CURRENT`

Export them as **Results As Table Function**, then import with the actual
download timestamp. The first screen must retain Financial Filing Date; the
second must retain FY+1 EPS Period End. After import:

```bash
uv run python scripts/audit_current_snapshot.py --as-of 2026-09-01
uv run sp500iq factor-snapshot --as-of 2026-09-01 --top 25
```

The verified current snapshot has all six factor families, with 489 companies
covered by revisions. The same command at `2026-08-31` returns zero investable
companies, which is the expected proof that the 2026-09-01 Capital IQ snapshots
cannot leak backward. The displayed leaders are research candidates, not an
automatic investment list.

The verified live Agent smoke uses one immutable EvidencePacket and the full
nine-node graph. `deepseek-v4-pro` completed four independent analysts, two
bull/bear rounds and a consensus judge; every accepted judge reference resolved
to an exact Evidence ID. A separate no-debate judge reuses the independent
analyst cache. This smoke proves API/schema/evidence integration only. Run the
frozen multi-sector model benchmark and 24-month Agent study before making model
quality or strategy-performance claims.

### A. Instruments

One current S&P 500 export is enough for the current display layer:

- Entity Name
- Entity ID
- Exchange: Ticker
- CIQ Sector / GICS sector
- Reporting currency, when available

This snapshot does not prove historical membership.

### B. Reported fundamentals

Export annual and, where practical, quarterly observations covering fiscal
periods from 2017 through 2026. Every row or wide workbook must include Entity
ID, Exchange: Ticker, Period Ended, period type, Financial Filing Date and the
snapshot/as-of date.

Preferred analytical metrics:

| Factor family | Direct metric or deterministic inputs |
|---|---|
| Value | `earnings_yield` or net income + market cap; `fcf_yield` or FCF + market cap; `ebitda_to_ev` or EBITDA + enterprise value |
| Quality | `roic` or NOPAT + invested capital; gross profit + total assets; net income + operating cash flow + total assets; net debt + EBITDA |
| Growth | `revenue_growth`, `eps_growth`, `margin_change`, or the current and prior-year components |

The importer maps common Capital IQ labels such as `IQ_TOTAL_REV`, `IQ_EBITDA`,
`IQ_FCF`, `IQ_TOTAL_ASSETS`, `IQ_MARKET_CAP` and `IQ_TEV` into the canonical
factor vocabulary. The immutable source workbook still preserves the original
keyfield labels.

Suggested filenames:

```text
data/exports/ciq/fundamentals_FY_2017_2026_asof_2026-08-31.xlsx
data/exports/ciq/fundamentals_FQ_2017_2026_asof_2026-08-31.xlsx
```

If Capital IQ restricts row or field count, split by fiscal year or metric
family. Do not remove Entity ID, ticker, period, filing date or as-of fields from
any split.

### C. Point-in-time estimates

Create one self-contained snapshot for every month from 2018-08 through
2026-08. The 2021-09 through 2026-08 months are a hard certification gate; the
earlier months initialize the walk-forward model.

Each snapshot should contain:

- Entity ID and Exchange: Ticker
- snapshot date / `effective_at`
- target fiscal period (or FY/FQ relative code plus the companion FY0/FQ0 Period Ended)
- normalized EPS FY+1 and revenue FY+1 consensus
- EPS revision over one month and three months
- estimate surprise
- analyst count when licensed
- target price and consensus rating only as supplementary evidence

Use an unambiguous monthly filename:

```text
data/exports/ciq/estimates_FY1_asof_2021-09-30.xlsx
data/exports/ciq/estimates_FY1_asof_2021-10-29.xlsx
...
data/exports/ciq/estimates_FY1_asof_2026-08-31.xlsx
```

A missing numeric estimate for an individual company is acceptable as missing
data and is never imputed. A missing monthly snapshot is not acceptable for the
certified study.

### D. Membership and prices

The NTU session did not expose historical S&P 500 membership. The platform uses
a pinned, explicitly non-Capital-IQ public reconstruction and records this fact
in every certification/report. If a licensed historical membership source later
becomes available, import it as `index_membership`; Capital IQ rows take
precedence without relabelling the public archive.

Yahoo adjusted daily bars provide the broad historical return/factor panel and
remain labelled `yahoo_adjusted`. Alpaca IEX remains labelled
`alpaca_iex_adjusted` and supports gaps, monitoring and paper execution. Prefer
Capital IQ adjusted total-return prices and corporate actions if NTU later
permits the export. None of these market feeds supplies company fundamentals or
evidence for the institutional research thesis.

The production market panel is stored at `IQ_PRICE_LAKE_PATH` as local Parquet;
Supabase retains its source manifest and downstream research results, not the
million-row public-price copies. Rebuild the cache deterministically with:

```bash
uv run sp500iq sync-yahoo-prices --start 2017-04-01 --end 2026-08-31
uv run sp500iq sync-alpaca-prices --start 2017-04-01 --end 2026-08-31
```

Verified current substitute coverage: 1,910,469 unique source observations,
2017-04-03 through 2026-08-31. The 60 study months have at least 99.2% exact
month-end constituent coverage (99.6% median), and the active-membership check
accepts zero adjusted one-day returns above 300%. These are engineering/data
quality results, not investment performance.

## 3. Import without exposing secrets

Use the internal-disk environment on macOS because this external volume creates
AppleDouble files inside virtual environments:

```bash
export UV_PROJECT_ENVIRONMENT=/Users/guohuiwen/.cache/sp500-institutional-quant-venv
uv sync --extra dev --extra tradingagents-upstream
uv run sp500iq preflight --live
uv run sp500iq init-db
```

Import each workbook. Re-importing an identical file is idempotent because its
SHA-256 already exists in the source manifest.

```bash
uv run sp500iq import-ciq fundamentals \
  /absolute/path/to/fundamentals_FY_2017_2026_asof_2026-08-31.xlsx

uv run sp500iq import-ciq estimates \
  /absolute/path/to/estimates_FY1_asof_2021-09-30.xlsx
```

Then inspect the gate:

```bash
uv run sp500iq serve
# open http://127.0.0.1:8000/data-status
# or GET http://127.0.0.1:8000/api/v1/data-quality
```

Do not proceed merely because a workbook imported. Resolve source errors and
confirm the full historical coverage window.

## 4. Research and selection sequence

Once Data Status is certified:

1. Run the base backtest with `POST /api/v1/backtests` using the frozen default
   `BacktestSpec`.
2. Inspect Factor Lab for rank IC, IC stability, quantile monotonicity, factor
   correlations and sector exposure. A high in-sample return alone does not
   qualify a factor. The composite gate requires at least four of six factor
   families and at least two institutional families; momentum/low-risk alone is
   diagnostic and cannot enter the investable rank.
3. Build approximately 40 representative evidence packets and run
   `POST /api/v1/model-benchmarks`. Freeze the best decision route and the
   within-two-points supporting route.
4. Run `POST /api/v1/research-runs` for the latest rebalance. With no explicit
   company IDs, the API selects leaders, deteriorating holdings and factor/ML
   disagreements, up to ten companies.
5. Run the 24-month Agent study with `historical_months=24`, once with debate
   and once through the stored ablation path.
6. Run `uv run sp500iq case-study` to freeze primary 10 bps results, 5/25 bps
   sensitivities, ablations, source hashes, code hash and model fingerprints.
7. Review the proposed 20–30 stock portfolio. Generate an Alpaca paper preview;
   submit only after explicit approval in the UI.

## 5. Release checklist

- Complete test suite passes from a clean environment.
- Data Status is point-in-time certified for the frozen study window.
- Stable company IDs and ticker-at-date checks pass across known renames; closed
  ticker intervals reject recycled-symbol prices and no anomalous price is
  silently used.
- Model benchmark, 24-month Agent study and all backtest ablations exist.
- Reports distinguish Capital IQ, public membership, Yahoo and Alpaca observations.
- `.env`, `data/raw`, `data/exports`, `output` and licensed files are absent from
  `git status` and the staged diff.
- A secret scan and licensed-data scan return no findings.
- README result tables are generated from the frozen manifest, not hand-edited.
- Only then is the project pushed to the public GitHub repository.
