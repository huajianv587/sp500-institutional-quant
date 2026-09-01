# Capital IQ Pro export contracts

The importer accepts CSV, XLSX, XLSM and XLS. Column names are case-insensitive and common Capital IQ labels are normalized. Keep every original export immutable; the platform archives a SHA-256-addressed copy and records the source ID on every row.

## Required exports

| Dataset argument | Required columns | Optional columns / notes |
|---|---|---|
| `instruments` | company ID, ticker, company name, GICS sector, `effective_at`, `as_of_date` | currency |
| `index_membership` | company ID, ticker-at-date, member-from date, `effective_at`, `as_of_date` | member-to date, index code (defaults to SP500) |
| `fundamentals` | company ID, ticker, fiscal period end, period type, `effective_at`, `as_of_date`, metric, value | unit; wide exports are automatically melted |
| `estimates` | company ID, ticker, fiscal period, `effective_at`, `as_of_date`, metric, value | `valid_to`, unit; a CIQ relative-period export may instead carry the estimate's FY/FQ code plus its companion FY0/FQ0 Period Ended date |
| `prices` | company ID, ticker, price date, OHLC, adjusted close, source, `effective_at`, `as_of_date` | volume; include corporate-action-adjusted total-return history and SPY |
| `ownership` | company ID, ticker, `effective_at`, `as_of_date` | institutional percentage and change |
| `insider_transactions` | company ID, ticker, transaction date/type, shares, `effective_at`, `as_of_date` | transaction value |

`effective_at` means the first timestamp at which the observation was available to the strategy—not the export time. `as_of_date` identifies the export/snapshot. A file without required availability fields is rejected and cannot certify a backtest.

## Suggested Screening templates

1. Historical S&P 500 membership: stable company ID, security ID, ticker-at-date, membership start/end, index code, snapshot date.
2. Statements: quarterly and annual reported line items plus filing/availability date. Include revenue, net income, EPS, operating cash flow, free cash flow, EBITDA, NOPAT, invested capital, gross profit, total assets, net debt, enterprise value and market cap where licensed.
3. Estimates: point-in-time EPS/revenue estimates, fiscal period, observation timestamp, validity end, revision and surprise fields.
4. Prices/actions: daily unadjusted OHLC, adjusted close/total-return price, splits/dividends and the timestamp at which corrections became available.
5. Supplementary: institutional ownership changes and insider transactions with availability timestamps.

Before the first licensed import, compare the actual NTU export headers with these contracts. Browser-assisted validation remains optional; manual exports work with the same importer.

## Verified NTU web export: 2026-09-01

The Capital IQ Pro system list **S&P 500 Constituents** was exported through Safari with 500 rows and these keyfield headers:

| Capital IQ header | Normalized field |
|---|---|
| `SP_ENTITY_NAME` | `company_name` |
| `SP_ENTITY_ID` | `company_id` |
| `SP_EXCHANGE_TICKER` | `ticker` (exchange prefix removed) |
| `MI_PRIMARY_INDUSTRY` | retained as an unused source column |
| `IQ_SECTOR` | `sector` |

Capital IQ places `SPGTable`/`SPGLabel` formula rows above the real header. The importer now detects the keyfield header automatically. The verified file had 500 unique company IDs and one missing sector value, so 499 rows passed the instrument contract and one row was rejected.

The web `Index Constituents` criterion exposed the current S&P 500 selection but no historical/as-of parameter. Therefore this export is registered only as a current instrument snapshot; it is never accepted as historical membership. Import it explicitly with its download timestamp:

```bash
uv run sp500iq import-ciq instruments /absolute/path/current-sp500.xlsx \
  --current-snapshot-as-of 2026-09-01 \
  --current-snapshot-effective-at 2026-09-01T04:17:07+00:00
```

The two current-snapshot options are accepted only for `instruments`,
`fundamentals` and `estimates`, and are always labelled live-only. Fundamentals
still require Capital IQ's source-provided Financial Filing Date. Estimates
still require a company-specific target fiscal period. Historical membership,
historical fundamentals/estimates and prices must carry source-provided
point-in-time fields; an operator timestamp never certifies history.

### Safari validation of the historical-membership gate

The NTU Capital IQ Pro session was revalidated in Safari on `2026-09-01` against three separate web paths:

- The S&P 500 `Index Constituents` page returned 503 current securities as of `2026-09-01`. Its `SP_CONSTITUENTS` field (`KeyField 319764`) has no date or period secondary key, so it is current-only.
- The Companies Screener exposed the same `SP_CONSTITUENTS` field. Its related official-weight and index-shares fields expose ranking controls, not an as-of date.
- The legacy `Component Companies` report accepted a `2021-09-30` membership-date label, but the S&P 500 request returned zero rows and `No data matches your current settings.` The historical report works with its listed SNL index variants; the session did not expose an SNL historical variant for the S&P 500.

These checks are capability evidence, not licensed data exports. Do not copy the 503 current constituents backward across the test period. Capital IQ historical-membership authority therefore remains unavailable under the NTU entitlement.

### Explicit public historical-membership fallback

For the fixed `2021-09-01` through `2026-08-31` study, run
`sync-public-sp500-membership`. It uses the MIT-licensed `pitindex` event log at
commit `c3d5d4961076a59041b3e1de90fe5ea052f61bb4`, then applies two source-linked
2026-08-18 corrections: RDDT replaced AVB under the S&P Global announcement,
and EQR changed ticker to VMRK under the issuer's SEC filing. The sync:

- archives the exact upstream files, licence, overrides and URLs in one immutable JSON bundle;
- hashes both the bundle and each upstream file;
- reconstructs member intervals rather than backfilling today's roster;
- collapses known duplicate share classes to one company entity;
- rejects no-op events, roster-size anomalies and end-roster reconciliation failures;
- stores `capital_iq_data=false` and emits a visible certification note.

This opens the point-in-time membership gate but does **not** turn the case study
into a fully Capital-IQ-sourced result. WRDS/CRSP remains the preferred upgrade
if NTU provides a CRSP subscription; otherwise the public reconstruction must
stay labelled in every report.

### Historical adjusted-price substitution

Where Capital IQ adjusted-price history is unavailable, the platform can load
Yahoo adjusted bars and Alpaca IEX daily bars (`adjustment=all`). Observations
remain labelled `yahoo_adjusted` or `alpaca_iex_adjusted`, live in the local
Parquet price lake, and are explicitly identified as non-Capital-IQ data in
certification. They must never be relabelled as Capital IQ observations.

The verified combined cache contains 1,910,469 unique source observations from
`2017-04-03` through `2026-08-31`. Historical membership supplies stable company
identity and ticker-at-date. In particular, `VIAC → PARA → PSKY` resolves to one
Capital IQ company ID, and prices under the recycled `PARA` ticker after
`2025-08-06` are rejected. `BRK.B` remains the tradable S&P representative even
when a current company snapshot uses `BRK.A`.

### Verified point-in-time fundamentals parameters

A second Safari export validated the S&P Capital IQ Fundamentals fields below for `Latest Fiscal Year` as of `2026-08-31`:

| Capital IQ field | Keyfield | Import meaning |
|---|---:|---|
| Total Revenue | `329288` / `IQ_TOTAL_REV` | wide metric, melted to `metric=iq_total_rev` |
| Period Ended | `329317` / `IQ_PERIOD_END` | `period_end` |
| Financial Filing Date | `329318` / `IQ_FINL_FILING_DATE` | source-provided `effective_at` |

The workbook's `SPGLabel` formulas preserve both `FY0` and `08/31/2026`. The importer extracts those formula parameters as `period_type=FY` and `as_of_date=2026-08-31`, and drops the non-observation `FY0` parameter row below the header. Formula metadata is retained in the source manifest. Because `Financial Filing Date` has day rather than intraday precision, it is conservatively stored at end-of-day so a same-day signal cannot treat the filing as available before publication.

The validation export intentionally demonstrated the missing-identity gate: it contained Entity Name and Entity ID but not Exchange:Ticker, so it was rejected with `MISSING_REQUIRED_COLUMNS: ticker` and was not written to production. A production fundamentals export must also include `SP_EXCHANGE_TICKER`.

### Verified current institutional factor template

The saved screen `TRAINING_V3_SP500_INSTITUTIONAL_FACTORS_CURRENT` uses the S&P
500 Companies universe and exports 500 unique Entity IDs. Its verified fields
cover revenue, gross profit, EBITDA, net income, assets, debt, equity, operating
cash flow, capex, one-year revenue growth, margins, P/E, P/B, TEV/EBITDA, ROA,
ROC, ROE, market capitalization, enterprise value, filing date, sector,
industry and Exchange:Ticker.

The importer assigns each metric its own `FY`, `LTM` or `CURRENT` semantics from
the SPGLabel metadata. Capital IQ percentage-point fields are divided by 100 at
ingestion and the normalization is recorded in the source manifest. The
verified 2026-09-01 snapshot produced 10,223 accepted observations across 21
canonical metrics, with no duplicate grain, future availability or future
fundamental period. It is suitable for live research only.

### Verified current estimates and revisions template

The saved screen `TRAINING_V3_SP500_ESTIMATES_REVISIONS_CURRENT` contains:

| Capital IQ field | Keyfield | Import meaning |
|---|---:|---|
| CIQ EPS Normalized Est | `290476` / `SP_EPS_NORM_EST` | `eps_estimate`, FY+1 |
| CIQ Revenue Est | `290526` / `SP_REV_EST` | `revenue_estimate`, FY+1 |
| EPS Normalized # of Analysts Last Month | `330485` | analyst coverage |
| EPS Normalized # Upward / Downward Last Month | `330486` / `330491` | one-month revision counts |
| EPS Normalized # Upward / Downward Last 3 Months | `330505` / `330506` | three-month revision counts |
| CIQ EPS Normalized Est Period End | `290486` / `SP_EPS_NORM_DATE_EST` | company-specific FY+1 target date |
| Exchange: Ticker | `331277` / `SP_EXCHANGE_TICKER` | point-in-time identifier |

The deterministic revision factor is `(up - down) / max(analyst coverage,
up + down)`, bounded to `[-1, 1]`. The verified 2026-09-01 workbook held 500
unique companies; 497 had valid target periods. It produced 3,434 accepted
observations across seven estimate metrics, with revision-family coverage for
489 companies. Capital IQ NA values and the three missing target periods were
rejected and recorded, never imputed.

### Earlier self-contained historical estimate validation

The verified estimate template uses the following self-contained fields as of `2026-08-31`:

| Capital IQ field | Keyfield | Parameters / import meaning |
|---|---:|---|
| CIQ EPS Normalized Actual/Estimate | `325375` / `SP_NORM_EPS_ACT_OR_EST` | `FY+1`, point-in-time estimate value |
| Period Ended | `329317` / `IQ_PERIOD_END` | `Latest Fiscal Year` (`FY0`), company-specific base period |
| Exchange: Ticker | `331277` / `SP_EXCHANGE_TICKER` | required point-in-time identifier |

The importer reads `FY+1`, `FY0` and the as-of date from the workbook's `SPGLabel` formulas. It deterministically derives each target `fiscal_period` by shifting the companion FY0 Period Ended date by one year. The date-only estimate snapshot is conservatively available at `23:59:59.999999` on the as-of date. The source manifest records the relative period code and derivation policy.

The 500-company validation produced 476 usable estimate observations. Capital IQ returned no numeric EPS value for 24 companies; those rows were recorded as `NULL_REQUIRED_VALUE` and rejected rather than imputed. Historical monthly estimate exports must repeat this same self-contained template at each signal cutoff.
