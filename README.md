# S&P 500 Institutional Multi-Agent Quant Platform

Local-first, Supabase-backed research software for turning point-in-time S&P 500
exports into auditable factors, walk-forward portfolios and evidence-grounded
multi-agent research reports. It is an academic/self-use research platform, not
investment advice. Alpaca integration is paper-only.

## What is complete

- Supabase `institutional_quant` schema, immutable source manifests and SHA-256 provenance.
- Capital IQ importers for instruments, fundamentals, estimates, membership and prices.
- Six factor families: value, quality, growth, estimate revisions, momentum and low risk.
- Walk-forward ElasticNet and histogram gradient boosting models predicting next-month excess return vs SPY.
- Deterministic 20–30 name long-only optimizer with volatility, position and turnover controls.
- LangGraph evidence packet, four analyst roles, bull/bear opening + rebuttal, and consensus judge.
- FastAPI, Jinja/HTMX and Plotly UI for Data Status, Factor Lab, Research Runs, Debate, Portfolio, Backtest and Paper Trading.
- DeepSeek structured-output routing and cache metadata; only final validated rationale is retained.
- Alpaca paper preview/approval workflow; no live endpoint is accepted.
- Regression suite: **710 passed, 1 skipped** (optional Bedrock dependency absent).

The current real-data case study is frozen at `output/manifests/case-study-05f56a33-8e7f-4017-be54-d411467b8edb.json` (the directory is intentionally git-ignored because it can contain licensed-data metadata). It covers 60 monthly observations from September 2021 through August 2026, with 5/10/25 bps cost sensitivity. The 10 bps ensemble produced 11.95% CAGR, 12.07% volatility, Sharpe 1.00 and −10.35% maximum drawdown; SPY produced 12.49% CAGR, 15.19% volatility, Sharpe 0.85 and −23.31% drawdown. ML-only was weaker (6.38% CAGR), so the result is reported as a risk-adjusted trade-off, not as forced alpha.

Important limitation: the supplied Capital IQ sector field (`IQ_SECTOR`, KeyField 329067) is a current classification without a historical As-Of parameter in the verified export screen. Historical rows are therefore labeled `Unknown`; the case study is PIT-certified for availability, membership and prices, but **sector-neutral exposure is not certified** until a dated CIQ sector export is supplied. Membership and adjusted prices are explicitly labeled non-CIQ substitutes (public historical reconstruction, Alpaca IEX/Yahoo).

## Architecture

```mermaid
flowchart TD
  CIQ[Capital IQ Pro CSV/XLSX exports] --> ARCH[Immutable raw archive + SHA-256]
  PUB[Labelled public S&P 500 reconstruction] --> ARCH
  MKT[Adjusted Yahoo / Alpaca IEX bars] --> ARCH
  ARCH --> GATE[Schema + provenance + PIT gates]
  GATE --> DB[(Supabase Postgres<br/>institutional_quant schema)]
  GATE --> LAKE[(Local Parquet price lake)]
  DB --> F[Factor registry<br/>value · quality · growth · revisions · momentum · low risk]
  LAKE --> F
  DB --> ML[Walk-forward ElasticNet + HGB]
  LAKE --> ML
  F --> RANK[Factor/ML rank]
  ML --> RANK
  RANK --> PACK[Immutable EvidencePacket]
  PACK --> A[Fundamental · valuation · estimates/peers · risk analysts]
  A --> D[Bull/bear opening + rebuttal]
  D --> J[Consensus judge<br/>five-tier rating + capped adjustment]
  J --> OPT[Deterministic portfolio optimizer]
  OPT --> REPORT[Reports, API and dashboard]
  OPT --> PREVIEW[Alpaca paper order preview]
  PREVIEW --> APPROVAL[Explicit user approval]
```

Python owns ratios, timestamps, factor scores, model windows, risk limits,
portfolio weights and orders. Agents explain and challenge the evidence; they
cannot invent data, override missing-data gates or submit orders.

## Author's design philosophy

1. **Institutional data is valuable only when its lineage survives.** Every row carries stable company identity, ticker-at-date, `effective_at`, `as_of_date`, source-file hash and ingestion time.
2. **The LLM is a research committee, not a calculator.** Deterministic Python calculations are reproducible and testable; the model contributes comparison, dissent and concise evidence-linked rationale.
3. **Debate must be falsifiable.** Bull and bear turns cite the same immutable packet, and the judge's score adjustment is bounded so rhetoric cannot dominate the quantitative rank.
4. **Honest underperformance is a valid result.** SPY, equal-weight, factor-only, ML-only and ensemble ablations are retained with transaction costs and statistical qualification; no test-period tuning is used to manufacture a win.
5. **Small and understandable beats over-engineered.** The platform is designed for learning and personal research, while keeping strict the few controls that matter: no look-ahead, no hidden data substitution, and paper-only execution.

## Data hierarchy and workflow

Capital IQ Pro is the primary company-research source. It supplies statements,
filing availability, consensus estimates/revisions, valuation inputs, peers,
ownership and (when licensed) research evidence. Market APIs are subordinate:
they supply return labels, momentum/volatility features, execution calendars and
paper-account synchronization. DeepSeek is a reasoning layer, never a source of
financial facts.

1. Export a Capital IQ screen as CSV/XLSX with stable entity IDs and all required availability fields; keep the original file locally.
2. Upload it on **Data Status** or call `sp500iq import-ciq`. The importer hashes, validates, normalizes units and writes to Supabase.
3. Run **Factor Lab** to inspect factor coverage, rank IC, quantile monotonicity, stability and correlations. A stock becomes investable only with at least four of six families and two institutional families.
4. Run the walk-forward backtest. Training, feature selection and hyperparameter selection use only observations available before each signal cutoff; execution is the next available session open.
5. Select roughly ten names for an EvidencePacket: new leaders, deteriorating holdings and factor/ML disagreements. Analysts work independently, then bull and bear researchers debate once and rebut once.
6. Review the judge's rating, dissent, uncertainties and evidence IDs. The optimizer converts the adjusted score into a 20–30 name proposal.
7. Review the **Paper Trading** preview. Submission is a separate, explicit user action and is accepted only by `https://paper-api.alpaca.markets`.

## Environment and quick start

```bash
cp .env.example .env
# Fill secrets locally; never commit .env.
export UV_PROJECT_ENVIRONMENT=/Users/guohuiwen/.cache/sp500-institutional-quant-venv
uv sync --extra dev
uv run sp500iq preflight
uv run sp500iq init-db
uv run sp500iq serve
```

Required production values are `SUPABASE_DB_URL`, `DEEPSEEK_API_KEY`,
`ALPACA_PAPER_KEY` and `ALPACA_PAPER_SECRET`. Set
`CIQ_EXTERNAL_PROCESSING_CONFIRMED=true` only after the NTU/S&P agreement permits
sending structured Capital IQ values to DeepSeek; set
`CIQ_CLOUD_STORAGE_CONFIRMED=true` only after cloud persistence is permitted.
The application never prints secret values. `IQ_DATABASE_BACKEND=supabase` is the
production default; DuckDB is reserved for synthetic tests and demos.

## Commands

```bash
uv run sp500iq import-ciq fundamentals /absolute/path/fundamentals.xlsx
uv run sp500iq import-ciq estimates /absolute/path/estimates.xlsx
uv run sp500iq sync-public-sp500-membership --start 2021-09-01 --end 2026-08-31 --refresh-identities
uv run sp500iq sync-yahoo-prices --start 2017-04-01 --end 2026-08-31
uv run sp500iq sync-alpaca-prices --start 2021-09-01 --end 2026-09-02
uv run sp500iq factor-snapshot --as-of 2026-09-01 --top 25
uv run sp500iq case-study
```

Open `http://127.0.0.1:8000`; OpenAPI is at `/docs`. For a no-key engineering
check use `uv run sp500iq synthetic-demo` and `uv run sp500iq backtest-demo`.

## Reproducibility and safety

- Raw exports and reports are ignored by Git; licensed files never enter tests.
- Case-study manifests record source hashes, code hash, model alias/version, reasoning level, prompt version and evidence hash.
- One retry is allowed for invalid structured output; malformed or uncited investment output fails closed.
- A changed model fingerprint starts a new experiment version; hidden reasoning is not persisted or displayed.
- Daily operations synchronize paper prices/fills and risk alerts; weekly updates can make limited adjustments; monthly runs refresh data, factors, agents and target weights.

## Attribution and licenses

The orchestration foundation is adapted from
[TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents),
pinned at commit `2448d0a12576f9b2ddcd5980a0630833423d1e1b`, under Apache-2.0.
The upstream source and license remain in `tradingagents/` and `LICENSE`.
The dashboard uses patterns from the MIT-licensed
[Tabler](https://github.com/tabler/tabler) UI kit; application code in
`institutional_quant/` is project-specific.

All Capital IQ data remains subject to the applicable S&P Global/NTU agreement.
This software is for research and education and does not provide personalized
investment advice.
