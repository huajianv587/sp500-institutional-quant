# Architecture and reproducibility

```text
Capital IQ exports -> immutable local archive -> PIT validation -> Supabase Postgres
                                                            |            |
Yahoo/Alpaca bars -> immutable local archive -> Parquet price lake       |
                                              |             |            |
                                              +---- deterministic factors + walk-forward ML
                                                                         |
                                                immutable EvidencePacket (structured values only)
                                                                         |
                                      LangGraph analysts -> debate -> consensus judge
                                                                         |
                                         capped adjustment -> deterministic optimizer
                                                                         |
                                              report / API / Alpaca paper preview
```

The `institutional_quant` Supabase schema holds normalized institutional observations, source manifests, data-quality issues, factors, model fingerprints, cached structured outputs, portfolios, jobs and backtests. Source files stay local. The high-volume daily price panel is a derived ZSTD-compressed Parquet lake. The shared store interface chooses Capital IQ over Yahoo and Yahoo over Alpaca for the same company-date, so an authoritative export can replace a substitute without changing factor or backtest code.

Ticker identity is point-in-time. Stable company IDs join renamed securities;
membership intervals supply ticker-at-date, and closed intervals reject later
re-use of the same ticker by another issuer. The current market-data QA contains
1,910,469 unique source observations from 2017-04-03 through 2026-08-31,
99.2% minimum month-end constituent coverage over the 60-month study window,
and zero accepted active-membership adjusted returns above 300%.

The factor engine uses sector-aware 2.5% winsorization and z-scores for value, quality, growth, estimate revisions, momentum and low risk. Financial ratios are deterministic. Walk-forward ElasticNet and histogram gradient boosting select settings using only past training/validation months.

The LangGraph has three stages: independent analysts, an optional two-round bull/bear debate, and an independent judge. Every accepted factual/numerical claim references an evidence ID. The LLM cannot change data gates, calculate weights or submit orders. Cache identity includes model alias/version metadata, prompt version and evidence hash.

The optimizer is long-only with a 5% stock cap, ±8 percentage-point sector active bands, an approximate 12% volatility target and a 20% monthly one-way turnover cap. Paper order submission is a separate, explicit approval boundary.
