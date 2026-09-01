from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import tempfile
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import httpx
import pandas as pd

from .config import Settings
from .schemas import DataQualityIssue, DatasetKind, ImportResult, Severity
from .storage import PRICE_COLUMNS, Store


def historical_symbol_map(store: Store, start: date, end: date) -> dict[str, str]:
    instruments = store.query_df(
        """
        SELECT company_id, ticker, effective_at
        FROM instruments ORDER BY effective_at DESC
        """
    )
    membership = store.query_df(
        """
        SELECT company_id, ticker, member_from
        FROM index_membership
        WHERE member_from <= ? AND (member_to IS NULL OR member_to >= ?)
        ORDER BY member_from DESC
        """,
        [end, start],
    )
    output: dict[str, str] = {}
    for frame in (instruments, membership):
        if frame.empty:
            continue
        for row in frame.itertuples(index=False):
            ticker = str(row.ticker).strip().upper()
            if ticker:
                output.setdefault(ticker, str(row.company_id))
    output["SPY"] = "BENCHMARK:SPY"
    return output


def historical_membership_intervals(
    store: Store,
) -> tuple[
    dict[tuple[str, str], list[tuple[date, date | None]]],
    dict[str, list[tuple[str, date, date | None]]],
]:
    """Index ticker validity used to reject recycled-symbol market data."""
    frame = store.query_df(
        """
        SELECT company_id, ticker, member_from, member_to
        FROM index_membership WHERE index_code = 'SP500'
        ORDER BY company_id, member_from
        """
    )
    by_security: dict[tuple[str, str], list[tuple[date, date | None]]] = {}
    by_company: dict[str, list[tuple[str, date, date | None]]] = {}
    for row in frame.itertuples(index=False):
        company_id = str(row.company_id)
        ticker = str(row.ticker).strip().upper()
        member_from = pd.Timestamp(row.member_from).date()
        member_to = None if pd.isna(row.member_to) else pd.Timestamp(row.member_to).date()
        by_security.setdefault((company_id, ticker), []).append((member_from, member_to))
        by_company.setdefault(company_id, []).append((ticker, member_from, member_to))
    return by_security, by_company


def ticker_at_price_date(
    company_id: str,
    requested_ticker: str,
    price_date: date,
    by_security: dict[tuple[str, str], list[tuple[date, date | None]]],
    by_company: dict[str, list[tuple[str, date, date | None]]],
    *,
    lookback_days: int = 430,
) -> str | None:
    """Return ticker-at-date or reject a price outside a known ticker's validity.

    The pre-membership allowance supplies momentum and risk lookback for a newly
    added company. Dates after a closed membership interval are rejected, which
    prevents a subsequently recycled ticker from being joined to the old issuer.
    """
    requested_ticker = requested_ticker.strip().upper()
    if requested_ticker == "SPY":
        return requested_ticker
    security_intervals = by_security.get((company_id, requested_ticker))
    if not security_intervals:
        return requested_ticker
    allowed = any(
        member_from - timedelta(days=lookback_days) <= price_date
        and (member_to is None or price_date <= member_to)
        for member_from, member_to in security_intervals
    )
    if not allowed:
        return None
    for ticker, member_from, member_to in by_company.get(company_id, []):
        if member_from <= price_date and (member_to is None or price_date <= member_to):
            return ticker
    return requested_ticker


def active_membership_return_outliers(
    frame: pd.DataFrame, store: Store, threshold: float = 3.0
) -> pd.DataFrame:
    """Find extreme adjusted moves only while the company is in the index."""
    ordered = frame.sort_values(["company_id", "ticker", "price_date"]).copy()
    adjusted_returns = ordered.groupby(["company_id", "ticker"])["adjusted_close"].pct_change()
    candidates = ordered.loc[adjusted_returns.abs() > threshold].copy()
    if candidates.empty:
        return candidates
    membership = store.query_df(
        "SELECT company_id, member_from, member_to FROM index_membership WHERE index_code = 'SP500'"
    )
    intervals: dict[str, list[tuple[date, date | None]]] = {}
    for row in membership.itertuples(index=False):
        intervals.setdefault(str(row.company_id), []).append(
            (
                pd.Timestamp(row.member_from).date(),
                None if pd.isna(row.member_to) else pd.Timestamp(row.member_to).date(),
            )
        )

    def is_active(row) -> bool:
        if str(row.ticker) == "SPY":
            return True
        day = pd.Timestamp(row.price_date).date()
        return any(
            member_from <= day and (member_to is None or member_to >= day)
            for member_from, member_to in intervals.get(str(row.company_id), [])
        )

    return candidates.loc[candidates.apply(is_active, axis=1)]


class AlpacaHistoricalPriceSync:
    """Download immutable, fully adjusted IEX daily bars from Alpaca.

    This is an explicitly labelled market-data substitution. It never identifies
    Alpaca observations as Capital IQ data and it does not use a live trading
    endpoint.
    """

    source = "alpaca_iex_adjusted"

    def __init__(
        self,
        store: Store,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.store = store
        self.settings = settings
        self.transport = transport

    def _headers(self) -> dict[str, str]:
        if not self.settings.alpaca_paper_key or not self.settings.alpaca_paper_secret:
            raise RuntimeError("ALPACA_PAPER_KEY and ALPACA_PAPER_SECRET are required")
        return {
            "APCA-API-KEY-ID": self.settings.alpaca_paper_key,
            "APCA-API-SECRET-KEY": self.settings.alpaca_paper_secret,
        }

    def _symbol_map(self, start: date, end: date) -> dict[str, str]:
        return historical_symbol_map(self.store, start, end)

    @staticmethod
    def _chunks(values: list[str], size: int) -> list[list[str]]:
        return [values[index : index + size] for index in range(0, len(values), size)]

    async def sync(
        self,
        start: date,
        end: date,
        *,
        batch_size: int = 50,
        sync_date: date | None = None,
    ) -> ImportResult:
        if end < start:
            raise ValueError("end must be on or after start")
        if not 1 <= batch_size <= 200:
            raise ValueError("batch_size must be between 1 and 200")

        symbol_map = self._symbol_map(start, end)
        by_security, by_company = historical_membership_intervals(self.store)
        symbols = sorted(symbol_map)
        if symbols == ["SPY"]:
            raise RuntimeError("No instrument or membership symbols are available")

        observed_symbols: set[str] = set()
        rows: list[dict[str, object]] = []
        request_pages = 0
        as_of_date = sync_date or datetime.now(timezone.utc).date()
        raw_dir = self.settings.raw_data_dir / "prices_alpaca"
        raw_dir.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix="alpaca_iex_",
                suffix=".jsonl",
                dir=raw_dir,
                delete=False,
            ) as temp_handle:
                temp_path = Path(temp_handle.name)
                async with httpx.AsyncClient(
                    base_url=self.settings.alpaca_data_base_url,
                    headers=self._headers(),
                    timeout=60,
                    transport=self.transport,
                ) as client:
                    for batch_number, batch in enumerate(
                        self._chunks(symbols, batch_size), start=1
                    ):
                        page_token: str | None = None
                        page_number = 0
                        while True:
                            params = {
                                "symbols": ",".join(batch),
                                "timeframe": "1Day",
                                "start": f"{start.isoformat()}T00:00:00Z",
                                "end": f"{(end + timedelta(days=1)).isoformat()}T00:00:00Z",
                                "adjustment": "all",
                                "feed": "iex",
                                "sort": "asc",
                                "limit": "10000",
                            }
                            if page_token:
                                params["page_token"] = page_token
                            response = await client.get("/v2/stocks/bars", params=params)
                            response.raise_for_status()
                            payload = response.json()
                            page_number += 1
                            request_pages += 1
                            temp_handle.write(
                                json.dumps(
                                    {
                                        "batch": batch_number,
                                        "page": page_number,
                                        "request": {
                                            "symbols": batch,
                                            "start": start.isoformat(),
                                            "end": end.isoformat(),
                                            "adjustment": "all",
                                            "feed": "iex",
                                        },
                                        "response": payload,
                                    },
                                    sort_keys=True,
                                    separators=(",", ":"),
                                )
                                + "\n"
                            )
                            for ticker, bars in (payload.get("bars") or {}).items():
                                normalized_ticker = str(ticker).strip().upper()
                                company_id = symbol_map.get(normalized_ticker)
                                if company_id is None:
                                    continue
                                observed_symbols.add(normalized_ticker)
                                for bar in bars:
                                    timestamp = pd.Timestamp(bar["t"])
                                    price_date = timestamp.date()
                                    ticker_at_date = ticker_at_price_date(
                                        company_id,
                                        normalized_ticker,
                                        price_date,
                                        by_security,
                                        by_company,
                                    )
                                    if ticker_at_date is None:
                                        continue
                                    rows.append(
                                        {
                                            "company_id": company_id,
                                            "ticker": ticker_at_date,
                                            "price_date": price_date,
                                            "open": float(bar["o"]),
                                            "high": float(bar["h"]),
                                            "low": float(bar["l"]),
                                            "close": float(bar["c"]),
                                            "adjusted_close": float(bar["c"]),
                                            "volume": float(bar.get("v") or 0.0),
                                            "source": self.source,
                                            "effective_at": datetime.combine(price_date, time.max),
                                            "as_of_date": as_of_date,
                                        }
                                    )
                            page_token = payload.get("next_page_token")
                            if not page_token:
                                break

            digest = hashlib.sha256(temp_path.read_bytes()).hexdigest()
            source_file_id = f"src_{digest[:24]}"
            archived = raw_dir / (
                f"{digest[:12]}_alpaca_iex_adjusted_bars_"
                f"{start.isoformat()}_{end.isoformat()}.jsonl"
            )
            frame = pd.DataFrame(rows, columns=PRICE_COLUMNS[:-2]).drop_duplicates(
                ["company_id", "ticker", "price_date"], keep="last"
            )
            frame["source_file_id"] = source_file_id
            frame["ingested_at"] = datetime.now(timezone.utc).replace(tzinfo=None)
            existing = self.store.source_by_hash(digest)
            if existing:
                temp_path.unlink(missing_ok=True)
                self.store.insert_frame("prices", frame)
                return ImportResult(
                    source_file_id=str(existing["source_file_id"]),
                    dataset=DatasetKind.PRICES,
                    sha256=digest,
                    archived_path=str(existing["archived_path"]),
                    imported_rows=int(existing["row_count"]),
                    idempotent=True,
                )
            shutil.move(str(temp_path), archived)
            self.store.insert_frame("prices", frame)
            self.store.register_source_file(
                source_file_id=source_file_id,
                dataset=DatasetKind.PRICES.value,
                original_name=archived.name,
                archived_path=str(archived),
                sha256=digest,
                row_count=len(frame),
                metadata={
                    "status": "accepted",
                    "provider": "alpaca",
                    "feed": "iex",
                    "adjustment": "all",
                    "source": self.source,
                    "source_scope": "explicit_market_data_substitution",
                    "capital_iq_data": False,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "as_of_date": as_of_date.isoformat(),
                    "requested_symbols": len(symbols),
                    "observed_symbols": len(observed_symbols),
                    "request_pages": request_pages,
                },
            )

            issues: list[DataQualityIssue] = []
            missing_symbols = sorted(set(symbols) - observed_symbols)
            if missing_symbols:
                issue = DataQualityIssue(
                    severity=Severity.WARNING,
                    dataset=DatasetKind.PRICES.value,
                    code="ALPACA_SYMBOLS_WITHOUT_BARS",
                    message=(
                        f"No IEX bars returned for {len(missing_symbols)} symbols: "
                        + ", ".join(missing_symbols[:25])
                    ),
                    source_file_id=source_file_id,
                )
                self.store.record_issue(issue)
                issues.append(issue)
            outlier_rows = active_membership_return_outliers(frame, self.store)
            if not outlier_rows.empty:
                affected = sorted(set(outlier_rows["ticker"].astype(str)))
                issue = DataQualityIssue(
                    severity=Severity.ERROR,
                    dataset=DatasetKind.PRICES.value,
                    code="ADJUSTED_PRICE_RETURN_OUTLIER",
                    message=(
                        f"Detected {len(outlier_rows)} adjusted daily returns above 300% "
                        "for: " + ", ".join(affected[:25])
                    ),
                    source_file_id=source_file_id,
                )
                self.store.record_issue(issue)
                issues.append(issue)
            return ImportResult(
                source_file_id=source_file_id,
                dataset=DatasetKind.PRICES,
                sha256=digest,
                archived_path=str(archived),
                imported_rows=len(frame),
                issues=issues,
            )
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)


class YahooHistoricalPriceSync:
    """Download split/dividend-adjusted daily OHLC from Yahoo's chart feed."""

    source = "yahoo_adjusted"
    # Yahoo reassigns a symbol after a delisting. Paramount's continuous history
    # moved to PSKY, while PARA now resolves to an unrelated issuer.
    symbol_overrides = {"PARA": "PSKY", "VIAC": "PSKY"}

    def __init__(
        self,
        store: Store,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.store = store
        self.settings = settings
        self.transport = transport

    @staticmethod
    def _normalize_yahoo_symbol(ticker: str) -> str:
        return ticker.replace(".", "-")

    @classmethod
    def _yahoo_symbol(cls, ticker: str) -> str:
        return cls._normalize_yahoo_symbol(cls.symbol_overrides.get(ticker, ticker))

    async def sync(
        self,
        start: date,
        end: date,
        *,
        concurrency: int = 8,
        sync_date: date | None = None,
    ) -> ImportResult:
        if end < start:
            raise ValueError("end must be on or after start")
        if not 1 <= concurrency <= 16:
            raise ValueError("concurrency must be between 1 and 16")
        symbol_map = historical_symbol_map(self.store, start, end)
        by_security, by_company = historical_membership_intervals(self.store)
        symbols = sorted(symbol_map)
        as_of_date = sync_date or datetime.now(timezone.utc).date()
        period1 = int(datetime.combine(start, time.min, tzinfo=timezone.utc).timestamp())
        period2 = int(
            datetime.combine(end + timedelta(days=1), time.min, tzinfo=timezone.utc).timestamp()
        )
        semaphore = asyncio.Semaphore(concurrency)

        async with httpx.AsyncClient(
            base_url="https://query1.finance.yahoo.com",
            headers={"User-Agent": "Mozilla/5.0 institutional-quant-research"},
            timeout=45,
            transport=self.transport,
        ) as client:

            async def fetch(ticker: str) -> tuple[str, dict[str, object]]:
                async with semaphore:
                    yahoo_symbol = self._yahoo_symbol(ticker)
                    response = await client.get(
                        f"/v8/finance/chart/{quote(yahoo_symbol, safe='')}",
                        params={
                            "period1": str(period1),
                            "period2": str(period2),
                            "interval": "1d",
                            "events": "div,splits",
                            "includeAdjustedClose": "true",
                        },
                    )
                    if response.status_code != 200:
                        return ticker, {
                            "status": response.status_code,
                            "error": response.text[:500],
                        }
                    return ticker, response.json()

            responses = await asyncio.gather(*(fetch(ticker) for ticker in symbols))

        raw_dir = self.settings.raw_data_dir / "prices_yahoo"
        raw_dir.mkdir(parents=True, exist_ok=True)
        observed_symbols: set[str] = set()
        rows: list[dict[str, object]] = []
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix="yahoo_chart_",
            suffix=".jsonl",
            dir=raw_dir,
            delete=False,
        ) as temp_handle:
            temp_path = Path(temp_handle.name)
            for ticker, payload in responses:
                temp_handle.write(
                    json.dumps(
                        {
                            "ticker": ticker,
                            "request": {
                                "start": start.isoformat(),
                                "end": end.isoformat(),
                                "interval": "1d",
                                "adjustment": "adjclose_factor_applied_to_ohlc",
                            },
                            "response": payload,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                chart = payload.get("chart") if isinstance(payload, dict) else None
                results = chart.get("result") if isinstance(chart, dict) else None
                if not results:
                    continue
                result = results[0]
                timestamps = result.get("timestamp") or []
                indicators = result.get("indicators") or {}
                quotes = indicators.get("quote") or []
                adjusted = indicators.get("adjclose") or []
                if not timestamps or not quotes or not adjusted:
                    continue
                quote_values = quotes[0]
                adjusted_values = adjusted[0].get("adjclose") or []
                observed_symbols.add(ticker)
                for index, timestamp in enumerate(timestamps):
                    close = (quote_values.get("close") or [])[index]
                    adjusted_close = adjusted_values[index]
                    if close in (None, 0) or adjusted_close is None:
                        continue
                    factor = float(adjusted_close) / float(close)
                    values = {
                        field: (quote_values.get(field) or [None] * len(timestamps))[index]
                        for field in ("open", "high", "low", "close")
                    }
                    if any(value is None for value in values.values()):
                        continue
                    price_date = datetime.fromtimestamp(int(timestamp), tz=timezone.utc).date()
                    ticker_at_date = ticker_at_price_date(
                        symbol_map[ticker],
                        ticker,
                        price_date,
                        by_security,
                        by_company,
                    )
                    if ticker_at_date is None:
                        continue
                    rows.append(
                        {
                            "company_id": symbol_map[ticker],
                            "ticker": ticker_at_date,
                            "price_date": price_date,
                            "open": float(values["open"]) * factor,
                            "high": float(values["high"]) * factor,
                            "low": float(values["low"]) * factor,
                            "close": float(values["close"]) * factor,
                            "adjusted_close": float(adjusted_close),
                            "volume": float(
                                (quote_values.get("volume") or [0] * len(timestamps))[index] or 0
                            ),
                            "source": self.source,
                            "effective_at": datetime.combine(price_date, time.max),
                            "as_of_date": as_of_date,
                        }
                    )

        digest = hashlib.sha256(temp_path.read_bytes()).hexdigest()
        source_file_id = f"src_{digest[:24]}"
        archived = raw_dir / (
            f"{digest[:12]}_yahoo_adjusted_bars_{start.isoformat()}_{end.isoformat()}.jsonl"
        )
        frame = pd.DataFrame(rows, columns=PRICE_COLUMNS[:-2]).drop_duplicates(
            ["company_id", "ticker", "price_date"], keep="last"
        )
        frame["source_file_id"] = source_file_id
        frame["ingested_at"] = datetime.now(timezone.utc).replace(tzinfo=None)
        existing = self.store.source_by_hash(digest)
        if existing:
            temp_path.unlink(missing_ok=True)
            self.store.insert_frame("prices", frame)
            return ImportResult(
                source_file_id=str(existing["source_file_id"]),
                dataset=DatasetKind.PRICES,
                sha256=digest,
                archived_path=str(existing["archived_path"]),
                imported_rows=int(existing["row_count"]),
                idempotent=True,
            )
        shutil.move(str(temp_path), archived)
        self.store.insert_frame("prices", frame)
        self.store.register_source_file(
            source_file_id=source_file_id,
            dataset=DatasetKind.PRICES.value,
            original_name=archived.name,
            archived_path=str(archived),
            sha256=digest,
            row_count=len(frame),
            metadata={
                "status": "accepted",
                "provider": "yahoo_chart",
                "source": self.source,
                "source_scope": "explicit_market_data_substitution",
                "capital_iq_data": False,
                "ohlc_adjustment": "adjclose_divided_by_raw_close",
                "start": start.isoformat(),
                "end": end.isoformat(),
                "requested_symbols": len(symbols),
                "observed_symbols": len(observed_symbols),
            },
        )
        issues: list[DataQualityIssue] = []
        missing_symbols = sorted(set(symbols) - observed_symbols)
        if missing_symbols:
            issue = DataQualityIssue(
                severity=Severity.WARNING,
                dataset=DatasetKind.PRICES.value,
                code="YAHOO_SYMBOLS_WITHOUT_BARS",
                message=(
                    f"No Yahoo chart bars returned for {len(missing_symbols)} symbols: "
                    + ", ".join(missing_symbols[:25])
                ),
                source_file_id=source_file_id,
            )
            self.store.record_issue(issue)
            issues.append(issue)
        outlier_rows = active_membership_return_outliers(frame, self.store)
        if not outlier_rows.empty:
            affected = sorted(set(outlier_rows["ticker"].astype(str)))
            issue = DataQualityIssue(
                severity=Severity.ERROR,
                dataset=DatasetKind.PRICES.value,
                code="ADJUSTED_PRICE_RETURN_OUTLIER",
                message=(
                    f"Detected {len(outlier_rows)} active-membership adjusted daily returns "
                    "above 300% for: " + ", ".join(affected[:25])
                ),
                source_file_id=source_file_id,
            )
            self.store.record_issue(issue)
            issues.append(issue)
        return ImportResult(
            source_file_id=source_file_id,
            dataset=DatasetKind.PRICES,
            sha256=digest,
            archived_path=str(archived),
            imported_rows=len(frame),
            issues=issues,
        )
