from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from .config import Settings
from .schemas import DataQualityIssue, DatasetKind, ImportResult, Severity
from .storage import Store

ALIASES: dict[str, tuple[str, ...]] = {
    "company_id": (
        "company_id",
        "companyid",
        "iq_company_id",
        "spgiq_company_id",
        "entity_id",
        "sp_entity_id",
        "spciq_id",
    ),
    "ticker": (
        "ticker",
        "trading_symbol",
        "symbol",
        "exchange_ticker",
        "sp_exchange_ticker",
    ),
    "company_name": (
        "company_name",
        "company",
        "name",
        "entity_name",
        "sp_entity_name",
    ),
    "sector": ("sector", "gics_sector", "industry_sector", "iq_sector"),
    "currency": ("currency", "reporting_currency"),
    "effective_at": (
        "effective_at",
        "speffectivedate",
        "sp_effective_date",
        "filing_date",
        "financial_filing_date",
        "iq_finl_filing_date",
        "available_at",
        "data_available_date",
    ),
    "as_of_date": ("as_of_date", "asofdate", "snapshot_date", "export_date"),
    "member_from": ("member_from", "membership_start", "start_date", "effective_from"),
    "member_to": ("member_to", "membership_end", "end_date", "effective_to"),
    "index_code": ("index_code", "index", "index_name"),
    "period_end": ("period_end", "periodend", "fiscal_period_end", "iq_period_end"),
    "period_type": ("period_type", "periodtype", "frequency"),
    "metric": ("metric", "data_item", "item", "field"),
    "value": ("value", "metric_value", "data_value"),
    "unit": ("unit", "units"),
    "fiscal_period": ("fiscal_period", "fiscal_period_end", "estimate_period"),
    "valid_to": ("valid_to", "sptodate", "sp_to_date"),
    "price_date": ("price_date", "date", "trading_date"),
    "open": ("open", "open_price"),
    "high": ("high", "high_price"),
    "low": ("low", "low_price"),
    "close": ("close", "close_price"),
    "adjusted_close": ("adjusted_close", "adj_close", "adjustedclose"),
    "volume": ("volume", "trading_volume"),
    "source": ("source", "price_source"),
    "institutional_pct": ("institutional_pct", "institutional_ownership_pct"),
    "institutional_change": ("institutional_change", "institutional_ownership_change"),
    "transaction_date": ("transaction_date", "trade_date"),
    "transaction_type": ("transaction_type", "type", "trade_type"),
    "shares": ("shares", "share_count"),
}

# Capital IQ exports use keyfield labels while hand-curated fixtures and other
# enterprise feeds often use plain-English names.  The raw header is retained in
# the immutable source archive; this map gives the analytical layer one stable
# metric vocabulary.
CANONICAL_METRIC_ALIASES: dict[str, str] = {
    "iq_total_rev": "revenue",
    "iq_total_revenue": "revenue",
    "total_revenue": "revenue",
    "iq_net_income": "net_income",
    "iq_fcf": "free_cash_flow",
    "iq_free_cash_flow": "free_cash_flow",
    "iq_cash_from_ops": "operating_cash_flow",
    "iq_cash_from_oper": "operating_cash_flow",
    "iq_operating_cash_flow": "operating_cash_flow",
    "cash_from_operations": "operating_cash_flow",
    "iq_ebitda": "ebitda",
    "iq_nopat": "nopat",
    "iq_invested_capital": "invested_capital",
    "iq_gross_profit": "gross_profit",
    "iq_total_assets": "total_assets",
    "iq_net_debt": "net_debt",
    "iq_market_cap": "market_cap",
    "market_capitalization": "market_cap",
    "iq_tev": "enterprise_value",
    "iq_total_enterprise_value": "enterprise_value",
    "total_enterprise_value": "enterprise_value",
    "iq_diluted_eps": "eps",
    "iq_eps_diluted": "eps",
    "diluted_eps": "eps",
    "iq_operating_margin": "operating_margin",
    "iq_oper_margin": "operating_margin",
    "sp_norm_eps_act_or_est": "eps_estimate",
    "sp_normalized_eps_actual_estimate": "eps_estimate",
    "sp_revenue_estimate": "revenue_estimate",
    "iq_eps_revision_1m": "eps_revision_1m",
    "eps_revision_1_month": "eps_revision_1m",
    "iq_eps_revision_3m": "eps_revision_3m",
    "eps_revision_3_month": "eps_revision_3m",
    "iq_estimate_surprise": "estimate_surprise",
}


def _column_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


@dataclass(frozen=True)
class Contract:
    table: str
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()
    date_columns: tuple[str, ...] = ()
    datetime_columns: tuple[str, ...] = ("effective_at",)
    defaults: dict[str, object] | None = None


CONTRACTS: dict[DatasetKind, Contract] = {
    DatasetKind.INSTRUMENTS: Contract(
        "instruments",
        ("company_id", "ticker", "company_name", "sector", "effective_at", "as_of_date"),
        ("currency",),
        date_columns=("as_of_date",),
        defaults={"currency": "USD"},
    ),
    DatasetKind.INDEX_MEMBERSHIP: Contract(
        "index_membership",
        (
            "company_id",
            "ticker",
            "member_from",
            "effective_at",
            "as_of_date",
        ),
        ("member_to", "index_code"),
        date_columns=("member_from", "member_to", "as_of_date"),
        defaults={"index_code": "SP500"},
    ),
    DatasetKind.FUNDAMENTALS: Contract(
        "fundamentals",
        (
            "company_id",
            "ticker",
            "period_end",
            "period_type",
            "effective_at",
            "as_of_date",
            "metric",
            "value",
        ),
        ("unit",),
        date_columns=("period_end", "as_of_date"),
    ),
    DatasetKind.ESTIMATES: Contract(
        "estimates",
        (
            "company_id",
            "ticker",
            "fiscal_period",
            "effective_at",
            "as_of_date",
            "metric",
            "value",
        ),
        ("valid_to", "unit"),
        date_columns=("fiscal_period", "as_of_date"),
        datetime_columns=("effective_at", "valid_to"),
    ),
    DatasetKind.PRICES: Contract(
        "prices",
        (
            "company_id",
            "ticker",
            "price_date",
            "open",
            "high",
            "low",
            "close",
            "adjusted_close",
            "effective_at",
            "as_of_date",
        ),
        ("volume", "source"),
        date_columns=("price_date", "as_of_date"),
        defaults={"source": "capital_iq"},
    ),
    DatasetKind.OWNERSHIP: Contract(
        "ownership",
        ("company_id", "ticker", "effective_at", "as_of_date"),
        ("institutional_pct", "institutional_change"),
        date_columns=("as_of_date",),
    ),
    DatasetKind.INSIDER_TRANSACTIONS: Contract(
        "insider_transactions",
        (
            "company_id",
            "ticker",
            "transaction_date",
            "effective_at",
            "as_of_date",
            "transaction_type",
            "shares",
        ),
        ("value",),
        date_columns=("transaction_date", "as_of_date"),
    ),
}


TABLE_COLUMNS: dict[str, list[str]] = {
    "instruments": [
        "company_id",
        "ticker",
        "company_name",
        "sector",
        "currency",
        "effective_at",
        "as_of_date",
        "source_file_id",
        "ingested_at",
    ],
    "index_membership": [
        "company_id",
        "ticker",
        "index_code",
        "member_from",
        "member_to",
        "effective_at",
        "as_of_date",
        "source_file_id",
        "ingested_at",
    ],
    "fundamentals": [
        "company_id",
        "ticker",
        "period_end",
        "period_type",
        "effective_at",
        "as_of_date",
        "metric",
        "value",
        "unit",
        "source_file_id",
        "ingested_at",
    ],
    "estimates": [
        "company_id",
        "ticker",
        "fiscal_period",
        "effective_at",
        "valid_to",
        "as_of_date",
        "metric",
        "value",
        "unit",
        "source_file_id",
        "ingested_at",
    ],
    "prices": [
        "company_id",
        "ticker",
        "price_date",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "source",
        "effective_at",
        "as_of_date",
        "source_file_id",
        "ingested_at",
    ],
    "ownership": [
        "company_id",
        "ticker",
        "effective_at",
        "as_of_date",
        "institutional_pct",
        "institutional_change",
        "source_file_id",
        "ingested_at",
    ],
    "insider_transactions": [
        "company_id",
        "ticker",
        "transaction_date",
        "effective_at",
        "as_of_date",
        "transaction_type",
        "shares",
        "value",
        "source_file_id",
        "ingested_at",
    ],
}


class CapitalIQImporter:
    def __init__(self, store: Store, settings: Settings):
        self.store = store
        self.settings = settings

    @staticmethod
    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def read(path: Path) -> pd.DataFrame:
        frame, _ = CapitalIQImporter.read_with_metadata(path)
        return frame

    @staticmethod
    def read_with_metadata(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path), {}
        if suffix in {".xlsx", ".xlsm", ".xls"}:
            preview = pd.read_excel(path, header=None, nrows=25)
            header_row = CapitalIQImporter._detect_header_row(preview)
            frame = pd.read_excel(path, header=header_row)
            metadata = CapitalIQImporter._excel_formula_metadata(path)
            return frame.dropna(how="all").reset_index(drop=True), metadata
        raise ValueError("Capital IQ import supports CSV, XLSX, XLSM, and XLS")

    @staticmethod
    def _parse_spg_formula_metadata(formulas: Iterable[str]) -> dict[str, object]:
        as_of_dates: set[str] = set()
        period_codes: set[str] = set()
        period_codes_by_keyfield: dict[str, set[str]] = {}
        for formula in formulas:
            if "SPGLabel" not in formula:
                continue
            for month, day, year in re.findall(
                r'"(?:<>|<=|>=)?(\d{1,2})/(\d{1,2})/(\d{4})"', formula
            ):
                as_of_dates.add(date(int(year), int(month), int(day)).isoformat())
            match = re.search(
                r'SPGLabel\([^,]+,\s*([^,]+),\s*"([A-Za-z]+[+-]?\d+)"',
                formula,
            )
            if match:
                keyfield = match.group(1).strip()
                period_code = match.group(2).upper()
                period_codes.add(period_code)
                period_codes_by_keyfield.setdefault(keyfield, set()).add(period_code)
        metadata: dict[str, object] = {}
        if len(as_of_dates) == 1:
            metadata["embedded_as_of_date"] = next(iter(as_of_dates))
        elif as_of_dates:
            metadata["embedded_as_of_dates"] = sorted(as_of_dates)
        if period_codes:
            metadata["embedded_period_codes"] = sorted(period_codes)
            metadata["embedded_period_codes_by_keyfield"] = {
                keyfield: sorted(codes)
                for keyfield, codes in sorted(period_codes_by_keyfield.items())
            }
        return metadata

    @staticmethod
    def _excel_formula_metadata(path: Path) -> dict[str, object]:
        if path.suffix.lower() == ".xls":
            return {}
        from openpyxl import load_workbook

        workbook = load_workbook(path, read_only=True, data_only=False)
        try:
            sheet = workbook[workbook.sheetnames[0]]
            formulas = [
                value
                for row in sheet.iter_rows(min_row=1, max_row=12, values_only=True)
                for value in row
                if isinstance(value, str) and value.startswith("=")
            ]
            return CapitalIQImporter._parse_spg_formula_metadata(formulas)
        finally:
            workbook.close()

    @staticmethod
    def _detect_header_row(preview: pd.DataFrame) -> int:
        """Locate the real table header below Capital IQ's Excel formula preamble."""
        known = {alias for aliases in ALIASES.values() for alias in aliases} | set(ALIASES)
        identity = {
            "company_id",
            "companyid",
            "iq_company_id",
            "spgiq_company_id",
            "entity_id",
            "sp_entity_id",
            "spciq_id",
            "ticker",
            "exchange_ticker",
            "sp_exchange_ticker",
            "metric",
            "price_date",
        }
        best_row = 0
        best_score = 0
        for row_number, row in preview.iterrows():
            keys = {
                _column_key(value)
                for value in row.tolist()
                if pd.notna(value) and str(value).strip()
            }
            score = len(keys & known)
            if score > best_score and keys & identity:
                best_row = int(row_number)
                best_score = score
        return best_row if best_score >= 2 else 0

    @staticmethod
    def _rename_columns(frame: pd.DataFrame) -> pd.DataFrame:
        keyed = {_column_key(column): column for column in frame.columns}
        renames: dict[str, str] = {}
        for canonical, aliases in ALIASES.items():
            for alias in aliases:
                if alias in keyed:
                    renames[keyed[alias]] = canonical
                    break
        return frame.rename(columns=renames)

    @staticmethod
    def _normalize_identifiers(frame: pd.DataFrame) -> pd.DataFrame:
        normalized = frame.copy()
        if "company_id" in normalized.columns:
            normalized["company_id"] = normalized["company_id"].map(
                lambda value: (
                    None
                    if pd.isna(value)
                    else str(int(value))
                    if isinstance(value, float) and value.is_integer()
                    else str(value).strip()
                )
            )
        if "ticker" in normalized.columns:
            normalized["ticker"] = (
                normalized["ticker"].astype("string").str.strip().str.rsplit(":", n=1).str[-1]
            )
        if "company_name" in normalized.columns:
            normalized["company_name"] = (
                normalized["company_name"]
                .astype("string")
                .str.strip()
                .str.replace(r"\s+\([A-Za-z0-9._-]+:[^)]+\)$", "", regex=True)
            )
        if "sector" in normalized.columns:
            normalized["sector"] = normalized["sector"].astype("string").str.strip()
        return normalized

    @staticmethod
    def _drop_parameter_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        if "company_id" not in frame.columns:
            return frame, 0
        identity_columns = [
            column for column in ("company_id", "ticker", "company_name") if column in frame.columns
        ]
        missing_identity = frame[identity_columns].isna().all(axis=1)

        def is_parameter_row(row: pd.Series) -> bool:
            values = [
                str(value).strip()
                for value in row.tolist()
                if pd.notna(value) and str(value).strip()
            ]
            return bool(values) and all(
                re.fullmatch(r"[A-Za-z]+[+-]?\d+", value) for value in values
            )

        parameter_rows = missing_identity & frame.apply(is_parameter_row, axis=1)
        return frame.loc[~parameter_rows].copy(), int(parameter_rows.sum())

    @staticmethod
    def _derive_estimate_fiscal_period(
        frame: pd.DataFrame, source_metadata: dict[str, object]
    ) -> tuple[pd.DataFrame, str | None]:
        """Derive a dated estimate target from CIQ's relative FY/FQ period code.

        Capital IQ estimate fields identify the target as FY+1/FQ+1 in the
        SPGLabel formula.  A companion point-in-time Period Ended field supplies
        each company's FY0/FQ0 date.  The relative shift is deterministic and
        keeps the exported file self-contained.
        """
        if "fiscal_period" in frame.columns or "period_end" not in frame.columns:
            return frame, None
        raw_codes = source_metadata.get("embedded_period_codes")
        if not isinstance(raw_codes, list):
            return frame, None
        parsed: list[tuple[str, int, str]] = []
        for raw_code in raw_codes:
            match = re.fullmatch(r"([A-Za-z]+)([+-]?\d+)", str(raw_code))
            if not match:
                continue
            prefix = match.group(1).upper()
            offset = int(match.group(2))
            if offset != 0 and prefix in {"FY", "FQ"}:
                parsed.append((prefix, offset, str(raw_code).upper()))
        if len(parsed) != 1:
            return frame, None

        prefix, offset, period_code = parsed[0]
        months = offset * (12 if prefix == "FY" else 3)
        derived = frame.copy()
        period_end = pd.to_datetime(derived["period_end"], errors="coerce")
        derived["fiscal_period"] = period_end + pd.DateOffset(months=months)
        return derived, period_code

    @staticmethod
    def _wide_to_long(frame: pd.DataFrame, dataset: DatasetKind) -> pd.DataFrame:
        if dataset not in {DatasetKind.FUNDAMENTALS, DatasetKind.ESTIMATES}:
            return frame
        if {"metric", "value"}.issubset(frame.columns):
            return frame
        identity = {
            "company_id",
            "ticker",
            "company_name",
            "period_end",
            "period_type",
            "fiscal_period",
            "effective_at",
            "valid_to",
            "as_of_date",
            "unit",
        }
        value_columns = [column for column in frame.columns if column not in identity]
        if not value_columns:
            return frame
        return frame.melt(
            id_vars=[column for column in frame.columns if column in identity],
            value_vars=value_columns,
            var_name="metric",
            value_name="value",
        )

    def import_file(
        self,
        path: str | Path,
        dataset: DatasetKind,
        *,
        current_snapshot_as_of: date | None = None,
        current_snapshot_effective_at: datetime | None = None,
    ) -> ImportResult:
        current_snapshot = any(
            value is not None for value in (current_snapshot_as_of, current_snapshot_effective_at)
        )
        if current_snapshot:
            if dataset is not DatasetKind.INSTRUMENTS:
                raise ValueError("Current-snapshot timestamps are permitted only for instruments")
            if current_snapshot_as_of is None or current_snapshot_effective_at is None:
                raise ValueError(
                    "Both current_snapshot_as_of and current_snapshot_effective_at are required"
                )
        if self.store.is_cloud and not self.settings.ciq_cloud_storage_confirmed:
            raise RuntimeError(
                "Set CIQ_CLOUD_STORAGE_CONFIRMED=true only after confirming that "
                "your NTU/S&P agreement permits storing Capital IQ values in Supabase"
            )
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        digest = self.sha256(source)
        existing = self.store.source_by_hash(digest)
        if existing:
            return ImportResult(
                source_file_id=str(existing["source_file_id"]),
                dataset=dataset,
                sha256=digest,
                archived_path=str(existing["archived_path"]),
                imported_rows=int(existing["row_count"]),
                idempotent=True,
            )

        source_file_id = f"src_{digest[:24]}"
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", source.name)
        archived = self.settings.raw_data_dir / dataset.value / f"{digest[:12]}_{safe_name}"
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, archived)

        frame, source_metadata = self.read_with_metadata(archived)
        source_column_keys = {_column_key(column) for column in frame.columns}
        date_only_availability = bool(
            source_column_keys & {"filing_date", "financial_filing_date", "iq_finl_filing_date"}
        )
        if date_only_availability:
            source_metadata["effective_at_granularity"] = "date"
            source_metadata["date_only_availability_policy"] = "end_of_day"
        frame = self._rename_columns(frame)
        frame = self._normalize_identifiers(frame)
        frame, parameter_rows_dropped = self._drop_parameter_rows(frame)
        if current_snapshot:
            if "as_of_date" not in frame.columns:
                frame["as_of_date"] = current_snapshot_as_of
            if "effective_at" not in frame.columns:
                frame["effective_at"] = current_snapshot_effective_at
        elif "as_of_date" not in frame.columns and source_metadata.get("embedded_as_of_date"):
            frame["as_of_date"] = source_metadata["embedded_as_of_date"]
        if (
            dataset is DatasetKind.ESTIMATES
            and "effective_at" not in frame.columns
            and source_metadata.get("embedded_as_of_date")
        ):
            frame["effective_at"] = source_metadata["embedded_as_of_date"]
            date_only_availability = True
            source_metadata["effective_at_granularity"] = "date"
            source_metadata["date_only_availability_policy"] = "end_of_day"
            source_metadata["estimate_availability_source"] = "embedded_spg_as_of_date"
        if dataset is DatasetKind.FUNDAMENTALS and "period_type" not in frame.columns:
            period_codes = source_metadata.get("embedded_period_codes")
            prefixes = (
                {re.match(r"[A-Za-z]+", code).group(0) for code in period_codes}
                if isinstance(period_codes, list) and period_codes
                else set()
            )
            if len(prefixes) == 1:
                frame["period_type"] = next(iter(prefixes))
        if dataset is DatasetKind.ESTIMATES:
            frame, estimate_period_code = self._derive_estimate_fiscal_period(
                frame, source_metadata
            )
            if estimate_period_code:
                source_metadata["estimate_period_code"] = estimate_period_code
                source_metadata["estimate_fiscal_period_policy"] = (
                    "shift_companion_period_end_by_spg_period_code"
                )
        frame = self._wide_to_long(frame, dataset)
        if "metric" in frame.columns:
            frame["metric"] = frame["metric"].map(
                lambda value: CANONICAL_METRIC_ALIASES.get(_column_key(value), _column_key(value))
            )
        contract = CONTRACTS[dataset]
        issues: list[DataQualityIssue] = []
        missing = [column for column in contract.required if column not in frame.columns]
        if missing:
            issue = DataQualityIssue(
                severity=Severity.ERROR,
                dataset=dataset.value,
                code="MISSING_REQUIRED_COLUMNS",
                message=f"Missing required columns: {', '.join(missing)}",
                source_file_id=source_file_id,
            )
            self.store.record_issue(issue)
            self.store.register_source_file(
                source_file_id=source_file_id,
                dataset=dataset.value,
                original_name=source.name,
                archived_path=str(archived),
                sha256=digest,
                row_count=0,
                metadata={"status": "rejected", "missing_columns": missing},
            )
            raise ValueError(issue.message)

        for column, value in (contract.defaults or {}).items():
            if column not in frame.columns:
                frame[column] = value
        for column in contract.optional:
            if column not in frame.columns:
                frame[column] = None

        for column in contract.date_columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.date
        for column in contract.datetime_columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True).dt.tz_localize(
                None
            )
            if column == "effective_at" and date_only_availability:
                frame[column] = (
                    frame[column].dt.normalize()
                    + pd.Timedelta(days=1)
                    - pd.Timedelta(microseconds=1)
                )

        numeric_columns = {
            "value",
            "open",
            "high",
            "low",
            "close",
            "adjusted_close",
            "volume",
            "institutional_pct",
            "institutional_change",
            "shares",
        }
        for column in numeric_columns.intersection(frame.columns):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        required_nulls = frame[list(contract.required)].isna().any(axis=1)
        rejected_rows = int(required_nulls.sum())
        if rejected_rows:
            value_only_missing = pd.Series(False, index=frame.index)
            if dataset in {DatasetKind.FUNDAMENTALS, DatasetKind.ESTIMATES}:
                non_value_required = [column for column in contract.required if column != "value"]
                value_only_missing = frame["value"].isna() & frame[non_value_required].notna().all(
                    axis=1
                )
            fatal_count = int((required_nulls & ~value_only_missing).sum())
            unavailable_count = int((required_nulls & value_only_missing).sum())
            if unavailable_count:
                issue = DataQualityIssue(
                    severity=Severity.WARNING,
                    dataset=dataset.value,
                    code="NULL_REQUIRED_VALUE",
                    message=(
                        f"Rejected {unavailable_count} rows whose source returned no "
                        "numeric observation"
                    ),
                    source_file_id=source_file_id,
                )
                issues.append(issue)
                self.store.record_issue(issue)
            if fatal_count:
                issue = DataQualityIssue(
                    severity=Severity.ERROR,
                    dataset=dataset.value,
                    code="NULL_REQUIRED_VALUE",
                    message=f"Rejected {fatal_count} rows with invalid required identity/time values",
                    source_file_id=source_file_id,
                )
                issues.append(issue)
                self.store.record_issue(issue)
            frame = frame.loc[~required_nulls].copy()

        cutoff = pd.to_datetime(frame["as_of_date"]).dt.normalize() + pd.Timedelta(days=1)
        future = pd.to_datetime(frame["effective_at"]) >= cutoff
        if future.any():
            count = int(future.sum())
            issue = DataQualityIssue(
                severity=Severity.ERROR,
                dataset=dataset.value,
                code="FUTURE_EFFECTIVE_TIMESTAMP",
                message=f"Rejected {count} rows whose effective_at is after as_of_date",
                source_file_id=source_file_id,
            )
            issues.append(issue)
            self.store.record_issue(issue)
            frame = frame.loc[~future].copy()
            rejected_rows += count

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        frame["source_file_id"] = source_file_id
        frame["ingested_at"] = now
        selected = frame[TABLE_COLUMNS[contract.table]].copy()
        self.store.insert_frame(contract.table, selected)
        self.store.register_source_file(
            source_file_id=source_file_id,
            dataset=dataset.value,
            original_name=source.name,
            archived_path=str(archived),
            sha256=digest,
            row_count=len(selected),
            metadata={
                "status": "accepted",
                "rejected_rows": rejected_rows,
                "columns": list(selected.columns),
                "source_scope": "current_snapshot" if current_snapshot else "point_in_time_export",
                "timestamp_provenance": (
                    "operator_supplied_export_timestamp"
                    if current_snapshot
                    else "embedded_spg_formula_and_source_columns"
                    if source_metadata.get("embedded_as_of_date")
                    else "source_columns"
                ),
                "source_formula_metadata": source_metadata,
                "parameter_rows_dropped": parameter_rows_dropped,
            },
        )
        return ImportResult(
            source_file_id=source_file_id,
            dataset=dataset,
            sha256=digest,
            archived_path=str(archived),
            imported_rows=len(selected),
            rejected_rows=rejected_rows,
            issues=issues,
        )


def certify_point_in_time(
    store: Store, start_date, end_date, required: Iterable[str] | None = None
) -> tuple[bool, list[str]]:
    required = tuple(required or ("index_membership", "fundamentals", "estimates", "prices"))
    notes: list[str] = []
    coverage_failures: list[str] = []
    status = {row["dataset"]: row for row in store.source_status()}
    synthetic = store.query_df(
        "SELECT COUNT(*) AS count FROM source_files WHERE original_name LIKE ?",
        ["synthetic_%"],
    )
    if not synthetic.empty and int(synthetic.iloc[0]["count"]):
        notes.append(
            "SYNTHETIC DATA ONLY: engineering validation; not a Capital IQ investment result"
        )
    missing = [dataset for dataset in required if dataset not in status]
    if missing:
        notes.append(f"Missing authoritative datasets: {', '.join(missing)}")

    if "fundamentals" in status:
        fundamental_dates = store.query_df(
            "SELECT MIN(effective_at) AS first, MAX(effective_at) AS last FROM fundamentals"
        )
        if fundamental_dates.empty or pd.isna(fundamental_dates.iloc[0]["first"]):
            coverage_failures.append("No point-in-time fundamental observations")
        else:
            first_fundamental = pd.Timestamp(fundamental_dates.iloc[0]["first"]).date()
            last_fundamental = pd.Timestamp(fundamental_dates.iloc[0]["last"]).date()
            if first_fundamental > start_date:
                coverage_failures.append(
                    "Point-in-time fundamentals begin after the requested backtest start"
                )
            # Annual statements can legitimately be several months old at a signal date.
            # A 15-month freshness allowance catches a current-only snapshot without
            # requiring quarterly coverage from every issuer.
            minimum_recent_filing = (pd.Timestamp(end_date) - pd.DateOffset(months=15)).date()
            if last_fundamental < minimum_recent_filing:
                coverage_failures.append(
                    "Point-in-time fundamentals end too early for the requested backtest end"
                )

    if "estimates" in status:
        estimate_dates = store.query_df(
            """
            SELECT DISTINCT effective_at FROM estimates
            WHERE effective_at >= ? AND effective_at < ? ORDER BY effective_at
            """,
            [
                datetime.combine(start_date, datetime.min.time()),
                datetime.combine(end_date + pd.Timedelta(days=1), datetime.min.time()),
            ],
        )
        observed_months = {
            pd.Timestamp(value).to_period("M")
            for value in estimate_dates.get("effective_at", [])
            if not pd.isna(value)
        }
        expected_months = set(pd.period_range(start_date, end_date, freq="M"))
        missing_months = sorted(expected_months - observed_months)
        if missing_months:
            preview = ", ".join(str(month) for month in missing_months[:6])
            suffix = " ..." if len(missing_months) > 6 else ""
            coverage_failures.append(
                "Point-in-time estimates are missing "
                f"{len(missing_months)} monthly signal snapshot(s): {preview}{suffix}"
            )

    notes.extend(coverage_failures)

    error_placeholders = ",".join("?" for _ in required)
    errors = store.query_df(
        f"""
        SELECT dataset, code, COUNT(*) AS count
        FROM data_quality_issues
        WHERE severity = 'error'
          AND dataset IN ({error_placeholders})
          AND NOT (dataset = 'estimates' AND code = 'NULL_REQUIRED_VALUE')
        GROUP BY dataset, code ORDER BY dataset, code
        """,
        list(required),
    )
    for row in errors.to_dict(orient="records"):
        notes.append(f"{row['dataset']}: {row['code']} ({row['count']} rows/files)")

    coverage = store.query_df(
        """
        SELECT MIN(member_from) AS first_member, MAX(member_to) AS last_member,
               SUM(CASE WHEN member_to IS NULL THEN 1 ELSE 0 END) AS active_members
        FROM index_membership WHERE index_code = 'SP500'
        """
    )
    if coverage.empty or pd.isna(coverage.iloc[0]["first_member"]):
        notes.append("No historical S&P 500 membership coverage")
    else:
        first_member = pd.Timestamp(coverage.iloc[0]["first_member"]).date()
        last_member = (
            None
            if pd.isna(coverage.iloc[0]["last_member"])
            else pd.Timestamp(coverage.iloc[0]["last_member"]).date()
        )
        if first_member > start_date:
            notes.append("Historical membership begins after the requested backtest start")
        elif (
            not int(coverage.iloc[0]["active_members"] or 0)
            and last_member
            and last_member < end_date
        ):
            notes.append("Historical membership ends before the requested backtest end")

        membership_sources = store.query_df(
            "SELECT metadata_json FROM source_files WHERE dataset = 'index_membership'"
        )
        substitutes: set[str] = set()
        for raw_metadata in membership_sources.get("metadata_json", []):
            metadata = raw_metadata
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}
            if isinstance(metadata, dict) and metadata.get("capital_iq_data") is False:
                substitutes.add(str(metadata.get("provider") or "unknown_public_source"))
        if substitutes:
            notes.append(
                "Membership coverage uses explicitly labelled non-Capital-IQ source(s): "
                + ", ".join(sorted(substitutes))
            )

    first_price, last_price = store.price_coverage()
    if first_price is None or last_price is None:
        notes.append("No price history")
    else:
        if first_price > start_date or last_price < end_date:
            notes.append("Price history does not span the requested backtest window")
        substitutes = [str(value) for value in store.price_sources() if str(value) != "capital_iq"]
        if substitutes:
            notes.append(
                "Price coverage uses explicitly labelled non-Capital-IQ source(s): "
                + ", ".join(substitutes)
            )

    certified = not missing and errors.empty and not coverage_failures
    certified = certified and not any(note.startswith("No ") for note in notes)
    certified = certified and not any("does not span" in note for note in notes)
    certified = certified and not any("begins after" in note for note in notes)
    if certified:
        notes.append("Point-in-time availability, membership, and price coverage checks passed")
    return certified, notes
