from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone

import httpx
import pandas as pd

from .config import Settings
from .schemas import DataQualityIssue, DatasetKind, ImportResult, Severity
from .storage import Store


@dataclass(frozen=True)
class MembershipEvent:
    effective_date: date
    action: str
    ticker: str
    source_url: str
    announced_at: date | None = None
    reason: str | None = None


class PublicSP500MembershipSync:
    """Reconstruct company-level S&P 500 membership from auditable public sources.

    The historical event log is pinned to an immutable ``pitindex`` commit. Events
    after that build are explicit, source-linked overrides. This fallback is never
    represented as Capital IQ data.
    """

    upstream_commit = "c3d5d4961076a59041b3e1de90fe5ea052f61bb4"
    raw_base_url = f"https://raw.githubusercontent.com/arielNacamulli/pitindex/{upstream_commit}"
    upstream_files = (
        "pitindex/data/sp500_seed.csv",
        "pitindex/data/sp500_changes.csv",
        "pitindex/data/sp500_current.csv",
        "pitindex/data/build_metadata.json",
        "LICENSE",
    )
    supported_through = date(2026, 8, 31)
    source_name = "pitindex_spglobal_official_reconstruction"

    official_overrides = (
        MembershipEvent(
            date(2026, 8, 18),
            "removed",
            "AVB",
            "https://press.spglobal.com/2026-08-13-Reddit-Set-to-Join-S-P-500-and-Sun-Communities-to-Join-S-P-MidCap-400",
            date(2026, 8, 13),
            "Reddit replaced AvalonBay Communities in the S&P 500.",
        ),
        MembershipEvent(
            date(2026, 8, 18),
            "added",
            "RDDT",
            "https://press.spglobal.com/2026-08-13-Reddit-Set-to-Join-S-P-500-and-Sun-Communities-to-Join-S-P-MidCap-400",
            date(2026, 8, 13),
            "Reddit replaced AvalonBay Communities in the S&P 500.",
        ),
        MembershipEvent(
            date(2026, 8, 18),
            "removed",
            "EQR",
            "https://www.sec.gov/Archives/edgar/data/906107/000114036126033377/ef20080318_8k.htm",
            date(2026, 8, 17),
            "Equity Residential changed its name and NYSE ticker to VMRK.",
        ),
        MembershipEvent(
            date(2026, 8, 18),
            "added",
            "VMRK",
            "https://www.sec.gov/Archives/edgar/data/906107/000114036126033377/ef20080318_8k.htm",
            date(2026, 8, 17),
            "Equity Residential changed its name and NYSE ticker to VMRK.",
        ),
    )

    # Collapse known multiple share classes to the Capital IQ company entity.
    # BRK.B remains the tradable representative even though the current CIQ export
    # labelled the company BRK.A.
    canonical_groups = {
        "BRK.A": "BRK.B",
        "BRK.B": "BRK.B",
        "BK": "BNY",
        "BNY": "BNY",
        "BALL": "BALL",
        "BLL": "BALL",
        "DISCA": "DISCA",
        "DISCK": "DISCA",
        "EQR": "VMRK",
        "VMRK": "VMRK",
        "FOX": "FOXA",
        "FOXA": "FOXA",
        "GOOG": "GOOGL",
        "GOOGL": "GOOGL",
        "NWS": "NWSA",
        "NWSA": "NWSA",
        "PARA": "PSKY",
        "PSKY": "PSKY",
        "VIAC": "PSKY",
        "GEN": "GEN",
        "SYMC": "GEN",
        "MMC": "MRSH",
        "MRSH": "MRSH",
        "UA": "UAA",
        "UAA": "UAA",
        "WLTW": "WTW",
        "WTW": "WTW",
    }

    def __init__(
        self,
        store: Store,
        settings: Settings,
        transport: httpx.BaseTransport | None = None,
    ):
        self.store = store
        self.settings = settings
        self.transport = transport

    def _download_bundle(self, end: date) -> tuple[bytes, dict[str, bytes]]:
        files: dict[str, bytes] = {}
        with httpx.Client(
            timeout=60,
            follow_redirects=True,
            transport=self.transport,
        ) as client:
            for name in self.upstream_files:
                response = client.get(f"{self.raw_base_url}/{name}")
                response.raise_for_status()
                files[name] = response.content
        payload = {
            "coverage_end": end.isoformat(),
            "provider": self.source_name,
            "upstream_commit": self.upstream_commit,
            "files": {name: content.decode("utf-8") for name, content in files.items()},
            "official_overrides": [
                {
                    "effective_date": event.effective_date.isoformat(),
                    "action": event.action,
                    "ticker": event.ticker,
                    "source_url": event.source_url,
                    "announced_at": (
                        event.announced_at.isoformat() if event.announced_at else None
                    ),
                    "reason": event.reason,
                }
                for event in self.official_overrides
                if event.effective_date <= end
            ],
            "canonical_ticker_groups": self.canonical_groups,
        }
        return (
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(),
            files,
        )

    @staticmethod
    def _csv_rows(content: bytes) -> list[dict[str, str]]:
        return list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))

    def _events(self, files: dict[str, bytes], end: date) -> tuple[date, set[str], list]:
        seed_rows = self._csv_rows(files["pitindex/data/sp500_seed.csv"])
        if not seed_rows:
            raise ValueError("pitindex S&P 500 seed is empty")
        seed_date = date.fromisoformat(seed_rows[0]["effective_date"])
        seed = {row["ticker"].strip().upper() for row in seed_rows}
        events = [
            MembershipEvent(
                effective_date=date.fromisoformat(row["date"]),
                action=row["action"],
                ticker=row["ticker"].strip().upper(),
                source_url=(
                    f"https://github.com/arielNacamulli/pitindex/tree/{self.upstream_commit}"
                ),
                reason=row.get("reason") or None,
            )
            for row in self._csv_rows(files["pitindex/data/sp500_changes.csv"])
            if date.fromisoformat(row["date"]) <= end
        ]
        events.extend(event for event in self.official_overrides if event.effective_date <= end)
        events.sort(
            key=lambda event: (
                event.effective_date,
                0 if event.action == "removed" else 1,
                event.ticker,
            )
        )
        return seed_date, seed, events

    def _ticker_company_map(self) -> dict[str, str]:
        output: dict[str, str] = {}
        for table in ("instruments", "fundamentals", "estimates", "prices"):
            frame = self.store.query_df(
                f"SELECT DISTINCT ticker, company_id FROM {table} ORDER BY ticker, company_id"
            )
            for row in frame.itertuples(index=False):
                ticker = str(row.ticker).strip().upper()
                if ticker:
                    output.setdefault(ticker, str(row.company_id))
        for alias, canonical in self.canonical_groups.items():
            company_id = output.get(canonical) or output.get(alias)
            if company_id:
                output[alias] = company_id
                output[canonical] = company_id
        return output

    def _collapse_roster(self, roster: set[str], company_map: dict[str, str]) -> dict[str, str]:
        candidates: dict[str, list[str]] = defaultdict(list)
        for ticker in roster:
            canonical = self.canonical_groups.get(ticker, ticker)
            company_id = company_map.get(ticker) or company_map.get(canonical)
            company_id = company_id or f"PUBLICSP500:{canonical}"
            candidates[company_id].append(ticker)
        output: dict[str, str] = {}
        for company_id, tickers in candidates.items():
            preferred = [self.canonical_groups.get(ticker, ticker) for ticker in tickers]
            available_preferred = sorted(set(preferred).intersection(tickers))
            output[company_id] = (
                available_preferred[0] if available_preferred else sorted(tickers)[0]
            )
        return output

    def _build_intervals(
        self,
        seed_date: date,
        seed: set[str],
        events: list[MembershipEvent],
        company_map: dict[str, str],
    ) -> tuple[list[dict[str, object]], set[str], list[MembershipEvent], int, int]:
        roster = set(seed)
        collapsed = self._collapse_roster(roster, company_map)
        active = {
            company_id: {"ticker": ticker, "member_from": seed_date}
            for company_id, ticker in collapsed.items()
        }
        intervals: list[dict[str, object]] = []
        noops: list[MembershipEvent] = []
        security_counts = [len(roster)]
        company_counts = [len(collapsed)]

        grouped: dict[date, list[MembershipEvent]] = defaultdict(list)
        for event in events:
            grouped[event.effective_date].append(event)
        for effective_date in sorted(grouped):
            for event in grouped[effective_date]:
                if event.action == "removed":
                    if event.ticker not in roster:
                        noops.append(event)
                    roster.discard(event.ticker)
                elif event.action == "added":
                    if event.ticker in roster:
                        noops.append(event)
                    roster.add(event.ticker)
                else:
                    raise ValueError(f"Unsupported membership action: {event.action}")

            next_collapsed = self._collapse_roster(roster, company_map)
            for company_id, state in list(active.items()):
                next_ticker = next_collapsed.get(company_id)
                if next_ticker != state["ticker"]:
                    intervals.append(
                        {
                            "company_id": company_id,
                            "ticker": state["ticker"],
                            "member_from": state["member_from"],
                            "member_to": effective_date - timedelta(days=1),
                        }
                    )
                    del active[company_id]
            for company_id, ticker in next_collapsed.items():
                if company_id not in active:
                    active[company_id] = {
                        "ticker": ticker,
                        "member_from": effective_date,
                    }
            collapsed = next_collapsed
            security_counts.append(len(roster))
            company_counts.append(len(collapsed))

        for company_id, state in active.items():
            intervals.append(
                {
                    "company_id": company_id,
                    "ticker": state["ticker"],
                    "member_from": state["member_from"],
                    "member_to": None,
                }
            )
        return (
            intervals,
            roster,
            noops,
            min(security_counts),
            max(company_counts),
        )

    def sync(
        self,
        *,
        start: date = date(2021, 9, 1),
        end: date = date(2026, 8, 31),
        sync_date: date | None = None,
        refresh_identity_map: bool = False,
    ) -> ImportResult:
        if end < start:
            raise ValueError("end must be on or after start")
        if end > self.supported_through:
            raise ValueError(
                f"Public membership source is verified only through {self.supported_through}"
            )
        bundle, files = self._download_bundle(end)
        digest = hashlib.sha256(bundle).hexdigest()
        existing = self.store.source_by_hash(digest)
        if existing and not refresh_identity_map:
            return ImportResult(
                source_file_id=str(existing["source_file_id"]),
                dataset=DatasetKind.INDEX_MEMBERSHIP,
                sha256=digest,
                archived_path=str(existing["archived_path"]),
                imported_rows=int(existing["row_count"]),
                idempotent=True,
            )

        seed_date, seed, events = self._events(files, end)
        if start < seed_date:
            raise ValueError(f"Requested start {start} precedes source coverage {seed_date}")
        company_map = self._ticker_company_map()
        intervals, final_roster, noops, min_security_count, max_company_count = (
            self._build_intervals(seed_date, seed, events, company_map)
        )
        if noops:
            raise ValueError(f"Membership reconstruction produced {len(noops)} no-op events")
        if min_security_count < 490 or max_company_count > 505:
            raise ValueError("Membership roster count left the accepted S&P 500 sanity band")

        current_rows = self._csv_rows(files["pitindex/data/sp500_current.csv"])
        expected_final = {row["ticker"].strip().upper() for row in current_rows}
        for event in self.official_overrides:
            if event.effective_date > end:
                continue
            if event.action == "removed":
                expected_final.discard(event.ticker)
            else:
                expected_final.add(event.ticker)
        if final_roster != expected_final:
            difference = sorted(final_roster.symmetric_difference(expected_final))
            raise ValueError(
                "Membership end-roster reconciliation failed: " + ", ".join(difference[:20])
            )

        observed_as_of = sync_date or datetime.now(timezone.utc).date()
        source_file_id = str(existing["source_file_id"]) if existing else f"src_{digest[:24]}"
        raw_dir = self.settings.raw_data_dir / "index_membership_public"
        raw_dir.mkdir(parents=True, exist_ok=True)
        archived = raw_dir / (
            f"{digest[:12]}_sp500_membership_{start.isoformat()}_{end.isoformat()}.json"
        )
        archived.write_bytes(bundle)

        ingested_at = datetime.now(timezone.utc).replace(tzinfo=None)
        frame = pd.DataFrame(intervals)
        frame["index_code"] = "SP500"
        frame["effective_at"] = frame["member_from"].map(
            lambda value: datetime.combine(value, time.min)
        )
        frame["as_of_date"] = observed_as_of
        frame["source_file_id"] = source_file_id
        frame["ingested_at"] = ingested_at
        frame = frame[
            [
                "company_id",
                "ticker",
                "index_code",
                "member_from",
                "member_to",
                "effective_at",
                "as_of_date",
                "source_file_id",
                "ingested_at",
            ]
        ].sort_values(["member_from", "company_id"])

        prior_sources = self.store.query_df(
            "SELECT source_file_id, metadata_json FROM source_files "
            "WHERE dataset = 'index_membership'"
        )
        superseded: list[str] = []
        for row in prior_sources.to_dict(orient="records"):
            metadata = row.get("metadata_json")
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}
            if isinstance(metadata, dict) and metadata.get("provider") == self.source_name:
                superseded.append(str(row["source_file_id"]))
        if superseded:
            placeholders = ",".join("?" for _ in superseded)
            self.store.execute(
                f"DELETE FROM index_membership WHERE source_file_id IN ({placeholders})",
                superseded,
            )
        self.store.insert_frame("index_membership", frame)

        file_hashes = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
        unmatched = sorted(
            {str(value) for value in frame["company_id"] if str(value).startswith("PUBLICSP500:")}
        )
        self.store.register_source_file(
            source_file_id=source_file_id,
            dataset=DatasetKind.INDEX_MEMBERSHIP.value,
            original_name=archived.name,
            archived_path=str(archived),
            sha256=digest,
            row_count=len(frame),
            metadata={
                "status": "accepted",
                "provider": self.source_name,
                "source_scope": "public_point_in_time_reconstruction",
                "capital_iq_data": False,
                "upstream_commit": self.upstream_commit,
                "upstream_file_sha256": file_hashes,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "as_of_date": observed_as_of.isoformat(),
                "security_roster_size_end": len(final_roster),
                "company_membership_rows": len(frame),
                "unmatched_public_company_ids": len(unmatched),
                "official_override_urls": sorted(
                    {
                        event.source_url
                        for event in self.official_overrides
                        if event.effective_date <= end
                    }
                ),
                "superseded_public_source_ids": sorted(superseded),
                "identity_map_refreshed": refresh_identity_map,
            },
        )
        issue = DataQualityIssue(
            severity=Severity.WARNING,
            dataset=DatasetKind.INDEX_MEMBERSHIP.value,
            code="PUBLIC_MEMBERSHIP_RECONSTRUCTION",
            message=(
                "Historical S&P 500 membership uses a pinned MIT-licensed community "
                "event log plus source-linked official overrides, not Capital IQ or CRSP."
            ),
            source_file_id=source_file_id,
        )
        self.store.record_issue(issue)
        issues = [issue]
        if unmatched:
            unmatched_issue = DataQualityIssue(
                severity=Severity.WARNING,
                dataset=DatasetKind.INDEX_MEMBERSHIP.value,
                code="PUBLIC_MEMBERSHIP_UNMATCHED_COMPANY_IDS",
                message=(
                    f"{len(unmatched)} historical entities lack a Capital IQ company-ID mapping; "
                    "they retain deterministic PUBLICSP500 identifiers."
                ),
                source_file_id=source_file_id,
            )
            self.store.record_issue(unmatched_issue)
            issues.append(unmatched_issue)
        return ImportResult(
            source_file_id=source_file_id,
            dataset=DatasetKind.INDEX_MEMBERSHIP,
            sha256=digest,
            archived_path=str(archived),
            imported_rows=len(frame),
            issues=issues,
        )
