from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from institutional_quant.config import Settings
from institutional_quant.index_data import MembershipEvent, PublicSP500MembershipSync
from institutional_quant.storage import DuckDBStore


def test_membership_intervals_preserve_ticker_change_and_collapse_share_classes(tmp_path):
    store = DuckDBStore(tmp_path / "membership.duckdb")
    store.initialize()
    store.insert_frame(
        "fundamentals",
        pd.DataFrame(
            [
                {
                    "company_id": "alphabet",
                    "ticker": "GOOGL",
                    "period_end": date(2020, 12, 31),
                    "period_type": "FY",
                    "effective_at": datetime(2021, 2, 1),
                    "as_of_date": date(2021, 2, 1),
                    "metric": "revenue",
                    "value": 1.0,
                    "unit": "USD",
                    "source_file_id": "fixture",
                    "ingested_at": datetime(2021, 2, 1),
                },
                {
                    "company_id": "vivmark",
                    "ticker": "VMRK",
                    "period_end": date(2020, 12, 31),
                    "period_type": "FY",
                    "effective_at": datetime(2021, 2, 1),
                    "as_of_date": date(2021, 2, 1),
                    "metric": "revenue",
                    "value": 1.0,
                    "unit": "USD",
                    "source_file_id": "fixture",
                    "ingested_at": datetime(2021, 2, 1),
                },
            ]
        ),
    )
    sync = PublicSP500MembershipSync(
        store,
        Settings(database_backend="duckdb", database_path=tmp_path / "membership.duckdb"),
    )
    intervals, final_roster, noops, _, _ = sync._build_intervals(
        date(2021, 1, 1),
        {"GOOG", "GOOGL", "EQR"},
        [
            MembershipEvent(date(2021, 2, 1), "removed", "EQR", "test"),
            MembershipEvent(date(2021, 2, 1), "added", "VMRK", "test"),
        ],
        sync._ticker_company_map(),
    )

    assert final_roster == {"GOOG", "GOOGL", "VMRK"}
    assert noops == []
    alphabet = [row for row in intervals if row["company_id"] == "alphabet"]
    assert alphabet == [
        {
            "company_id": "alphabet",
            "ticker": "GOOGL",
            "member_from": date(2021, 1, 1),
            "member_to": None,
        }
    ]
    vivmark = [row for row in intervals if row["company_id"] == "vivmark"]
    assert vivmark[0]["ticker"] == "EQR"
    assert vivmark[0]["member_to"] == date(2021, 1, 31)
    assert vivmark[1]["ticker"] == "VMRK"
    assert vivmark[1]["member_from"] == date(2021, 2, 1)


def test_membership_intervals_preserve_viac_para_company_identity(tmp_path):
    store = DuckDBStore(tmp_path / "para.duckdb")
    store.initialize()
    sync = PublicSP500MembershipSync(
        store,
        Settings(database_backend="duckdb", database_path=tmp_path / "para.duckdb"),
    )

    intervals, _, noops, _, _ = sync._build_intervals(
        date(2019, 12, 4),
        {"VIAC"},
        [
            MembershipEvent(date(2022, 2, 14), "removed", "VIAC", "test"),
            MembershipEvent(date(2022, 2, 14), "added", "PARA", "test"),
        ],
        {},
    )

    assert noops == []
    assert intervals == [
        {
            "company_id": "PUBLICSP500:PSKY",
            "ticker": "VIAC",
            "member_from": date(2019, 12, 4),
            "member_to": date(2022, 2, 13),
        },
        {
            "company_id": "PUBLICSP500:PSKY",
            "ticker": "PARA",
            "member_from": date(2022, 2, 14),
            "member_to": None,
        },
    ]


def test_membership_refresh_rekeys_public_identity_after_ciq_import(tmp_path, monkeypatch):
    store = DuckDBStore(tmp_path / "refresh.duckdb")
    store.initialize()
    settings = Settings(
        database_backend="duckdb",
        database_path=tmp_path / "refresh.duckdb",
        raw_data_dir=tmp_path / "raw",
    )
    sync = PublicSP500MembershipSync(store, settings)
    tickers = {f"T{value:03d}" for value in range(500)}
    current_csv = ("ticker\n" + "\n".join(sorted(tickers)) + "\n").encode()
    files = dict.fromkeys(sync.upstream_files, b"fixture")
    files["pitindex/data/sp500_current.csv"] = current_csv
    monkeypatch.setattr(sync, "_download_bundle", lambda end: (b"fixed-bundle", files))
    monkeypatch.setattr(
        sync,
        "_events",
        lambda files, end: (date(2021, 9, 1), tickers, []),
    )

    first = sync.sync(start=date(2021, 9, 1), end=date(2021, 9, 30))
    assert first.imported_rows == 500
    before = store.query_df("SELECT company_id FROM index_membership WHERE ticker = 'T000'").iloc[
        0
    ]["company_id"]
    assert before == "PUBLICSP500:T000"

    store.insert_frame(
        "instruments",
        pd.DataFrame(
            [
                {
                    "company_id": "CIQ:0",
                    "ticker": "T000",
                    "company_name": "Example",
                    "sector": "Industrials",
                    "currency": "USD",
                    "effective_at": datetime(2026, 9, 1),
                    "as_of_date": date(2026, 9, 1),
                    "source_file_id": "ciq-fixture",
                    "ingested_at": datetime(2026, 9, 1),
                }
            ]
        ),
    )
    refreshed = sync.sync(
        start=date(2021, 9, 1),
        end=date(2021, 9, 30),
        refresh_identity_map=True,
    )

    assert not refreshed.idempotent
    after = store.query_df("SELECT company_id FROM index_membership WHERE ticker = 'T000'").iloc[0][
        "company_id"
    ]
    assert after == "CIQ:0"
    assert store.query_df("SELECT COUNT(*) AS count FROM source_files").iloc[0]["count"] == 1
