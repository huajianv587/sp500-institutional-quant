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
