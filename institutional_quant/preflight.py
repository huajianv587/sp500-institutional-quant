from __future__ import annotations

from dataclasses import dataclass

import httpx

from .config import Settings
from .storage import create_store


@dataclass
class PreflightCheck:
    name: str
    ready: bool
    detail: str


def run_preflight(settings: Settings, live: bool = False) -> list[PreflightCheck]:
    checks = [
        PreflightCheck(
            "Supabase configuration",
            bool(settings.supabase_db_url),
            "configured" if settings.supabase_db_url else "SUPABASE_DB_URL is empty",
        ),
        PreflightCheck(
            "Capital IQ cloud storage gate",
            settings.ciq_cloud_storage_confirmed,
            "confirmed" if settings.ciq_cloud_storage_confirmed else "licence confirmation pending",
        ),
        PreflightCheck(
            "Capital IQ external model gate",
            settings.ciq_external_processing_confirmed,
            "confirmed"
            if settings.ciq_external_processing_confirmed
            else "licence confirmation pending",
        ),
        PreflightCheck(
            "DeepSeek",
            bool(settings.deepseek_api_key),
            "key configured" if settings.deepseek_api_key else "DEEPSEEK_API_KEY is empty",
        ),
        PreflightCheck(
            "Alpaca paper",
            bool(settings.alpaca_paper_key and settings.alpaca_paper_secret),
            "paper keys configured"
            if settings.alpaca_paper_key and settings.alpaca_paper_secret
            else "paper keys are incomplete",
        ),
        PreflightCheck(
            "Alpaca endpoint",
            settings.alpaca_paper_base_url == "https://paper-api.alpaca.markets",
            settings.alpaca_paper_base_url,
        ),
    ]
    if live and settings.supabase_db_url:
        try:
            store = create_store(settings)
            store.initialize()
            store.query_df("SELECT 1 AS ready")
            checks.append(PreflightCheck("Supabase connection", True, "schema and query succeeded"))
        except Exception as exc:
            checks.append(
                PreflightCheck("Supabase connection", False, f"{type(exc).__name__}: {exc}")
            )
    if live and settings.deepseek_api_key:
        try:
            response = httpx.get(
                f"{settings.deepseek_base_url.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {settings.deepseek_api_key}"},
                timeout=20,
            )
            response.raise_for_status()
            checks.append(PreflightCheck("DeepSeek connection", True, "model list succeeded"))
        except Exception as exc:
            checks.append(
                PreflightCheck("DeepSeek connection", False, f"{type(exc).__name__}: {exc}")
            )
    if live and settings.alpaca_paper_key and settings.alpaca_paper_secret:
        try:
            response = httpx.get(
                f"{settings.alpaca_paper_base_url.rstrip('/')}/v2/account",
                headers={
                    "APCA-API-KEY-ID": settings.alpaca_paper_key,
                    "APCA-API-SECRET-KEY": settings.alpaca_paper_secret,
                },
                timeout=20,
            )
            response.raise_for_status()
            account = response.json()
            status = str(account.get("status", "unknown"))
            blocked = bool(account.get("trading_blocked", False))
            checks.append(
                PreflightCheck(
                    "Alpaca paper connection",
                    status.upper() == "ACTIVE" and not blocked,
                    f"account status {status}; trading_blocked={blocked}",
                )
            )
        except Exception as exc:
            checks.append(
                PreflightCheck("Alpaca paper connection", False, f"{type(exc).__name__}: {exc}")
            )
    return checks
