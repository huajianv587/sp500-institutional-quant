from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Settings(BaseModel):
    database_backend: Literal["supabase", "duckdb"] = "supabase"
    supabase_db_url: str | None = None
    database_path: Path = Field(default=Path("data/institutional_quant.duckdb"))
    price_lake_path: Path = Field(default=Path("data/market/prices.parquet"))
    raw_data_dir: Path = Field(default=Path("data/raw"))
    report_dir: Path = Field(default=Path("output"))
    host: str = "127.0.0.1"
    port: int = 8000
    enable_scheduler: bool = False

    ciq_external_processing_confirmed: bool = False
    ciq_cloud_storage_confirmed: bool = False
    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_analyst_model: str = "deepseek-v4-pro"
    deepseek_decision_model: str = "deepseek-v4-pro"

    alpaca_paper_key: str | None = None
    alpaca_paper_secret: str | None = None
    alpaca_paper_base_url: str = "https://paper-api.alpaca.markets"
    alpaca_data_base_url: str = "https://data.alpaca.markets"

    @field_validator("alpaca_paper_base_url")
    @classmethod
    def paper_only(cls, value: str) -> str:
        normalized = value.rstrip("/")
        if normalized != "https://paper-api.alpaca.markets":
            raise ValueError("Only the Alpaca paper endpoint is permitted")
        return normalized

    @classmethod
    def from_env(cls) -> Settings:
        def as_bool(name: str, default: bool = False) -> bool:
            value = os.getenv(name)
            if value is None:
                return default
            return value.strip().lower() in {"1", "true", "yes", "on"}

        return cls(
            database_backend=os.getenv("IQ_DATABASE_BACKEND", "supabase").lower(),
            supabase_db_url=os.getenv("SUPABASE_DB_URL") or None,
            database_path=Path(os.getenv("IQ_DATABASE_PATH", "data/institutional_quant.duckdb")),
            price_lake_path=Path(os.getenv("IQ_PRICE_LAKE_PATH", "data/market/prices.parquet")),
            raw_data_dir=Path(os.getenv("IQ_RAW_DATA_DIR", "data/raw")),
            report_dir=Path(os.getenv("IQ_REPORT_DIR", "output")),
            host=os.getenv("IQ_HOST", "127.0.0.1"),
            port=int(os.getenv("IQ_PORT", "8000")),
            enable_scheduler=as_bool("IQ_ENABLE_SCHEDULER"),
            ciq_external_processing_confirmed=as_bool("CIQ_EXTERNAL_PROCESSING_CONFIRMED"),
            ciq_cloud_storage_confirmed=as_bool("CIQ_CLOUD_STORAGE_CONFIRMED"),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY") or None,
            deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            deepseek_analyst_model=os.getenv("DEEPSEEK_ANALYST_MODEL", "deepseek-v4-pro"),
            deepseek_decision_model=os.getenv("DEEPSEEK_DECISION_MODEL", "deepseek-v4-pro"),
            alpaca_paper_key=os.getenv("ALPACA_PAPER_KEY") or None,
            alpaca_paper_secret=os.getenv("ALPACA_PAPER_SECRET") or None,
            alpaca_paper_base_url=os.getenv(
                "ALPACA_PAPER_BASE_URL", "https://paper-api.alpaca.markets"
            ),
            alpaca_data_base_url=os.getenv("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets"),
        )

    def ensure_directories(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.price_lake_path.parent.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def require_supabase(self) -> str:
        if self.database_backend != "supabase":
            raise RuntimeError("Production database backend must be Supabase")
        if not self.supabase_db_url:
            raise RuntimeError("SUPABASE_DB_URL is required for the Supabase backend")
        return self.supabase_db_url
