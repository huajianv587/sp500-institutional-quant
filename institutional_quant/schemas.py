from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


class DatasetKind(str, Enum):
    INSTRUMENTS = "instruments"
    INDEX_MEMBERSHIP = "index_membership"
    FUNDAMENTALS = "fundamentals"
    ESTIMATES = "estimates"
    PRICES = "prices"
    MARKET_RETURNS = "market_returns"
    OWNERSHIP = "ownership"
    INSIDER_TRANSACTIONS = "insider_transactions"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Rating(str, Enum):
    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class DataQualityIssue(BaseModel):
    issue_id: str = Field(default_factory=lambda: str(uuid4()))
    severity: Severity
    dataset: str
    code: str
    message: str
    source_file_id: str | None = None
    row_number: int | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ImportResult(BaseModel):
    source_file_id: str
    dataset: DatasetKind
    sha256: str
    archived_path: str
    imported_rows: int
    rejected_rows: int = 0
    issues: list[DataQualityIssue] = Field(default_factory=list)
    idempotent: bool = False


class EvidenceItem(BaseModel):
    evidence_id: str
    label: str
    value: float | int | str | None
    unit: str | None = None
    effective_at: datetime
    source_file_id: str
    field: str


class EvidencePacket(BaseModel):
    company_id: str
    ticker: str
    sector: str
    as_of_date: date
    factor_score: float
    ml_score: float
    ensemble_score: float
    evidence: list[EvidenceItem]
    provenance_hash: str = ""

    @model_validator(mode="after")
    def add_hash(self) -> EvidencePacket:
        payload = self.model_dump(mode="json", exclude={"provenance_hash"})
        self.provenance_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return self

    @field_validator("evidence")
    @classmethod
    def evidence_must_be_available(cls, values: list[EvidenceItem], info):
        as_of = info.data.get("as_of_date")
        if as_of:
            cutoff = datetime.combine(as_of, datetime.max.time())
            future = [item.evidence_id for item in values if item.effective_at > cutoff]
            if future:
                raise ValueError(f"future evidence is not permitted: {future}")
        return values


class EvidenceClaim(BaseModel):
    text: str
    evidence_refs: list[str] = Field(min_length=1)


class AnalystView(BaseModel):
    role: str
    stance_score: float = Field(ge=-2.0, le=2.0)
    summary: str
    claims: list[EvidenceClaim]
    catalysts: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class DebateTurn(BaseModel):
    speaker: Literal["bull", "bear"]
    round_number: int = Field(ge=1, le=2)
    stance_score: float = Field(ge=-2.0, le=2.0)
    argument: str
    evidence_refs: list[str] = Field(min_length=1)
    rebuttal_to: str | None = None


class ConsensusDecision(BaseModel):
    company_id: str
    ticker: str
    as_of_date: date
    rating: Rating
    score_adjustment: float = Field(ge=-0.10, le=0.10)
    summary: str
    supporting_evidence: list[str] = Field(
        min_length=1,
        description="Exact EvidencePacket evidence_id values only; never prose or inline citations.",
    )
    dissent: str
    uncertainties: list[str]
    analyst_median_score: float = Field(ge=-2.0, le=2.0)
    model_alias: str
    model_version: str | None = None
    system_fingerprint: str | None = None
    prompt_version: str
    evidence_hash: str


class PortfolioPosition(BaseModel):
    company_id: str
    ticker: str
    sector: str
    weight: float = Field(ge=0.0, le=0.05 + 1e-8)
    score: float


class PortfolioRecommendation(BaseModel):
    portfolio_id: str = Field(default_factory=lambda: str(uuid4()))
    as_of_date: date
    cadence: Literal["monthly", "weekly"] = "monthly"
    benchmark: str = "SPY"
    target_volatility: float = 0.12
    expected_volatility: float | None = None
    one_way_turnover: float
    positions: list[PortfolioPosition]
    status: Literal["proposed", "approved", "held"] = "proposed"
    warnings: list[str] = Field(default_factory=list)


class BacktestSpec(BaseModel):
    start_date: date = date(2021, 9, 1)
    end_date: date = date(2026, 8, 31)
    benchmark: str = "SPY"
    transaction_cost_bps: float = Field(default=10.0, ge=0.0)
    min_positions: int = Field(default=20, ge=20, le=30)
    max_positions: int = Field(default=30, ge=20, le=40)
    agent_overlay: bool = False
    agent_variant: Literal["with_debate", "without_debate"] = "with_debate"

    @model_validator(mode="after")
    def validate_dates_and_positions(self) -> BacktestSpec:
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        if self.max_positions < self.min_positions:
            raise ValueError("max_positions must be >= min_positions")
        return self


class StrategyMetrics(BaseModel):
    strategy: str
    cagr: float
    annualized_volatility: float
    sharpe_zero_rf: float
    sortino_zero_rf: float
    max_drawdown: float
    beta: float
    information_ratio: float
    average_one_way_turnover: float
    monthly_hit_rate: float
    observations: int


class BacktestResult(BaseModel):
    backtest_id: str = Field(default_factory=lambda: str(uuid4()))
    spec: BacktestSpec
    certified_point_in_time: bool
    certification_notes: list[str]
    metrics: list[StrategyMetrics]
    monthly_returns: list[dict[str, Any]]
    factor_ic: list[dict[str, Any]]
    statistical_tests: list[dict[str, Any]] = Field(default_factory=list)
    sector_exposures: list[dict[str, Any]] = Field(default_factory=list)
    factor_diagnostics: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ModelConfig(BaseModel):
    model: str
    reasoning_effort: Literal["low", "high", "max"]
    role_scope: Literal["supporting", "decision"] = "decision"


class ModelBenchmarkResult(BaseModel):
    benchmark_id: str = Field(default_factory=lambda: str(uuid4()))
    model: str
    reasoning_effort: str
    cases: int
    schema_success_rate: float
    evidence_coverage: float
    unsupported_claim_rate: float
    rating_stability: float
    numerical_consistency: float
    median_latency_ms: float
    input_tokens: int
    output_tokens: int
    selected: bool = False
    selected_for: Literal["supporting", "decision"] | None = None
    role_scope: Literal["supporting", "decision"] = "decision"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ModelBenchmarkRequest(BaseModel):
    packets: list[EvidencePacket] = Field(min_length=1, max_length=50)
    configurations: list[ModelConfig] = Field(min_length=1, max_length=6)
    repeats: int = Field(default=3, ge=1, le=3)


class ResearchRunRequest(BaseModel):
    as_of_date: date
    company_ids: list[str] = Field(default_factory=list, max_length=10)
    with_debate: bool = True
    historical_months: int = Field(default=0, ge=0, le=24)


class PaperOrderPreview(BaseModel):
    preview_id: str = Field(default_factory=lambda: str(uuid4()))
    symbol: str
    side: Literal["buy", "sell"]
    qty: float = Field(gt=0)
    estimated_price: float = Field(gt=0)
    estimated_notional: float = Field(gt=0)
    paper_endpoint: str = "https://paper-api.alpaca.markets"
    expires_at: datetime | None = None


class PaperTarget(BaseModel):
    symbol: str
    target_weight: float = Field(ge=0.0, le=0.05 + 1e-8)
    quantity: float | None = Field(default=None, gt=0.0, le=1_000_000)


class PaperOrderPreviewRequest(BaseModel):
    targets: list[PaperTarget] = Field(min_length=1)


class PaperOrderSubmitRequest(BaseModel):
    approved: bool
    orders: list[PaperOrderPreview] = Field(min_length=1)


class OperationRequest(BaseModel):
    """Inputs shared by the deterministic daily/weekly/monthly runners."""

    as_of_date: date | None = None
    submit_paper_order: bool = False
    demo_order_quantity: float = Field(default=1.0, gt=0.0, le=1.0)


class OperationResult(BaseModel):
    operation_id: str
    cadence: Literal["daily", "weekly", "monthly", "full-cycle"]
    as_of_date: date
    status: Literal["completed", "held", "awaiting_approval", "failed"]
    message: str
    result: dict[str, Any] = Field(default_factory=dict)


class JobRecord(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid4()))
    kind: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    message: str = "Queued"
    result_ref: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
