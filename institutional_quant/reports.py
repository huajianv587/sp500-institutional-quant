from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .schemas import BacktestResult


def write_backtest_report(result: BacktestResult, directory: Path) -> tuple[Path, Path]:
    markdown_directory = directory / "reports"
    pdf_directory = directory / "pdf"
    markdown_directory.mkdir(parents=True, exist_ok=True)
    pdf_directory.mkdir(parents=True, exist_ok=True)
    stem = f"backtest-{result.backtest_id}"
    markdown_path = markdown_directory / f"{stem}.md"
    pdf_path = pdf_directory / f"{stem}.pdf"
    synthetic = any("SYNTHETIC DATA ONLY" in note for note in result.certification_notes)
    title = "Synthetic Engineering Case Study" if synthetic else "Institutional Quant Backtest"
    lines = [
        f"# {title}",
        "",
        f"Backtest ID: `{result.backtest_id}`  ",
        f"Window: {result.spec.start_date} to {result.spec.end_date}  ",
        f"Point-in-time certified: **{result.certified_point_in_time}**  ",
        f"Primary transaction cost: {result.spec.transaction_cost_bps:.0f} bps one way",
        "",
        "## Results",
        "",
        "| Strategy | CAGR | Volatility | Sharpe | Max drawdown | Turnover |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric in result.metrics:
        lines.append(
            f"| {metric.strategy} | {metric.cagr:.2%} | {metric.annualized_volatility:.2%} | "
            f"{metric.sharpe_zero_rf:.2f} | {metric.max_drawdown:.2%} | "
            f"{metric.average_one_way_turnover:.2%} |"
        )
    alpha_tests = [row for row in result.statistical_tests if "strategy" in row]
    if alpha_tests:
        lines.extend(
            [
                "",
                "## Statistical qualification",
                "",
                "| Strategy | Annualized alpha | Alpha t-stat | Observations |",
                "|---|---:|---:|---:|",
            ]
        )
        lines.extend(
            f"| {row['strategy']} | {row['annualized_alpha']:.2%} | "
            f"{row['alpha_t_stat']:.2f} | {row['observations']} |"
            for row in alpha_tests
        )
    sensitivity = result.factor_diagnostics.get("cost_sensitivity", {})
    if sensitivity:
        lines.extend(
            [
                "",
                "## Transaction-cost sensitivity",
                "",
                "| One-way cost | Strategy | CAGR | Sharpe |",
                "|---:|---|---:|---:|",
            ]
        )
        for cost, metrics in sensitivity.items():
            lines.extend(
                f"| {cost} bps | {metric['strategy']} | {metric['cagr']:.2%} | "
                f"{metric['sharpe_zero_rf']:.2f} |"
                for metric in metrics
            )
    lines.extend(["", "## Certification notes", ""])
    lines.extend(f"- {note}" for note in result.certification_notes)
    lines.extend(
        [
            "",
            "## Disclosure",
            "",
            "This is a research backtest, not investment advice. Results are sensitive to data availability, "
            "transaction-cost assumptions, model versions and portfolio constraints. Alpaca paper results are simulated.",
            "",
            "## Reproducibility payload",
            "",
            "```json",
            json.dumps(result.spec.model_dump(mode="json"), indent=2),
            "```",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]
    story.append(
        Paragraph(
            f"Window: {result.spec.start_date} to {result.spec.end_date} | "
            f"Point-in-time certified: {result.certified_point_in_time}",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 14))
    rows = [["Strategy", "CAGR", "Volatility", "Sharpe", "Max DD"]]
    rows.extend(
        [
            metric.strategy,
            f"{metric.cagr:.2%}",
            f"{metric.annualized_volatility:.2%}",
            f"{metric.sharpe_zero_rf:.2f}",
            f"{metric.max_drawdown:.2%}",
        ]
        for metric in result.metrics
    )
    table = Table(rows, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16324F")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.extend([table, Spacer(1, 14)])
    if sensitivity:
        sensitivity_rows = [["Cost", "Factor+ML CAGR", "Factor+ML Sharpe"]]
        for cost, metrics in sensitivity.items():
            ensemble = next(
                metric for metric in metrics if metric["strategy"] == "factor_ml_ensemble"
            )
            sensitivity_rows.append(
                [f"{cost} bps", f"{ensemble['cagr']:.2%}", f"{ensemble['sharpe_zero_rf']:.2f}"]
            )
        sensitivity_table = Table(sensitivity_rows, repeatRows=1)
        sensitivity_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.extend(
            [
                Paragraph("Transaction-cost sensitivity", styles["Heading2"]),
                sensitivity_table,
                Spacer(1, 10),
            ]
        )
    story.append(Paragraph("Certification notes", styles["Heading2"]))
    story.extend(Paragraph(note, styles["BodyText"]) for note in result.certification_notes)
    story.extend(
        [
            Spacer(1, 12),
            Paragraph(
                "Research use only. Paper trading is simulated and does not represent live execution.",
                styles["Italic"],
            ),
        ]
    )
    SimpleDocTemplate(str(pdf_path), pagesize=LETTER, title=title).build(story)
    return markdown_path, pdf_path
