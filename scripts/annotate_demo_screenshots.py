"""Create deterministic English callout banners for the operator screenshots."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "demo"
FONT_CANDIDATES = [
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def font(size: int):
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


steps = [
    ("step-01-capital-iq-screener.png", "ciq-step1.png", "STEP 1 — Select the Capital IQ Companies screener"),
    ("step-02-sp500-criterion.png", "ciq-step1.png", "STEP 2 — Apply the S&P 500 constituent criterion"),
    ("step-03-price-change-columns.png", "ciq-step1.png", "STEP 3 — Add Price Change 1D / 1W / 1M columns"),
    ("step-04-results-as-values.png", "ciq-step1.png", "STEP 4 — Run Results As Values and export CSV/XLSX"),
    ("step-05-data-status-dataset.png", "01-data-status.png", "STEP 5 — Select Current market returns (1D / 1W / 1M)"),
    ("step-06-file-upload.png", "upload-ready.png", "STEP 6 — Choose the unchanged Capital IQ file"),
    ("step-07-timestamps.png", "upload-ready.png", "STEP 7 — Enter availability and observed timestamps"),
    ("step-08-validation-success.png", "upload-result.png", "STEP 8 — Validate and import; review provenance"),
    ("step-09-daily-selection.png", "01-data-status.png", "STEP 9 — Run daily selection and risk monitoring"),
    ("step-10-weekly-adjustment.png", "05-portfolio.png", "STEP 10 — Run weekly adjustment (5% turnover cap)"),
    ("step-11-factor-lab.png", "02-factor-lab.png", "STEP 11 — Inspect the monthly factor snapshot"),
    ("step-12-research-run.png", "03-research-runs.png", "STEP 12 — Start the monthly research run"),
    ("step-13-debate.png", "04-debate.png", "STEP 13 — Read independent analyst debate turns"),
    ("step-14-consensus.png", "04-debate.png", "STEP 14 — Review consensus, dissent and evidence"),
    ("step-15-portfolio.png", "05-portfolio.png", "STEP 15 — Review deterministic target weights"),
    ("step-16-paper-preview.png", "07-paper-trading.png", "STEP 16 — Preview a one-share paper order"),
    ("step-17-paper-approval.png", "07-paper-trading.png", "STEP 17 — Approve and submit the unchanged preview"),
    ("step-18-fill-sync.png", "07-paper-trading.png", "STEP 18 — Synchronize simulated fills"),
    ("step-19-full-cycle.png", "01-data-status.png", "STEP 19 — Run full-cycle; stop at approval checkpoint"),
]


def annotate(output: str, source: str, label: str) -> None:
    image = Image.open(OUT / source).convert("RGB")
    banner_height = 88
    canvas = Image.new("RGB", (image.width, image.height + banner_height), "white")
    canvas.paste(image, (0, banner_height))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, image.width, banner_height), fill="#145fe8")
    draw.text((30, 24), label, fill="white", font=font(28))
    draw.text((30, 59), "English operator guide · source-backed · paper-only execution", fill="#dbe9ff", font=font(15))
    canvas.save(OUT / output, format="PNG", optimize=True)


for output, source, label in steps:
    annotate(output, source, label)
