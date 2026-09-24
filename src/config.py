import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

load_dotenv(ROOT / ".env")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "").strip()

SARVAM_MODEL = os.getenv(
    "SARVAM_MODEL",
    "sarvam-105b"
).strip()

SARVAM_API_URL = os.getenv(
    "SARVAM_API_URL",
    "https://api.sarvam.ai/v1/chat/completions"
).strip()

SARVAM_TIMEOUT = int(
    os.getenv("SARVAM_TIMEOUT", "120")
)

SARVAM_TEMPERATURE = float(
    os.getenv("SARVAM_TEMPERATURE", "0")
)

# Sarvam 105B pricing as of the current Sarvam pricing page.
# ₹ per 1M tokens.
INPUT_COST_PER_1M = float(
    os.getenv("INPUT_COST_PER_1M", "29.28")
)

OUTPUT_COST_PER_1M = float(
    os.getenv("OUTPUT_COST_PER_1M", "73.20")
)

DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

PROMPTS_PATH = DATA_DIR / "prompts.csv"
RESULTS_PATH = DATA_DIR / "results.csv"
SCORES_PATH = DATA_DIR / "scores.csv"

REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

REPORT_PATH = REPORTS_DIR / "evaluation_report.md"
