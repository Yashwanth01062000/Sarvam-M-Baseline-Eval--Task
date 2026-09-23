"""Configuration for the Sarvam-M baseline evaluation."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "").strip()
SARVAM_MODEL = os.getenv("SARVAM_MODEL", "sarvam-105b").strip()
SARVAM_API_URL = os.getenv(
    "SARVAM_API_URL",
    "https://api.sarvam.ai/v1/chat/completions",
).strip()
SARVAM_TIMEOUT = int(os.getenv("SARVAM_TIMEOUT", "60"))
SARVAM_TEMPERATURE = float(os.getenv("SARVAM_TEMPERATURE", "0"))

DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "logs"
REPORT_DIR = ROOT / "reports"
PROMPTS_PATH = DATA_DIR / "prompts.csv"
RESULTS_PATH = DATA_DIR / "results.csv"
SCORES_PATH = DATA_DIR / "scores.csv"
SUMMARY_PATH = DATA_DIR / "summary.csv"
FAILURES_PATH = DATA_DIR / "failures.csv"
LOG_PATH = LOG_DIR / "evaluation.log"
