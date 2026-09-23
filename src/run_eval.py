"""Run the 100-prompt evaluation."""

import argparse
import csv
import sys

from .config import (
    PROMPTS_PATH,
    RESULTS_PATH,
    SARVAM_MODEL,
    SARVAM_TEMPERATURE,
    LOG_PATH,
)
from .sarvam_client import SarvamClient, SarvamAPIError
from .utils import setup_logging


REQUIRED_COLUMNS = {
    "id",
    "task",
    "lang",
    "prompt",
    "reference",
}


EXPECTED_TASK_COUNTS = {
    "factual_qa": 25,
    "translation": 25,
    "summarization": 25,
    "reasoning_math": 25,
}


def validate_prompts():
    """Validate prompts.csv before making any API calls."""

    if not PROMPTS_PATH.exists():
        raise ValueError(
            f"prompts.csv was not found at: {PROMPTS_PATH}"
        )

    with open(
        PROMPTS_PATH,
        encoding="utf-8",
        newline=""
    ) as f:

        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("prompts.csv is empty.")

    missing_columns = REQUIRED_COLUMNS - set(rows[0].keys())

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if len(rows) != 100:
        raise ValueError(
            f"Expected exactly 100 prompts, found {len(rows)}"
        )

    ids = [row["id"] for row in rows]

    if len(set(ids)) != 100:
        raise ValueError(
            "Prompt IDs are not unique."
        )

    task_counts = {}

    for row in rows:
        task = row["task"]
        task_counts[task] = task_counts.get(task, 0) + 1

    if task_counts != EXPECTED_TASK_COUNTS:
        raise ValueError(
            "Unexpected task distribution.\n"
            f"Found: {task_counts}\n"
            f"Expected: {EXPECTED_TASK_COUNTS}"
        )

    return rows


def test_connection():
    """Send one small request to verify API connectivity."""

    print()
    print("=" * 60)
    print("SARVAM API CONNECTION TEST")
    print("=" * 60)

    client = SarvamClient()

    result = client.chat(
        "Reply with exactly: CONNECTION_OK"
    )

    if result["status"] != "success":

        print()
        print("❌ Connection test FAILED")
        print()
        print("Error:")
        print(result["error"])

        return 1

    print()
    print("✅ Connection test SUCCEEDED")
    print()
    print("Model:")
    print(result["model"])

    print()
    print("Response:")
    print(result["output"])

    print()
    print("Latency:")
    print(f"{result['latency_s']} seconds")

    print()
    print("=" * 60)

    return 0


def run_evaluation():
    """Run all 100 prompts against the Sarvam API."""

    setup_logging(LOG_PATH)

    print()
    print("=" * 60)
    print("SARVAM-M BASELINE EVALUATION")
    print("=" * 60)

    print()
    print("Validating prompts.csv...")

    rows = validate_prompts()
    client = SarvamClient()

    print("✅ Dataset validation successful")
    print()
    print("Total prompts:", len(rows))
    print("Model:", SARVAM_MODEL)
    print("Temperature:", SARVAM_TEMPERATURE)

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "id",
        "task",
        "lang",
        "prompt",
        "reference",
        "output",
        "latency_s",
        "status",
        "error",
        "model",
        "temperature",
        "input_tokens",
        "output_tokens",
        "total_tokens",
    ]

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8",
        newline=""
    ) as output_file:

        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for index, row in enumerate(rows, start=1):

            print()
            print(
                f"[{index}/100] "
                f"ID={row['id']} | "
                f"Task={row['task']} | "
                f"Language={row['lang']}"
            )

            result = client.chat(row["prompt"])

            writer.writerow(
                {
                    "id": row["id"],
                    "task": row["task"],
                    "lang": row["lang"],
                    "prompt": row["prompt"],
                    "reference": row["reference"],
                    "output": result["output"],
                    "latency_s": result["latency_s"],
                    "status": result["status"],
                    "error": result["error"],
                    "model": result["model"],
                    "temperature": SARVAM_TEMPERATURE,
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "total_tokens": result["total_tokens"],
                }
            )

    print()
    print("=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)

    print()
    print("Results saved to:")
    print(RESULTS_PATH)

    return 0


def main():

    parser = argparse.ArgumentParser(
        description="Sarvam-M baseline evaluation"
    )

    parser.add_argument(
        "--test-connection",
        action="store_true",
        help="Test the Sarvam API with one request",
    )

    args = parser.parse_args()

    try:

        if args.test_connection:
            return test_connection()

        return run_evaluation()

    except SarvamAPIError as error:

        print()
        print("❌ Sarvam API error:")
        print(error)

        return 1

    except ValueError as error:

        print()
        print("❌ Validation error:")
        print(error)

        return 1


if __name__ == "__main__":
    sys.exit(main())