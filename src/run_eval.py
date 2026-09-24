import csv
import time
from pathlib import Path

from src.config import (
    PROMPTS_PATH,
    RESULTS_PATH,
    SARVAM_API_KEY,
    SARVAM_API_URL,
    SARVAM_MODEL,
    SARVAM_TEMPERATURE,
    SARVAM_TIMEOUT,
)
from src.sarvam_client import SarvamClient


def load_prompts():
    with open(
        PROMPTS_PATH,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        return list(csv.DictReader(f))


def main():
    if not SARVAM_API_KEY:
        raise RuntimeError(
            "SARVAM_API_KEY is missing. "
            "Set it as a GitHub Actions secret or in .env."
        )

    rows = load_prompts()

    if len(rows) != 100:
        raise ValueError(
            f"Expected 100 prompts, found {len(rows)}"
        )

    ids = [int(row["id"]) for row in rows]

    if ids != list(range(1, 101)):
        raise ValueError(
            "prompts.csv must contain IDs 1 through 100."
        )

    client = SarvamClient(
        api_key=SARVAM_API_KEY,
        api_url=SARVAM_API_URL,
        model=SARVAM_MODEL,
        timeout=SARVAM_TIMEOUT,
    )

    output_rows = []

    for index, row in enumerate(rows, start=1):
        print(
            f"[{index}/100] "
            f"Running ID {row['id']} "
            f"({row['task']}, {row['lang']})"
        )

        start = time.perf_counter()

        try:
            response = client.chat(
                prompt=row["prompt"],
                temperature=SARVAM_TEMPERATURE,
            )

            latency = time.perf_counter() - start

            output = response.get("output", "")
            usage = response.get("usage", {})

            input_tokens = int(
                usage.get("prompt_tokens", 0)
            )

            output_tokens = int(
                usage.get("completion_tokens", 0)
            )

            total_tokens = int(
                usage.get(
                    "total_tokens",
                    input_tokens + output_tokens
                )
            )

            output_rows.append({
                "id": row["id"],
                "task": row["task"],
                "lang": row["lang"],
                "prompt": row["prompt"],
                "reference": row["reference"],
                "output": output,
                "latency_s": round(latency, 4),
                "status": "success",
                "error": "",
                "model": SARVAM_MODEL,
                "temperature": SARVAM_TEMPERATURE,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            })

        except Exception as exc:
            latency = time.perf_counter() - start

            output_rows.append({
                "id": row["id"],
                "task": row["task"],
                "lang": row["lang"],
                "prompt": row["prompt"],
                "reference": row["reference"],
                "output": "",
                "latency_s": round(latency, 4),
                "status": "error",
                "error": str(exc),
                "model": SARVAM_MODEL,
                "temperature": SARVAM_TEMPERATURE,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            })

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
        encoding="utf-8-sig",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(output_rows)

    success_count = sum(
        r["status"] == "success"
        for r in output_rows
    )

    print()
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Prompts:  {len(output_rows)}")
    print(f"Success:  {success_count}")
    print(f"Errors:   {len(output_rows) - success_count}")
    print(f"Results:  {RESULTS_PATH}")


if __name__ == "__main__":
    main()
