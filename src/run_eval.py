import csv
import time

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
    """Load evaluation prompts from prompts.csv."""

    with open(
        PROMPTS_PATH,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        return list(csv.DictReader(f))


def validate_prompts(prompts=None):
    if prompts is None:
        prompts = load_prompts()

    if len(prompts) != 100:
        raise ValueError(
            f"Expected exactly 100 prompts, but found {len(prompts)}"
        )

    required_columns = {"id", "task", "lang", "prompt", "reference"}

    for row in prompts:
        missing = required_columns - set(row.keys())

        if missing:
            raise ValueError(
                f"Missing columns {missing} in row: {row}"
            )

        if not row["id"]:
            raise ValueError(f"Missing id in row: {row}")

        if not row["task"]:
            raise ValueError(f"Missing task in row: {row}")

        if not row["lang"]:
            raise ValueError(f"Missing language in row: {row}")

        if not row["prompt"]:
            raise ValueError(f"Missing prompt in row: {row}")

        if "reference" not in row:
            raise ValueError(f"Missing reference in row: {row}")

    return prompts
    
def main():

    if not SARVAM_API_KEY:
        raise RuntimeError(
            "SARVAM_API_KEY is missing. "
            "Set it in the .env file or "
            "GitHub Actions secret."
        )

    prompts = load_prompts()

    validate_prompts(prompts)

    client = SarvamClient(
        api_key=SARVAM_API_KEY,
        api_url=SARVAM_API_URL,
        model=SARVAM_MODEL,
        timeout=SARVAM_TIMEOUT,
    )

    results = []

    print()
    print("=" * 70)
    print("SARVAM BASELINE EVALUATION")
    print("=" * 70)
    print(f"Model       : {SARVAM_MODEL}")
    print(f"Temperature : {SARVAM_TEMPERATURE}")
    print(f"Prompts     : {len(prompts)}")
    print("=" * 70)
    print()

    for index, row in enumerate(
        prompts,
        start=1
    ):

        print(
            f"[{index:03d}/100] "
            f"ID={row['id']} | "
            f"Task={row['task']} | "
            f"Lang={row['lang']}"
        )

        start_time = time.perf_counter()

        try:

            response = client.chat(
                prompt=row["prompt"],
                temperature=SARVAM_TEMPERATURE,
            )

            latency = (
                time.perf_counter()
                - start_time
            )

            output = response.get(
                "output",
                ""
            )

            usage = response.get(
                "usage",
                {}
            )

            input_tokens = int(
                usage.get(
                    "prompt_tokens",
                    0
                ) or 0
            )

            output_tokens = int(
                usage.get(
                    "completion_tokens",
                    0
                ) or 0
            )

            total_tokens = int(
                usage.get(
                    "total_tokens",
                    input_tokens + output_tokens
                ) or 0
            )

            results.append({
                "id": row["id"],
                "task": row["task"],
                "lang": row["lang"],
                "prompt": row["prompt"],
                "reference": row["reference"],
                "output": output,
                "latency_s": round(
                    latency,
                    4
                ),
                "status": "success",
                "error": "",
                "model": SARVAM_MODEL,
                "temperature": SARVAM_TEMPERATURE,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            })

            print(
                f"       SUCCESS | "
                f"{latency:.2f}s | "
                f"{total_tokens} tokens"
            )

        except Exception as exc:

            latency = (
                time.perf_counter()
                - start_time
            )

            results.append({
                "id": row["id"],
                "task": row["task"],
                "lang": row["lang"],
                "prompt": row["prompt"],
                "reference": row["reference"],
                "output": "",
                "latency_s": round(
                    latency,
                    4
                ),
                "status": "error",
                "error": str(exc),
                "model": SARVAM_MODEL,
                "temperature": SARVAM_TEMPERATURE,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            })

            print(
                f"       ERROR | {exc}"
            )

    fields = [
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
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(results)

    successful = sum(
        row["status"] == "success"
        for row in results
    )

    failed = sum(
        row["status"] == "error"
        for row in results
    )

    total_tokens = sum(
        int(row["total_tokens"])
        for row in results
    )

    print()
    print("=" * 70)
    print("EVALUATION FINISHED")
    print("=" * 70)
    print(f"Total prompts : {len(results)}")
    print(f"Successful    : {successful}")
    print(f"Failed        : {failed}")
    print(f"Total tokens  : {total_tokens:,}")
    print(f"Results file  : {RESULTS_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
