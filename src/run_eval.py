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


def validate_prompts(prompts):
    """
    Validate the evaluation prompt dataset.

    Requirements:
    - Exactly 100 prompts
    - IDs 1 through 100
    - Required columns present
    - 25 prompts per task
    """

    if len(prompts) != 100:
        raise ValueError(
            f"Expected exactly 100 prompts, "
            f"but found {len(prompts)}."
        )

    required_columns = {
        "id",
        "task",
        "lang",
        "prompt",
        "reference",
    }

    for row in prompts:

        missing = (
            required_columns
            - set(row.keys())
        )

        if missing:
            raise ValueError(
                f"Prompt ID {row.get('id')} "
                f"is missing columns: {missing}"
            )

        if not row["prompt"].strip():
            raise ValueError(
                f"Prompt ID {row['id']} "
                f"has an empty prompt."
            )

    ids = [
        int(row["id"])
        for row in prompts
    ]

    if ids != list(range(1, 101)):
        raise ValueError(
            "Prompt IDs must be exactly "
            "1 through 100."
        )

    expected_tasks = {
        "factual_qa": 25,
        "translation": 25,
        "summarization": 25,
        "reasoning_math": 25,
    }

    task_counts = {}

    for row in prompts:

        task = row["task"]

        task_counts[task] = (
            task_counts.get(task, 0) + 1
        )

    if task_counts != expected_tasks:
        raise ValueError(
            "Unexpected task distribution. "
            f"Expected {expected_tasks}, "
            f"found {task_counts}."
        )

    return True


def main():

    # ---------------------------------------------------------
    # 1. Validate API key
    # ---------------------------------------------------------

    if not SARVAM_API_KEY:

        raise RuntimeError(
            "SARVAM_API_KEY is missing. "
            "Set it in the .env file or "
            "GitHub Actions secret."
        )

    # ---------------------------------------------------------
    # 2. Load prompts
    # ---------------------------------------------------------

    prompts = load_prompts()

    # ---------------------------------------------------------
    # 3. Validate prompts
    # ---------------------------------------------------------

    validate_prompts(prompts)

    # ---------------------------------------------------------
    # 4. Create Sarvam client
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 5. Run all 100 prompts
    # ---------------------------------------------------------

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

            # -------------------------------------------------
            # Token usage
            # -------------------------------------------------

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
                    input_tokens
                    + output_tokens
                ) or 0
            )

            # -------------------------------------------------
            # Successful result
            # -------------------------------------------------

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

                "temperature":
                    SARVAM_TEMPERATURE,

                "input_tokens":
                    input_tokens,

                "output_tokens":
                    output_tokens,

                "total_tokens":
                    total_tokens,
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

            # -------------------------------------------------
            # Failed result
            # -------------------------------------------------

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

                "temperature":
                    SARVAM_TEMPERATURE,

                "input_tokens": 0,

                "output_tokens": 0,

                "total_tokens": 0,
            })

            print(
                f"       ERROR | {exc}"
            )

    # ---------------------------------------------------------
    # 6. Write results.csv
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 7. Evaluation summary
    # ---------------------------------------------------------

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
    print(
        f"Total prompts : {len(results)}"
    )
    print(
        f"Successful    : {successful}"
    )
    print(
        f"Failed        : {failed}"
    )
    print(
        f"Total tokens  : {total_tokens:,}"
    )
    print(
        f"Results file  : {RESULTS_PATH}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
