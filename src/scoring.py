import csv
import re
from collections import defaultdict

from sacrebleu.metrics import CHRF

from src.config import (
    RESULTS_PATH,
    SCORES_PATH,
    INPUT_COST_PER_1M,
    OUTPUT_COST_PER_1M,
)


def normalize(text):
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(
        r"[^\w\s]",
        "",
        text,
        flags=re.UNICODE,
    )
    return re.sub(r"\s+", " ", text).strip()


def exact_match(output, reference):
    return int(
        normalize(output) == normalize(reference)
    )


def factual_score(output, reference):
    output_norm = normalize(output)
    reference_norm = normalize(reference)

    if not output_norm or not reference_norm:
        return 0

    if reference_norm in output_norm:
        return 1

    return 0


def extract_number(text):
    matches = re.findall(
        r"-?\d+(?:\.\d+)?",
        text or "",
    )

    if not matches:
        return ""

    return matches[-1]


def reasoning_score(output, reference):
    output_number = extract_number(output)
    reference_number = extract_number(reference)

    if not output_number:
        return 0

    if output_number == reference_number:
        return 1

    return 0


def translation_score(output, reference):
    if not output.strip():
        return 0

    return round(
        CHRF().sentence_score(
            output,
            [reference],
        ).score,
        4,
    )


def summarization_score(output, reference):
    """
    Summarization requires human evaluation under the assignment.
    Therefore no fabricated human score is generated automatically.
    """
    return ""


def main():
    with open(
        RESULTS_PATH,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    scored = []

    for row in rows:
        task = row["task"]

        if row["status"] != "success":
            score = 0
            score_type = "failed_api_call"

        elif task == "factual_qa":
            score = factual_score(
                row["output"],
                row["reference"],
            )
            score_type = "exact_match"

        elif task == "translation":
            score = translation_score(
                row["output"],
                row["reference"],
            )
            score_type = "chrF"

        elif task == "summarization":
            score = summarization_score(
                row["output"],
                row["reference"],
            )
            score_type = "human_rating_pending"

        elif task == "reasoning_math":
            score = reasoning_score(
                row["output"],
                row["reference"],
            )
            score_type = "final_answer_match"

        else:
            score = ""
            score_type = "unknown"

        input_tokens = int(
            row.get("input_tokens", 0) or 0
        )

        output_tokens = int(
            row.get("output_tokens", 0) or 0
        )

        total_tokens = int(
            row.get("total_tokens", 0) or 0
        )

        cost_in = (
            input_tokens / 1_000_000
        ) * INPUT_COST_PER_1M

        cost_out = (
            output_tokens / 1_000_000
        ) * OUTPUT_COST_PER_1M

        total_cost = cost_in + cost_out

        scored.append({
            "id": row["id"],
            "task": task,
            "lang": row["lang"],
            "score": score,
            "score_type": score_type,
            "latency_s": row["latency_s"],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_inr": round(
                total_cost,
                6,
            ),
            "status": row["status"],
            "error": row["error"],
        })

    with open(
        SCORES_PATH,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        fields = [
            "id",
            "task",
            "lang",
            "score",
            "score_type",
            "latency_s",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "estimated_cost_inr",
            "status",
            "error",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(scored)

    print(f"Scores written to {SCORES_PATH}")


if __name__ == "__main__":
    main()
