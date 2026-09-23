"""Task-specific scoring helpers and score CSV generation."""

import csv
import re

from sacrebleu.metrics import CHRF

from .config import RESULTS_PATH, SCORES_PATH


def normalize(text):
    text = (text or "").strip().lower()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove punctuation while preserving Unicode letters/numbers
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)

    # Normalize whitespace again
    text = re.sub(r"\s+", " ", text).strip()

    return text


def exact_match(hypothesis, reference):
    return int(normalize(hypothesis) == normalize(reference))


def factual_qa_score(output, reference):
    """
    Check whether the reference answer is contained in the model output.

    Factual QA responses can contain explanations, so comparing the entire
    response against the short reference answer is too strict.
    """

    output_norm = normalize(output)
    reference_norm = normalize(reference)

    if not output_norm or not reference_norm:
        return 0

    # Direct answer containment
    if reference_norm in output_norm:
        return 1

    # Handle numeric references such as 96, 366, etc.
    if re.fullmatch(r"-?\d+(?:\.\d+)?", reference_norm):
        numbers = re.findall(r"-?\d+(?:\.\d+)?", output_norm)

        if reference_norm in numbers:
            return 1

    return 0


def extract_final_answer(text):
    text = (text or "").strip()

    patterns = [
        r"(?:final answer|answer)\s*[:\-]\s*([^\n]+)",
        r"####\s*([^\n]+)",
    ]

    for p in patterns:
        m = re.search(p, text, flags=re.I)

        if m:
            return m.group(1).strip().rstrip(".")

    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)

    return numbers[-1] if numbers else text


def reasoning_score(output, reference):
    return exact_match(
        extract_final_answer(output),
        extract_final_answer(reference),
    )


def summarization_score(faithfulness, fluency):
    if faithfulness in ("", None) or fluency in ("", None):
        return ""

    return round(
        (float(faithfulness) + float(fluency)) / 2,
        2,
    )

def extract_translation(output):
    """
    Extract the likely translation from a verbose model response.

    If the model returns a short direct answer, use it directly.
    If the response contains markdown bold text, prefer the bold
    sentence because models often put the actual translation there.
    """

    text = (output or "").strip()

    if not text:
        return ""

    # Look for markdown bold content.
    bold_matches = re.findall(r"\*\*(.*?)\*\*", text, flags=re.DOTALL)

    if bold_matches:
        # Prefer a reasonably sentence-like bold answer.
        for candidate in bold_matches:
            candidate = candidate.strip()

            if len(candidate) > 1:
                return candidate

    # Look for blockquote-style answers.
    quote_matches = re.findall(
        r"(?m)^>\s*(.+)$",
        text,
    )

    if quote_matches:
        return quote_matches[0].strip()

    # Look for a line after common translation labels.
    patterns = [
        r"(?is)(?:translation|translated sentence|direct translation)"
        r"\s*[:\-]\s*\n?\s*(?:\*\*)?(.+?)(?:\*\*)?(?:\n|$)",
        r"(?is)(?:most direct translation|primary translation)"
        r"\s*[:\-]?\s*\n?\s*(?:\*\*)?(.+?)(?:\*\*)?(?:\n|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            candidate = match.group(1).strip()

            if candidate:
                return candidate

    # Otherwise use the complete output.
    return text


def score_translation(output, reference):
    return round(
        CHRF().sentence_score(
            output or "",
            [reference or ""],
        ).score,
        4,
    )

def validate_manual_summary_scores(path):
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    result = {}

    for r in rows:
        if r["faithfulness_score"] and r["fluency_score"]:
            result[r["id"]] = summarization_score(
                r["faithfulness_score"],
                r["fluency_score"],
            )

    return result

def build_scores(manual_path=None):
    manual = (
        validate_manual_summary_scores(manual_path)
        if manual_path
        else {}
    )

    with open(RESULTS_PATH, encoding="utf-8-sig", newline="") as f:
        results = list(csv.DictReader(f))

    rows = []

    for r in results:
        task = r["task"]

        if task == "factual_qa":
            score = factual_qa_score(
                r["output"],
                r["reference"],
            )
            metric = "correctness"

        elif task == "translation":
            score = score_translation(
                r["output"],
                r["reference"],
            )
            metric = "chrf"

        elif task == "reasoning_math":
            score = reasoning_score(
                r["output"],
                r["reference"],
            )
            metric = "final_answer_match"

        elif task == "summarization":
            score = manual.get(r["id"], "")
            metric = "human_rating"

        else:
            score = ""
            metric = "unknown"

        rows.append(
            {
                "id": r["id"],
                "task": task,
                "lang": r["lang"],
                "score": score,
                "metric": metric,
                "reference": r["reference"],
                "output": r["output"],
                "notes": "",
            }
        )

    with open(
        SCORES_PATH,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "task",
                "lang",
                "score",
                "metric",
                "reference",
                "output",
                "notes",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {SCORES_PATH}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manual-summary",
        help="CSV with id,faithfulness_score,fluency_score,review_notes",
    )

    args = parser.parse_args()

    build_scores(args.manual_summary)