from __future__ import annotations

import csv
import re
import statistics
from pathlib import Path

from sacrebleu.metrics import CHRF


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

RESULTS_PATH = DATA_DIR / "results.csv"
SCORES_PATH = DATA_DIR / "scores.csv"
AVERAGES_PATH = DATA_DIR / "scores_averages.csv"


# ---------------------------------------------------------
# METRIC
# ---------------------------------------------------------

chrf = CHRF()


# ---------------------------------------------------------
# GENERAL HELPERS
# ---------------------------------------------------------

def normalize(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def load_csv(path: Path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------
# FACTUAL QA
# ---------------------------------------------------------

def factual_qa_score(output: str, reference: str) -> int:
    output_norm = normalize(output)
    reference_norm = normalize(reference)

    if not output_norm or not reference_norm:
        return 0

    if reference_norm in output_norm:
        return 1

    if re.fullmatch(r"-?\d+(?:\.\d+)?", reference_norm):
        numbers = re.findall(r"-?\d+(?:\.\d+)?", output_norm)
        if reference_norm in numbers:
            return 1

    return 0


# ---------------------------------------------------------
# REASONING / MATH
# ---------------------------------------------------------

def extract_final_answer(text: str) -> str:
    text = (text or "").strip()

    patterns = [
        r"(?:final answer|answer)\s*[:\-]\s*([^\n]+)",
        r"####\s*([^\n]+)",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            return match.group(1).strip().rstrip(".")

    numbers = re.findall(
        r"-?\d+(?:\.\d+)?",
        text,
    )

    if numbers:
        return numbers[-1]

    return text


def reasoning_score(output: str, reference: str) -> int:
    output_answer = normalize(
        extract_final_answer(output)
    )

    reference_answer = normalize(
        extract_final_answer(reference)
    )

    return int(output_answer == reference_answer)


# ---------------------------------------------------------
# TRANSLATION
# ---------------------------------------------------------

def detect_script(text: str) -> str:
    """
    Detect the main script used by the reference/output.

    Returns:
        latin
        devanagari
        tamil
        other
    """

    if not text:
        return "other"

    devanagari_count = len(
        re.findall(r"[\u0900-\u097F]", text)
    )

    tamil_count = len(
        re.findall(r"[\u0B80-\u0BFF]", text)
    )

    latin_count = len(
        re.findall(r"[A-Za-z]", text)
    )

    counts = {
        "devanagari": devanagari_count,
        "tamil": tamil_count,
        "latin": latin_count,
    }

    script = max(
        counts,
        key=counts.get,
    )

    if counts[script] == 0:
        return "other"

    return script


def clean_translation_line(line: str) -> str:
    """
    Remove markdown/list formatting from a candidate
    translation line.
    """

    line = line.strip()

    # Markdown bullets
    line = re.sub(
        r"^\s*(?:[-*+>])\s*",
        "",
        line,
    )

    # Numbered lists
    line = re.sub(
        r"^\s*\d+[.)]\s*",
        "",
        line,
    )

    # Markdown headings
    line = re.sub(
        r"^\s*#+\s*",
        "",
        line,
    )

    # Markdown emphasis
    line = line.replace("**", "")
    line = line.replace("__", "")
    line = line.replace("`", "")

    # Quotes
    line = line.strip(
        '"“”\''
    )

    return line.strip()


def get_translation_candidates(output: str):
    """
    Extract sentence-like lines from a verbose model response.
    """

    candidates = []

    for raw_line in re.split(
        r"[\r\n]+",
        output or "",
    ):
        line = clean_translation_line(raw_line)

        if not line:
            continue

        if len(line) > 250:
            continue

        lower = line.lower().strip().rstrip(":")

        # Skip obvious explanatory headings.
        ignored_headings = {
            "translation",
            "primary translation",
            "most direct translation",
            "most direct and common translation",
            "most direct and common way to translate this is",
            "here are a few ways to translate that sentence into english",
            "other options",
            "other alternatives",
            "breakdown",
            "quick breakdown",
            "summary",
        }

        if lower in ignored_headings:
            continue

        candidates.append(line)

    return candidates


def script_compatible(candidate: str, target_script: str) -> bool:
    """
    Reject wrong-language/script outputs.
    """

    candidate_script = detect_script(candidate)

    return candidate_script == target_script


def extract_translation(
    output: str,
    reference: str,
) -> tuple[str, str]:
    """
    Find the most likely actual translation.

    The reference determines the expected target script.

    Example:
        Reference = Hindi
        Output contains a long explanation + Hindi translation

    The Hindi sentence is extracted and only that sentence
    is passed to chrF.
    """

    if not output or not output.strip():
        return "", "empty_output"

    target_script = detect_script(reference)

    candidates = get_translation_candidates(output)

    compatible = [
        candidate
        for candidate in candidates
        if script_compatible(
            candidate,
            target_script,
        )
    ]

    if not compatible:
        return "", "wrong_language_or_no_translation"

    # Choose the candidate with the highest chrF
    # against the reference.
    scored_candidates = []

    for candidate in compatible:
        score = chrf.sentence_score(
            candidate,
            [reference],
        ).score

        scored_candidates.append(
            (
                score,
                candidate,
            )
        )

    scored_candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    best_score, best_candidate = scored_candidates[0]

    return best_candidate, "extracted_translation"


def translation_score(
    output: str,
    reference: str,
):
    """
    Score ONLY the extracted translation.

    Empty/wrong-language outputs receive 0.
    """

    extracted, status = extract_translation(
        output,
        reference,
    )

    if not extracted:
        return 0.0, status, ""

    score = chrf.sentence_score(
        extracted,
        [reference],
    ).score

    return (
        round(score, 4),
        status,
        extracted,
    )


# ---------------------------------------------------------
# BUILD SCORES
# ---------------------------------------------------------

def build_scores():

    results = load_csv(
        RESULTS_PATH
    )

    output_rows = []

    for row in results:

        row_id = int(row["id"])
        task = row["task"].strip()
        lang = row["lang"].strip()

        score = ""
        metric = ""
        reference = row.get(
            "reference",
            "",
        )
        output = row.get(
            "output",
            "",
        )
        notes = ""

        # ---------------------------------------------
        # FACTUAL QA
        # ---------------------------------------------

        if task == "factual_qa":

            score = factual_qa_score(
                output,
                reference,
            )

            metric = "correctness"

            if score == 1:
                notes = "Correct"
            else:
                notes = (
                    "Reference-based mismatch "
                    "or incorrect answer."
                )

        # ---------------------------------------------
        # TRANSLATION
        # ---------------------------------------------

        elif task == "translation":

            score, extraction_status, extracted = (
                translation_score(
                    output,
                    reference,
                )
            )

            metric = "chrF_extracted"

            if extraction_status == "empty_output":

                notes = (
                    "Empty model output; "
                    "translation score = 0."
                )

            elif (
                extraction_status
                == "wrong_language_or_no_translation"
            ):

                notes = (
                    "No valid translation found "
                    "in the expected target script; "
                    "score = 0."
                )

            else:

                notes = (
                    f"Extracted translation: "
                    f"{extracted}"
                )

        # ---------------------------------------------
        # REASONING / MATH
        # ---------------------------------------------

        elif task == "reasoning_math":

            score = reasoning_score(
                output,
                reference,
            )

            metric = "final_answer_match"

            if score == 1:
                notes = "Correct final answer."
            else:
                notes = "Final answer mismatch."

        # ---------------------------------------------
        # SUMMARIZATION
        # ---------------------------------------------

        elif task == "summarization":

            metric = "human_rating"

            # These are the updated manual ratings.
            manual_scores = {
                51: (5, 5),
                52: (5, 5),
                53: (5, 5),
                54: (5, 5),
                55: (5, 5),
                56: (5, 3),
                57: (5, 3),
                58: (4, 4),
                59: (4, 4),
                60: (5, 5),
                61: (5, 5),
                62: (5, 5),
                63: None,
                64: (2, 3),
                65: (5, 5),
                66: (5, 5),
                67: None,
                68: (5, 5),
                69: (2, 2),
                70: (4, 4),
                71: (5, 5),
                72: (4, 4),
                73: (5, 5),
                74: None,
                75: None,
            }

            rating = manual_scores.get(
                row_id
            )

            if rating is not None:

                faithfulness, fluency = rating

                score = round(
                    (
                        faithfulness
                        + fluency
                    )
                    * 10,
                    2,
                )

                notes = (
                    f"Faithfulness "
                    f"{faithfulness}/5; "
                    f"Fluency "
                    f"{fluency}/5."
                )

            else:

                score = ""

                notes = (
                    "No model output; "
                    "not included in "
                    "summarization average."
                )

        output_rows.append(
            {
                "id": row_id,
                "task": task,
                "lang": lang,
                "score": score,
                "metric": metric,
                "reference": reference,
                "output": output,
                "notes": notes,
            }
        )

    # -----------------------------------------------------
    # WRITE scores.csv
    # -----------------------------------------------------

    fieldnames = [
        "id",
        "task",
        "lang",
        "score",
        "metric",
        "reference",
        "output",
        "notes",
    ]

    with open(
        SCORES_PATH,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            output_rows
        )

    # -----------------------------------------------------
    # WRITE AVERAGES
    # -----------------------------------------------------

    groups = {}

    for row in output_rows:

        if row["score"] == "":
            continue

        key = (
            row["task"],
            row["lang"],
        )

        groups.setdefault(
            key,
            [],
        ).append(
            float(row["score"])
        )

    with open(
        AVERAGES_PATH,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "task",
                "lang",
                "count",
                "average_score",
            ]
        )

        for (
            task,
            lang,
        ), values in sorted(
            groups.items()
        ):

            writer.writerow(
                [
                    task,
                    lang,
                    len(values),
                    round(
                        statistics.mean(values),
                        4,
                    ),
                ]
            )

    print(
        f"Saved {len(output_rows)} rows to "
        f"{SCORES_PATH}"
    )

    print(
        f"Saved averages to "
        f"{AVERAGES_PATH}"
    )


if __name__ == "__main__":
    build_scores()