import csv
from collections import defaultdict
from pathlib import Path

from src.config import (
    RESULTS_PATH,
    SCORES_PATH,
    REPORT_PATH,
)


def read_csv(path):
    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        return list(csv.DictReader(f))


def mean(values):
    values = [
        float(v)
        for v in values
        if v not in ("", None)
    ]

    if not values:
        return None

    return sum(values) / len(values)


def main():
    results = read_csv(RESULTS_PATH)
    scores = read_csv(SCORES_PATH)

    success = [
        r for r in results
        if r["status"] == "success"
    ]

    failures = [
        r for r in results
        if r["status"] != "success"
    ]

    total_latency = mean(
        [r["latency_s"] for r in results]
    )

    total_input_tokens = sum(
        int(r["input_tokens"] or 0)
        for r in results
    )

    total_output_tokens = sum(
        int(r["output_tokens"] or 0)
        for r in results
    )

    total_tokens = sum(
        int(r["total_tokens"] or 0)
        for r in results
    )

    total_cost = sum(
        float(r["estimated_cost_inr"] or 0)
        for r in scores
    )

    task_groups = defaultdict(list)

    for row in scores:
        task_groups[row["task"]].append(row)

    language_groups = defaultdict(list)

    for row in scores:
        language_groups[row["lang"]].append(row)

    report = []

    report.append("# Sarvam Baseline Evaluation Report")
    report.append("")
    report.append("## 1. Evaluation Summary")
    report.append("")
    report.append(f"- Total prompts: **{len(results)}**")
    report.append(f"- Successful API calls: **{len(success)}**")
    report.append(f"- Failed API calls: **{len(failures)}**")
    report.append(
        f"- Average latency: **{total_latency:.3f} seconds**"
    )
    report.append(
        f"- Total input tokens: **{total_input_tokens:,}**"
    )
    report.append(
        f"- Total output tokens: **{total_output_tokens:,}**"
    )
    report.append(
        f"- Total tokens: **{total_tokens:,}**"
    )
    report.append(
        f"- Estimated API cost: **₹{total_cost:.4f}**"
    )
    report.append("")

    report.append("## 2. Task-Level Averages")
    report.append("")
    report.append(
        "| Task | Prompts | Average Score | Average Latency | Tokens | Cost |"
    )
    report.append(
        "|---|---:|---:|---:|---:|---:|"
    )

    for task, rows in sorted(task_groups.items()):
        numeric_scores = [
            float(r["score"])
            for r in rows
            if r["score"] != ""
        ]

        avg_score = (
            mean(numeric_scores)
            if numeric_scores
            else None
        )

        lat = mean(
            [r["latency_s"] for r in rows]
        )

        tokens = sum(
            int(r["total_tokens"] or 0)
            for r in rows
        )

        cost = sum(
            float(r["estimated_cost_inr"] or 0)
            for r in rows
        )

        score_text = (
            f"{avg_score:.4f}"
            if avg_score is not None
            else "Pending human scoring"
        )

        report.append(
            f"| {task} | {len(rows)} | "
            f"{score_text} | "
            f"{lat:.3f}s | "
            f"{tokens:,} | "
            f"₹{cost:.4f} |"
        )

    report.append("")
    report.append("## 3. Language-Level Averages")
    report.append("")
    report.append(
        "| Language | Prompts | Average Score | Average Latency |"
    )
    report.append(
        "|---|---:|---:|---:|"
    )

    for lang, rows in sorted(language_groups.items()):
        numeric_scores = [
            float(r["score"])
            for r in rows
            if r["score"] != ""
        ]

        avg_score = (
            mean(numeric_scores)
            if numeric_scores
            else None
        )

        lat = mean(
            [r["latency_s"] for r in rows]
        )

        score_text = (
            f"{avg_score:.4f}"
            if avg_score is not None
            else "Pending human scoring"
        )

        report.append(
            f"| {lang} | {len(rows)} | "
            f"{score_text} | {lat:.3f}s |"
        )

    report.append("")
    report.append("## 4. Failure Analysis")
    report.append("")

    if not failures:
        report.append(
            "No API failures occurred during the evaluation run."
        )
    else:
        report.append(
            f"{len(failures)} prompts failed during the API run:"
        )
        report.append("")

        for row in failures:
            report.append(
                f"- ID {row['id']} "
                f"({row['task']}, {row['lang']}): "
                f"{row['error']}"
            )

    report.append("")
    report.append("## 5. Latency Analysis")
    report.append("")

    latency_values = [
        float(r["latency_s"])
        for r in results
    ]

    if latency_values:
        report.append(
            f"- Minimum latency: **{min(latency_values):.3f}s**"
        )
        report.append(
            f"- Maximum latency: **{max(latency_values):.3f}s**"
        )
        report.append(
            f"- Average latency: **{mean(latency_values):.3f}s**"
        )

    report.append("")
    report.append("## 6. Token Usage and Cost")
    report.append("")
    report.append(
        f"- Input tokens: **{total_input_tokens:,}**"
    )
    report.append(
        f"- Output tokens: **{total_output_tokens:,}**"
    )
    report.append(
        f"- Total tokens: **{total_tokens:,}**"
    )
    report.append(
        f"- Estimated cost: **₹{total_cost:.4f}**"
    )
    report.append("")
    report.append(
        "Cost is calculated using the configured Sarvam 105B "
        "input/output token rates."
    )

    report.append("")
    report.append("## 7. Evaluation Notes")
    report.append("")
    report.append(
        "- Evaluation was generated directly from the current "
        "`data/prompts.csv`."
    )
    report.append(
        "- Temperature was set to 0."
    )
    report.append(
        "- `results.csv` and `scores.csv` were generated from "
        "the same evaluation run."
    )
    report.append(
        "- Summarization scores require human ratings and are "
        "not fabricated by the automated scorer."
    )

    Path(REPORT_PATH).write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
