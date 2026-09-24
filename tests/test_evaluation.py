import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        return list(csv.DictReader(f))


def test_prompts_has_100_rows():
    path = ROOT / "data" / "prompts.csv"

    rows = read_csv(path)

    assert len(rows) == 100

    ids = [int(row["id"]) for row in rows]

    assert ids == list(range(1, 101))


def test_each_task_has_25_prompts():
    rows = read_csv(
        ROOT / "data" / "prompts.csv"
    )

    tasks = {}

    for row in rows:
        tasks.setdefault(
            row["task"],
            0
        )
        tasks[row["task"]] += 1

    assert tasks["factual_qa"] == 25
    assert tasks["translation"] == 25
    assert tasks["summarization"] == 25
    assert tasks["reasoning_math"] == 25


def test_results_has_100_rows():
    path = ROOT / "data" / "results.csv"

    if not path.exists():
        return

    rows = read_csv(path)

    assert len(rows) == 100


def test_scores_has_100_rows():
    path = ROOT / "data" / "scores.csv"

    if not path.exists():
        return

    rows = read_csv(path)

    assert len(rows) == 100
