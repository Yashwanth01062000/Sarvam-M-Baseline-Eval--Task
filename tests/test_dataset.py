import csv
from src.run_eval import validate_prompts

def test_dataset():
    rows = validate_prompts()
    assert len(rows) == 100
    assert len({r["id"] for r in rows}) == 100
    counts = {}
    for r in rows:
        counts[r["task"]] = counts.get(r["task"], 0) + 1
    assert counts == {
        "factual_qa": 25,
        "translation": 25,
        "summarization": 25,
        "reasoning_math": 25,
    }
