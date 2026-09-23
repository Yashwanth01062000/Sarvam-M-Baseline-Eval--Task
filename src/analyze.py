"""Create summaries and a simple failure-analysis file."""
import csv
from collections import defaultdict
from .config import RESULTS_PATH, SCORES_PATH, SUMMARY_PATH, FAILURES_PATH

def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def run():
    with open(RESULTS_PATH, encoding="utf-8", newline="") as f:
        results = list(csv.DictReader(f))
    with open(SCORES_PATH, encoding="utf-8", newline="") as f:
        scores = {r["id"]: r for r in csv.DictReader(f)}

    groups = defaultdict(list)
    for r in results:
        s = to_float(scores.get(r["id"], {}).get("score"))
        groups[(r["task"], r["lang"])].append((r, s))

    summary = []
    for (task, lang), items in sorted(groups.items()):
        scored = [s for _, s in items if s is not None]
        latencies = [to_float(r["latency_s"]) for r, _ in items if to_float(r["latency_s"]) is not None]
        summary.append({
            "task": task,
            "lang": lang,
            "count": len(items),
            "scored_count": len(scored),
            "average_score": round(sum(scored) / len(scored), 4) if scored else "",
            "average_latency_s": round(sum(latencies) / len(latencies), 4) if latencies else "",
            "failed_api_calls": sum(r["status"] != "success" for r, _ in items),
        })

    with open(SUMMARY_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else [
            "task","lang","count","scored_count","average_score","average_latency_s","failed_api_calls"
        ])
        writer.writeheader()
        writer.writerows(summary)

    failures = []
    for r in results:
        failure_type = ""
        output = (r["output"] or "").strip()
        if r["status"] != "success":
            failure_type = "api_error"
        elif not output:
            failure_type = "empty_answer"
        elif r["task"] in {"factual_qa", "reasoning_math"} and to_float(scores.get(r["id"], {}).get("score")) == 0:
            failure_type = "incorrect_answer"
        if failure_type:
            failures.append({
                "id": r["id"], "task": r["task"], "lang": r["lang"],
                "failure_type": failure_type, "prompt": r["prompt"],
                "reference": r["reference"], "output": r["output"], "notes": "",
            })

    with open(FAILURES_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id","task","lang","failure_type","prompt","reference","output","notes"
        ])
        writer.writeheader()
        writer.writerows(failures)

    print(f"Saved: {SUMMARY_PATH}")
    print(f"Saved: {FAILURES_PATH}")

if __name__ == "__main__":
    run()
