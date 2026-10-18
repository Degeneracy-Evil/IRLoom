"""Transparent descriptive statistics; no learned model or future-IR features."""
import json
from pathlib import Path
import statistics


def summarize(records):
    rows = []
    for r in records:
        row = {k: r.get(k) for k in ("case_id", "candidate_id", "status", "failure_stage", "error",
                                    "compile_seconds", "duplicate_of", "ir_changed")}
        samples = r.get("timings_ms", [])
        if r["status"] == "ok" and r.get("correctness", {}).get("status") == "pass" and len(samples) >= 10:
            median = statistics.median(samples)
            q = statistics.quantiles(samples, n=4, method="inclusive")
            row.update(median_ms=median, min_ms=min(samples), max_ms=max(samples),
                       mad_ms=statistics.median(abs(x-median) for x in samples),
                       iqr_ms=q[2]-q[0], sample_count=len(samples))
        rows.append(row)
    for row in rows:
        baseline = next((x for x in rows if x["case_id"] == row["case_id"] and
                         x["candidate_id"] == "default" and "median_ms" in x), None)
        if baseline and "median_ms" in row:
            row["speedup_vs_default"] = baseline["median_ms"] / row["median_ms"]
    return sorted(rows, key=lambda r: (r["case_id"], r.get("median_ms", float("inf"))))


def save_summary(records, path):
    rows = summarize(records)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    return rows
