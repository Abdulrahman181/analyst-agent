from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def _rate(flags: list[bool]) -> float:
    return sum(flags) / len(flags) if flags else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _group(results: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[bool]] = defaultdict(list)
    for item in results:
        groups[str(item[key])].append(bool(item["correct"]))
    return {
        name: {"n": len(flags), "accuracy": _rate(flags)} for name, flags in sorted(groups.items())
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    answer_items = [r for r in results if r["expect"] == "answer"]
    decline_items = [r for r in results if r["expect"] == "decline"]
    correct_total = sum(1 for r in results if r["correct"])
    ex_correct = sum(1 for r in answer_items if r["correct"])
    low, high = wilson_interval(ex_correct, len(answer_items))
    model_answers = [
        r
        for r in results
        if r["answer_source"] == "model" and r["status"] in ("answered", "ungrounded")
    ]
    agent_runs = [r for r in results if r["mode"] == "agent"]
    by_language = {
        lang: _rate([bool(r["correct"]) for r in answer_items if r["language"] == lang])
        for lang in sorted({r["language"] for r in answer_items})
    }
    gap = None
    if "en" in by_language and "ar" in by_language:
        gap = by_language["en"] - by_language["ar"]
    calibration: dict[str, dict[str, float]] = {}
    for band in ("high", "medium", "low"):
        rows = [r for r in results if r["confidence"] == band]
        if rows:
            calibration[band] = {
                "n": len(rows),
                "accuracy": _rate([bool(r["correct"]) for r in rows]),
            }
    return {
        "n_items": n,
        "n_answer_items": len(answer_items),
        "n_decline_items": len(decline_items),
        "overall_correct": correct_total,
        "overall_accuracy": correct_total / n if n else 0.0,
        "exec_accuracy": _rate([bool(r["correct"]) for r in answer_items]),
        "exec_accuracy_ci95": [low, high],
        "exec_accuracy_strict": _rate([bool(r["correct_strict"]) for r in answer_items]),
        "valid_sql_rate": _rate([bool(r["valid_sql"]) for r in answer_items]),
        "decline_accuracy": _rate([bool(r["correct"]) for r in decline_items]),
        "grounded_rate": _rate([bool(r["grounded"]) for r in model_answers])
        if model_answers
        else None,
        "grounding_rejection_rate": _rate([r["grounding_rejections"] > 0 for r in agent_runs])
        if agent_runs
        else None,
        "security_block_rate": _rate([r["security_blocks"] > 0 for r in results]),
        "failed_rate": _rate([r["status"] == "failed" for r in results]),
        "language_exec_accuracy": by_language,
        "language_gap_en_minus_ar": gap,
        "by_category": _group(results, "category"),
        "by_difficulty": _group(results, "difficulty"),
        "by_language": _group(results, "language"),
        "calibration": calibration,
        "mean_steps": _mean([float(r["steps"]) for r in results]),
        "mean_total_tokens": _mean(
            [float(r["prompt_tokens"] + r["completion_tokens"]) for r in results]
        ),
        "mean_wall_s": _mean([float(r["wall_s"]) for r in results]),
        "stop_reasons": dict(Counter(r["stop_reason"] for r in results)),
    }


def _per_item_rates(report: dict[str, Any]) -> dict[str, float]:
    groups: dict[str, list[bool]] = defaultdict(list)
    for item in report["results"]:
        groups[item["item_id"]].append(bool(item["correct"]))
    return {key: _rate(flags) for key, flags in groups.items()}


def compare_reports(
    baseline: dict[str, Any], candidate: dict[str, Any], max_drop: float
) -> dict[str, Any]:
    base_rate = baseline["aggregate"]["overall_accuracy"]
    cand_rate = candidate["aggregate"]["overall_accuracy"]
    base_items = _per_item_rates(baseline)
    cand_items = _per_item_rates(candidate)
    common = sorted(set(base_items) & set(cand_items))
    delta = cand_rate - base_rate
    return {
        "baseline_rate": base_rate,
        "candidate_rate": cand_rate,
        "delta": delta,
        "max_drop": max_drop,
        "passed": delta >= -max_drop,
        "regressions": [k for k in common if cand_items[k] < base_items[k]],
        "improvements": [k for k in common if cand_items[k] > base_items[k]],
        "items_compared": len(common),
    }
