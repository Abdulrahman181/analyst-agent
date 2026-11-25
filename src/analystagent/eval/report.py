from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def render_markdown(report: dict[str, Any]) -> str:
    meta = report["meta"]
    agg = report["aggregate"]
    low, high = agg["exec_accuracy_ci95"]
    lines = [
        "# AnalystAgent evaluation report",
        "",
        f"- Mode: {meta.get('mode', 'n/a')}",
        f"- Provider: {meta.get('provider', 'n/a')}",
        f"- Model: {meta.get('model', 'n/a')}",
        f"- Started: {meta['started_at']}",
        f"- Python: {meta['python']} on {meta['platform']}",
        f"- Items: {agg['n_items']} ({agg['n_answer_items']} answerable, "
        f"{agg['n_decline_items']} must-decline), repeats: {meta['repeats']}",
        "",
        "## Aggregate",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Execution accuracy (answerable, projection-tolerant) | {_pct(agg['exec_accuracy'])} |",
        f"| Wilson 95% interval (items treated as independent) | {_pct(low)} to {_pct(high)} |",
        f"| Execution accuracy, strict column match | {_pct(agg['exec_accuracy_strict'])} |",
        f"| Valid-SQL rate (cited SQL executed) | {_pct(agg['valid_sql_rate'])} |",
        f"| Correct decline rate (unanswerable or unsafe) | {_pct(agg['decline_accuracy'])} |",
        f"| Overall accuracy (all items) | {_pct(agg['overall_accuracy'])} |",
        f"| Grounded-answer rate (model-written answers) | {_pct(agg['grounded_rate'])} |",
        f"| Runs with a grounding rejection | {_pct(agg['grounding_rejection_rate'])} |",
        f"| Runs with a guard security block | {_pct(agg['security_block_rate'])} |",
        f"| Failed runs (no answer produced) | {_pct(agg['failed_rate'])} |",
        f"| Mean LLM turns | {agg['mean_steps']:.1f} |",
        f"| Mean tokens per item | {agg['mean_total_tokens']:.0f} |",
        f"| Mean wall time per item | {agg['mean_wall_s']:.1f}s |",
        "",
        "## Execution accuracy by language",
        "",
        "| Language | Execution accuracy |",
        "|---|---|",
    ]
    for lang, value in agg["language_exec_accuracy"].items():
        lines.append(f"| {lang} | {_pct(value)} |")
    gap = agg["language_gap_en_minus_ar"]
    if gap is not None:
        lines.append(f"| gap (en minus ar) | {gap * 100:+.1f} points |")
    lines += [
        "",
        "## Confidence calibration",
        "",
        "| Confidence | Items | Accuracy |",
        "|---|---|---|",
    ]
    for band, stats in agg["calibration"].items():
        lines.append(f"| {band} | {int(stats['n'])} | {_pct(stats['accuracy'])} |")
    for title, key in (
        ("By difficulty", "by_difficulty"),
        ("By category", "by_category"),
    ):
        lines += ["", f"## {title}", "", "| Group | Items | Accuracy |", "|---|---|---|"]
        for name, stats in agg[key].items():
            lines.append(f"| {name} | {int(stats['n'])} | {_pct(stats['accuracy'])} |")
    lines += ["", "## Stop reasons", "", "| Reason | Items |", "|---|---|"]
    for reason, count in sorted(agg["stop_reasons"].items()):
        lines.append(f"| {reason} | {count} |")
    lines += [
        "",
        "## Per item",
        "",
        "| Item | Repeat | Correct | Status | Stop | Confidence | Turns | Tokens |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for item in report["results"]:
        tokens = item["prompt_tokens"] + item["completion_tokens"]
        lines.append(
            f"| {item['item_id']} | {item['repeat']} | {item['correct']} | {item['status']}"
            f" | {item['stop_reason']} | {item['confidence']} | {item['steps']} | {tokens} |"
        )
    lines.append("")
    return "\n".join(lines)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or "run"


def write_report(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = report["meta"]
    stamp = re.sub(r"[^0-9]", "", meta["started_at"])[:14]
    stem = (
        f"eval_{_slug(str(meta.get('mode', 'na')))}_{_slug(str(meta.get('provider', 'na')))}"
        f"_{_slug(str(meta.get('model', 'na')))}_{stamp}"
    )
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path
