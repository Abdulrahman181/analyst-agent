from __future__ import annotations

from typing import Any


def render_rows(
    columns: tuple[str, ...], rows: tuple[tuple[Any, ...], ...], language: str, limit: int = 30
) -> str:
    title = "النتيجة" if language == "ar" else "Result"
    if not rows:
        return f"{title}: " + ("لا توجد صفوف." if language == "ar" else "no rows.")
    lines = []
    for row in rows[:limit]:
        cells = [f"{name}={value}" for name, value in zip(columns, row, strict=False)]
        lines.append("; ".join(cells))
    return f"{title}:\n" + "\n".join(lines)
