from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    type: str
    notnull: bool
    pk: bool


@dataclass(frozen=True)
class ForeignKey:
    column: str
    ref_table: str
    ref_column: str


@dataclass(frozen=True)
class TableInfo:
    name: str
    columns: tuple[ColumnInfo, ...]
    foreign_keys: tuple[ForeignKey, ...]
    row_count: int


@dataclass(frozen=True)
class SchemaInfo:
    tables: dict[str, TableInfo] = field(default_factory=dict)

    @property
    def table_names(self) -> frozenset[str]:
        return frozenset(self.tables)

    @property
    def column_names(self) -> frozenset[str]:
        return frozenset(c.name.lower() for t in self.tables.values() for c in t.columns)

    def find_table(self, name: str) -> TableInfo | None:
        lowered = name.lower()
        for table in self.tables.values():
            if table.name.lower() == lowered:
                return table
        return None


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def load_schema(path: str | Path) -> SchemaInfo:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"database not found: {resolved}")
    conn = sqlite3.connect(resolved.as_uri() + "?mode=ro", uri=True)
    try:
        names = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        tables: dict[str, TableInfo] = {}
        for name in names:
            columns = tuple(
                ColumnInfo(row[1], row[2] or "", bool(row[3]), bool(row[5]))
                for row in conn.execute(f"PRAGMA table_info({_quote(name)})")
            )
            fks = tuple(
                ForeignKey(row[3], row[2], row[4] or "")
                for row in conn.execute(f"PRAGMA foreign_key_list({_quote(name)})")
            )
            count = conn.execute(f"SELECT COUNT(*) FROM {_quote(name)}").fetchone()[0]
            tables[name] = TableInfo(name, columns, fks, int(count))
        return SchemaInfo(tables)
    finally:
        conn.close()


def render_table(table: TableInfo) -> str:
    fk_by_column = {fk.column: fk for fk in table.foreign_keys}
    parts = []
    for column in table.columns:
        text = f"{column.name} {column.type}".strip()
        if column.pk:
            text += " PK"
        fk = fk_by_column.get(column.name)
        if fk:
            text += f" -> {fk.ref_table}.{fk.ref_column}"
        parts.append(text)
    return f"{table.name} ({table.row_count} rows): " + ", ".join(parts)


def render_overview(schema: SchemaInfo) -> str:
    return "\n".join(render_table(schema.tables[name]) for name in sorted(schema.tables))
