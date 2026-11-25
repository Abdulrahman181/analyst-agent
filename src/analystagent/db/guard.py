from __future__ import annotations

import logging
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from analystagent.db.connection import DENIED_FUNCTIONS, MAX_SQL_LENGTH
from analystagent.db.schema import SchemaInfo

logging.getLogger("sqlglot").setLevel(logging.ERROR)

FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.Command,
    exp.Pragma,
    exp.Attach,
    exp.Detach,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    exp.Merge,
    exp.Set,
    exp.Use,
    exp.Analyze,
)

PSEUDO_COLUMNS = frozenset({"rowid", "_rowid_", "oid"})

SECURITY_CODES = frozenset(
    {
        "forbidden_statement",
        "forbidden_function",
        "not_select",
        "multiple_statements",
        "table_function",
        "qualified_table",
    }
)


@dataclass(frozen=True)
class GuardResult:
    ok: bool
    sql: str
    code: str | None = None
    reason: str | None = None
    tables: tuple[str, ...] = ()

    @property
    def security(self) -> bool:
        if self.ok or self.code is None:
            return False
        if self.code in SECURITY_CODES:
            return True
        return self.code == "unknown_table" and "sqlite_" in (self.reason or "").lower()

    def message(self) -> str:
        return f"[{self.code}] {self.reason}" if not self.ok else "ok"


def _reject(sql: str, code: str, reason: str) -> GuardResult:
    return GuardResult(False, sql, code, reason)


def _cte_names(tree: exp.Expression) -> set[str]:
    return {cte.alias.lower() for cte in tree.find_all(exp.CTE) if cte.alias}


def _declared_names(tree: exp.Expression) -> set[str]:
    names: set[str] = set()
    for alias in tree.find_all(exp.Alias):
        if alias.alias:
            names.add(alias.alias.lower())
    for table_alias in tree.find_all(exp.TableAlias):
        for column in table_alias.columns:
            names.add(column.name.lower())
    return names


def check_sql(sql: str, schema: SchemaInfo) -> GuardResult:
    text = (sql or "").strip()
    if not text:
        return _reject(sql, "empty", "The SQL text is empty.")
    if len(text) > MAX_SQL_LENGTH:
        return _reject(sql, "too_long", f"The SQL exceeds {MAX_SQL_LENGTH} characters.")
    try:
        statements = [s for s in sqlglot.parse(text, read="sqlite") if s is not None]
    except SqlglotError as exc:
        return _reject(sql, "parse_error", f"The SQL could not be parsed: {str(exc)[:160]}")
    if not statements:
        return _reject(sql, "empty", "The SQL contains no statement.")
    if len(statements) > 1:
        return _reject(sql, "multiple_statements", "Only a single statement is allowed.")
    tree = statements[0]
    if not isinstance(tree, exp.Select | exp.SetOperation):
        return _reject(
            sql, "not_select", "Only read-only SELECT queries (optionally with WITH) are allowed."
        )
    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES):
            return _reject(
                sql, "forbidden_statement", f"{type(node).__name__} operations are not allowed."
            )
        if isinstance(node, exp.Into):
            return _reject(sql, "forbidden_statement", "SELECT INTO is not allowed.")
        if isinstance(node, exp.Anonymous | exp.Func):
            name = (node.name or "").lower()
            if isinstance(node, exp.Anonymous):
                name = str(node.this).lower()
            if name in DENIED_FUNCTIONS:
                return _reject(sql, "forbidden_function", f"The function {name}() is not allowed.")

    allowed = {name.lower(): name for name in schema.table_names}
    ctes = _cte_names(tree)
    used: list[str] = []
    for table in tree.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            return _reject(sql, "table_function", "Table-valued functions are not allowed in FROM.")
        if table.db or table.catalog:
            return _reject(sql, "qualified_table", "Schema-qualified table names are not allowed.")
        name = table.name.lower()
        if name in ctes:
            continue
        if name not in allowed:
            known = ", ".join(sorted(allowed.values()))
            return _reject(
                sql, "unknown_table", f"Table '{table.name}' does not exist. Tables: {known}."
            )
        used.append(allowed[name])

    known_columns = schema.column_names | PSEUDO_COLUMNS | _declared_names(tree)
    for column in tree.find_all(exp.Column):
        if isinstance(column.this, exp.Star) or not column.name:
            continue
        if column.name.lower() not in known_columns:
            return _reject(
                sql,
                "unknown_column",
                f"Column '{column.name}' does not exist in any table. "
                "Use single quotes for string literals and describe_table to check columns.",
            )
    return GuardResult(True, text, tables=tuple(sorted(set(used))))
