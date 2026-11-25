from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from analystagent.db.connection import connect_readonly
from analystagent.db.schema import SchemaInfo, load_schema


@dataclass(frozen=True)
class Database:
    path: Path
    schema: SchemaInfo

    @classmethod
    def open(cls, path: str | Path) -> Database:
        resolved = Path(path).resolve()
        return cls(resolved, load_schema(resolved))

    def connect(self) -> sqlite3.Connection:
        return connect_readonly(self.path, self.schema.table_names)
