from __future__ import annotations

import hashlib
import shutil
import sqlite3
from pathlib import Path

import pytest

from analystagent.db.connection import QueryError, QueryTimeout, connect_readonly, execute_query


@pytest.fixture
def conn(chinook):
    connection = chinook.connect()
    yield connection
    connection.close()


def test_simple_select(conn):
    result = execute_query(conn, "SELECT COUNT(*) FROM Track", 10, 5)
    assert result.rows == ((3503,),)
    assert result.columns == ("COUNT(*)",)
    assert not result.truncated
    assert result.row_count == 1


def test_row_cap_marks_truncation(conn):
    result = execute_query(conn, "SELECT TrackId FROM Track", 5, 5)
    assert result.row_count == 5 and result.truncated


def test_exactly_at_cap_is_not_truncated(conn):
    result = execute_query(conn, "SELECT GenreId FROM Genre", 25, 5)
    assert result.row_count == 25 and not result.truncated


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM Track",
        "INSERT INTO Genre VALUES (99, 'x')",
        "UPDATE Genre SET Name = 'x'",
        "DROP TABLE Track",
        "CREATE TABLE x (a INT)",
        "PRAGMA writable_schema = 1",
        "ATTACH DATABASE ':memory:' AS m",
        "VACUUM",
    ],
)
def test_engine_refuses_writes_even_without_the_guard(conn, sql):
    with pytest.raises(QueryError):
        execute_query(conn, sql, 5, 5)


def test_system_tables_are_not_readable(conn):
    with pytest.raises(QueryError):
        execute_query(conn, "SELECT * FROM sqlite_master", 5, 5)


@pytest.mark.parametrize("function", ["randomblob(8)", "zeroblob(10)", "random()"])
def test_denied_functions_are_refused_by_the_engine(conn, function):
    with pytest.raises(QueryError):
        execute_query(conn, f"SELECT {function}", 5, 5)


def test_cte_reads_are_allowed(conn):
    result = execute_query(conn, "WITH t(a) AS (SELECT 41) SELECT a + 1 FROM t", 5, 5)
    assert result.rows == ((42,),)


def test_infinite_recursion_hits_the_time_limit(conn):
    sql = "WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM c) SELECT COUNT(*) FROM c"
    with pytest.raises(QueryTimeout):
        execute_query(conn, sql, 5, 0.3)


def test_connection_is_usable_after_a_timeout(conn):
    sql = "WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM c) SELECT COUNT(*) FROM c"
    with pytest.raises(QueryTimeout):
        execute_query(conn, sql, 5, 0.2)
    assert execute_query(conn, "SELECT 1", 5, 5).rows == ((1,),)


def test_cartesian_product_is_bounded_by_row_cap(conn):
    result = execute_query(conn, "SELECT a.TrackId FROM Track a, Track b, Track c", 10, 5)
    assert result.row_count == 10 and result.truncated


def test_sql_errors_are_wrapped(conn):
    with pytest.raises(QueryError, match="no such column"):
        execute_query(conn, "SELECT nope FROM Track", 5, 5)


def test_blob_values_are_replaced_by_a_placeholder(tmp_path):
    path = tmp_path / "blob.sqlite"
    raw = sqlite3.connect(path)
    raw.execute("CREATE TABLE B (id INTEGER, data BLOB)")
    raw.execute("INSERT INTO B VALUES (1, ?)", (b"\x00\x01\x02",))
    raw.commit()
    raw.close()
    connection = connect_readonly(path, frozenset({"B"}))
    try:
        result = execute_query(connection, "SELECT data FROM B", 5, 5)
    finally:
        connection.close()
    assert result.rows == (("<blob 3 bytes>",),)


def test_oversized_values_are_refused(conn):
    sql = (
        "WITH RECURSIVE t(s) AS (SELECT 'aaaaaaaaaa' UNION ALL SELECT s || s FROM t "
        "WHERE length(s) < 5000000) SELECT MAX(length(s)) FROM t"
    )
    with pytest.raises(QueryError, match="too big"):
        execute_query(conn, sql, 5, 5)


def test_unicode_round_trip(conn):
    result = execute_query(conn, "SELECT upper('abc'), length('عربي')", 5, 5)
    assert result.rows == (("ABC", 4),)


def test_missing_database_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        connect_readonly(tmp_path / "missing.sqlite", frozenset())


def test_path_with_spaces_and_unicode(tmp_path, chinook):
    folder = tmp_path / "my data ملفات"
    folder.mkdir()
    copy = folder / "copy.sqlite"
    shutil.copy(chinook.path, copy)
    connection = connect_readonly(copy, chinook.schema.table_names)
    try:
        assert execute_query(connection, "SELECT COUNT(*) FROM Genre", 5, 5).rows == ((25,),)
    finally:
        connection.close()


def test_database_file_is_never_modified(chinook, conn):
    before = hashlib.sha256(Path(chinook.path).read_bytes()).hexdigest()
    for sql in ("DELETE FROM Customer", "DROP TABLE Track", "UPDATE Genre SET Name = 'x'"):
        with pytest.raises(QueryError):
            execute_query(conn, sql, 5, 5)
    after = hashlib.sha256(Path(chinook.path).read_bytes()).hexdigest()
    assert before == after


def test_table_outside_allow_list_is_blocked_by_the_engine(chinook):
    restricted = connect_readonly(chinook.path, frozenset({"Track"}))
    try:
        assert execute_query(restricted, "SELECT COUNT(*) FROM Track", 5, 5).rows == ((3503,),)
        with pytest.raises(QueryError):
            execute_query(restricted, "SELECT COUNT(*) FROM Customer", 5, 5)
    finally:
        restricted.close()
