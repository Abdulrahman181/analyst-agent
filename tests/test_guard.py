from __future__ import annotations

import pytest

from analystagent.db.guard import check_sql


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT COUNT(*) FROM Customer",
        "select name from track limit 3",
        "SELECT t.Name FROM Track t JOIN Album a ON a.AlbumId = t.AlbumId LIMIT 3",
        "SELECT Name FROM Track WHERE Name LIKE '%rock%' ORDER BY Name",
        "SELECT 1 UNION SELECT 2",
        "SELECT 'x' || 'y'",
        "WITH t(a, b) AS (SELECT 1, 2) SELECT a FROM t",
        "WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM c WHERE n < 5) "
        "SELECT n FROM c",
        "SELECT rowid FROM Track LIMIT 1",
        "SELECT COUNT(*) AS n FROM Track ORDER BY n",
        "SELECT * FROM (SELECT 1 AS a)",
        "SELECT strftime('%Y', InvoiceDate) AS y FROM Invoice GROUP BY y",
        "SELECT Name FROM Track;",
        "SELECT 1 -- ; DROP TABLE Customer",
        "SELECT Name FROM Artist WHERE Name = 'Ünïcode'",
        "SELECT CustomerId, SUM(Total) OVER (PARTITION BY CustomerId) FROM Invoice",
    ],
)
def test_allowed_queries(sql, chinook):
    verdict = check_sql(sql, chinook.schema)
    assert verdict.ok, verdict.message()
    assert verdict.sql == sql.strip()


@pytest.mark.parametrize(
    ("sql", "code"),
    [
        ("DELETE FROM Customer", "not_select"),
        ("UPDATE Customer SET Email = 'x'", "not_select"),
        ("INSERT INTO Genre VALUES (99, 'x')", "not_select"),
        ("DROP TABLE Customer", "not_select"),
        ("CREATE TABLE x (a INT)", "not_select"),
        ("ALTER TABLE Customer ADD COLUMN x INT", "not_select"),
        ("PRAGMA table_info(Track)", "not_select"),
        ("ATTACH DATABASE 'x.db' AS y", "not_select"),
        ("VACUUM", "not_select"),
        ("BEGIN", "not_select"),
        ("COMMIT", "not_select"),
        ("EXPLAIN SELECT 1", "not_select"),
        ("WITH a AS (SELECT 1) INSERT INTO Genre SELECT * FROM a", "not_select"),
        ("SELECT 1; SELECT 2", "multiple_statements"),
        ("SELECT 1; DROP TABLE Customer", "multiple_statements"),
        ("SELECT load_extension('evil')", "forbidden_function"),
        ("SELECT randomblob(1000000)", "forbidden_function"),
        ("SELECT zeroblob(1000000000)", "forbidden_function"),
        ("SELECT * INTO copy FROM Track", "forbidden_statement"),
        ("SELECT * FROM pragma_table_info('Track')", "table_function"),
        ("SELECT value FROM json_each('[1,2]')", "table_function"),
        ("SELECT * FROM main.Track", "qualified_table"),
        ("SELECT * FROM temp.sqlite_temp_master", "qualified_table"),
        ("SELECT * FROM sqlite_master", "unknown_table"),
        ("SELECT * FROM Nope", "unknown_table"),
        ("SELECT * FROM Track JOIN Nope ON 1 = 1", "unknown_table"),
        ('SELECT "Nope" FROM Track', "unknown_column"),
        ("SELECT nope FROM Track", "unknown_column"),
        ("", "empty"),
        ("   ", "empty"),
        (";", "empty"),
    ],
)
def test_rejected_queries(sql, code, chinook):
    verdict = check_sql(sql, chinook.schema)
    assert not verdict.ok
    assert verdict.code == code, verdict.message()
    assert verdict.reason


def test_overlong_sql_is_rejected(chinook):
    verdict = check_sql("SELECT " + "1," * 4000 + "1", chinook.schema)
    assert verdict.code == "too_long"


def test_unparseable_sql_is_reported(chinook):
    verdict = check_sql("SELECT FROM WHERE (", chinook.schema)
    assert not verdict.ok
    assert verdict.code in {"parse_error", "not_select"}


def test_security_flag_separates_attacks_from_mistakes(chinook):
    attack = check_sql("DROP TABLE Customer", chinook.schema)
    probe = check_sql("SELECT * FROM sqlite_master", chinook.schema)
    mistake_table = check_sql("SELECT * FROM Nope", chinook.schema)
    mistake_column = check_sql("SELECT nope FROM Track", chinook.schema)
    ok = check_sql("SELECT 1", chinook.schema)
    assert attack.security and probe.security
    assert not mistake_table.security and not mistake_column.security and not ok.security


def test_table_names_are_case_insensitive(chinook):
    assert check_sql("SELECT * FROM TRACK LIMIT 1", chinook.schema).tables == ("Track",)


def test_tables_used_are_reported_once(chinook):
    verdict = check_sql(
        "SELECT a.Name FROM Artist a JOIN Artist b ON a.ArtistId = b.ArtistId", chinook.schema
    )
    assert verdict.tables == ("Artist",)


def test_cte_name_does_not_need_to_be_a_table(chinook):
    verdict = check_sql(
        "WITH totals AS (SELECT CustomerId AS cid, SUM(Total) AS s FROM Invoice GROUP BY 1) "
        "SELECT cid, s FROM totals ORDER BY s DESC",
        chinook.schema,
    )
    assert verdict.ok


def test_cte_cannot_hide_a_forbidden_table(chinook):
    verdict = check_sql("WITH x AS (SELECT * FROM sqlite_master) SELECT * FROM x", chinook.schema)
    assert verdict.code == "unknown_table"


def test_subquery_cannot_hide_a_forbidden_table(chinook):
    verdict = check_sql("SELECT (SELECT COUNT(*) FROM sqlite_master) FROM Track", chinook.schema)
    assert verdict.code == "unknown_table"


def test_forbidden_function_inside_subquery_is_found(chinook):
    verdict = check_sql(
        "SELECT Name FROM Track WHERE TrackId IN (SELECT randomblob(4))", chinook.schema
    )
    assert verdict.code == "forbidden_function"


def test_double_quoted_string_literal_is_not_silently_accepted(chinook):
    verdict = check_sql('SELECT Name FROM Genre WHERE Name = "Rock"', chinook.schema)
    assert verdict.code == "unknown_column"
    assert "single quotes" in verdict.reason


def test_double_quoted_real_identifier_is_accepted(chinook):
    assert check_sql('SELECT "Name" FROM "Genre"', chinook.schema).ok


def test_select_alias_may_be_used_in_order_by(chinook):
    assert check_sql("SELECT Total * 2 AS doubled FROM Invoice ORDER BY doubled", chinook.schema).ok
