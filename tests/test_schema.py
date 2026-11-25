from __future__ import annotations

import hashlib

import pytest

from analystagent.db.schema import load_schema, render_overview, render_table

CHINOOK_SHA256 = "7651ba378ac2fcd0dfc3c66fb101f7a7eed3ba39a612ec642b96e20702061f15"


def test_bundled_database_is_the_verified_chinook_file(chinook):
    assert hashlib.sha256(chinook.path.read_bytes()).hexdigest() == CHINOOK_SHA256


def test_license_is_bundled_next_to_the_data(chinook):
    license_text = (chinook.path.parent / "LICENSE-chinook.md").read_text(encoding="utf-8")
    assert "MIT" in license_text


def test_all_tables_are_discovered(chinook):
    assert set(chinook.schema.table_names) == {
        "Album",
        "Artist",
        "Customer",
        "Employee",
        "Genre",
        "Invoice",
        "InvoiceLine",
        "MediaType",
        "Playlist",
        "PlaylistTrack",
        "Track",
    }


def test_row_counts(chinook):
    counts = {name: t.row_count for name, t in chinook.schema.tables.items()}
    assert counts["Track"] == 3503
    assert counts["Invoice"] == 412
    assert counts["Customer"] == 59


def test_foreign_keys_and_primary_keys(chinook):
    track = chinook.schema.find_table("track")
    assert track is not None
    fks = {(fk.column, fk.ref_table, fk.ref_column) for fk in track.foreign_keys}
    assert ("AlbumId", "Album", "AlbumId") in fks
    assert [c.name for c in track.columns if c.pk] == ["TrackId"]


def test_find_table_is_case_insensitive_and_misses_cleanly(chinook):
    assert chinook.schema.find_table("INVOICELINE") is not None
    assert chinook.schema.find_table("nope") is None


def test_column_names_are_lowercased_for_lookup(chinook):
    assert "billingcountry" in chinook.schema.column_names


def test_render_table_mentions_keys(chinook):
    text = render_table(chinook.schema.tables["Album"])
    assert "AlbumId INTEGER PK" in text
    assert "ArtistId INTEGER -> Artist.ArtistId" in text
    assert "347 rows" in text


def test_overview_is_compact_enough_for_a_prompt(chinook):
    overview = render_overview(chinook.schema)
    assert overview.count("\n") == 10
    assert len(overview) < 4000


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_schema(tmp_path / "nope.sqlite")


def test_internal_tables_are_hidden(tiny_db):
    assert "sqlite_sequence" not in tiny_db.schema.table_names
