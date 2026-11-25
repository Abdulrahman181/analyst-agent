from __future__ import annotations

import pytest

from analystagent.semantic.glossary import GlossaryError, load_glossary


@pytest.fixture(scope="module")
def glossary():
    return load_glossary()


def ids(glossary, question):
    return {entry.id for entry in glossary.match(question)}


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is the total revenue?", {"revenue"}),
        ("ما هي إجمالي المبيعات؟", {"revenue"}),
        ("كم بلغت إيرادات العملاء في الولايات المتحدة؟", {"revenue", "customer", "v_usa"}),
        ("How many tracks are in the Rock genre?", {"track", "genre", "v_rock"}),
        ("كم عدد المقاطع في نوع الروك؟", {"track", "genre", "v_rock"}),
        ("Which artist has the most albums?", {"artist", "album"}),
        ("أي فنان لديه أكبر عدد من الألبومات؟", {"artist", "album"}),
        ("What is the average invoice total?", {"average_order", "invoice"}),
        ("ما هو أطول مقطع؟", {"track", "duration"}),
    ],
)
def test_expected_entries_are_found(glossary, question, expected):
    assert expected <= ids(glossary, question)


def test_arabic_definite_article_and_prefixes_are_ignored(glossary):
    assert "revenue" in ids(glossary, "بالإيرادات")
    assert "customer" in ids(glossary, "العملاء")
    assert "customer" in ids(glossary, "للعملاء")


def test_arabic_spelling_variants_match(glossary):
    assert "v_usa" in ids(glossary, "امريكا")
    assert "v_germany" in ids(glossary, "المانيا")
    assert "v_germany" in ids(glossary, "ألمانيا")


def test_english_terms_need_word_boundaries(glossary):
    assert "v_usa" not in ids(glossary, "business customers")
    assert "v_pop" not in ids(glossary, "popular tracks")
    assert "v_rock" not in ids(glossary, "a crockpot")


def test_the_word_us_does_not_trigger_the_usa_hint(glossary):
    assert "v_usa" not in ids(glossary, "Tell us the total revenue")


def test_value_hints_map_arabic_names_to_stored_values(glossary):
    text = glossary.render_hints("إيرادات ألمانيا")
    assert "'Germany'" in text


def test_notes_are_always_rendered(glossary):
    text = glossary.render_hints("zzz")
    assert "2021-01-01 to 2025-12-22" in text
    assert "profit" in text


def test_unrelated_question_matches_nothing(glossary):
    assert ids(glossary, "hello") == set()


def test_custom_glossary_file(tmp_path):
    path = tmp_path / "g.yaml"
    path.write_text(
        "notes: [n1]\nentries:\n  - id: x\n    terms_en: [widget]\n"
        "    terms_ar: [ودجت]\n    meaning: M\n",
        encoding="utf-8",
    )
    custom = load_glossary(path)
    assert {e.id for e in custom.match("Sell a widget")} == {"x"}
    assert {e.id for e in custom.match("بيع ودجت")} == {"x"}
    assert custom.render_hints("widget") == "- M\n- n1"


@pytest.mark.parametrize(
    "content",
    [
        "- just\n- a list\n",
        "entries:\n  - id: x\n",
        "entries:\n  - id: x\n    meaning: a\n  - id: x\n    meaning: b\n",
        "entries: [1]\n",
        "a: [unclosed",
    ],
)
def test_invalid_glossaries_are_rejected(tmp_path, content):
    path = tmp_path / "bad.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(GlossaryError):
        load_glossary(path)


def test_missing_glossary_file(tmp_path):
    with pytest.raises(GlossaryError):
        load_glossary(tmp_path / "missing.yaml")


def test_bundled_glossary_ids_are_unique(glossary):
    entry_ids = [e.id for e in glossary.entries]
    assert len(entry_ids) == len(set(entry_ids)) and len(entry_ids) > 20
