"""Unit tests for phone-set reconciliation (no model download required)."""
from app.g2p import build_maps, normalize, to_token_ids


def test_normalize_strips_stress_and_length():
    assert normalize("ˈiː") == "i"
    assert normalize("ˌʌ") == "ʌ"
    assert normalize("ɪ") == "ɪ"


def test_to_token_ids_exact_match():
    vocab = {"ð": 5, "ɪ": 6, "n": 7, "<pad>": 0}
    exact, norm = build_maps(vocab)
    ids, unmapped = to_token_ids(["ð", "ɪ", "n"], exact, norm)
    assert ids == [5, 6, 7]
    assert unmapped == []


def test_to_token_ids_normalized_fallback():
    # vocab has the bare symbol; G2P emitted a stressed/length-marked variant
    vocab = {"i": 3, "<pad>": 0}
    exact, norm = build_maps(vocab)
    ids, unmapped = to_token_ids(["ˈiː"], exact, norm)
    assert ids == [3]
    assert unmapped == []


def test_to_token_ids_reports_unmapped():
    vocab = {"ð": 5, "<pad>": 0}
    exact, norm = build_maps(vocab)
    ids, unmapped = to_token_ids(["ð", "zzz"], exact, norm)
    assert ids == [5]
    assert unmapped == ["zzz"]
