"""Unit tests for the ARPAbet->IPA->token bridge (no dataset/model download)."""
from app.g2p import build_maps
from stage3.arpabet_ipa import (
    arpabet_to_ipa,
    arpabet_to_token_ids,
    ipa_to_token_ids,
    strip_stress,
)


def test_strip_stress():
    assert strip_stress("IH1") == "IH"
    assert strip_stress("ah0") == "AH"
    assert strip_stress("N") == "N"


def test_arpabet_to_ipa_core_phones():
    assert arpabet_to_ipa("DH") == "ð"
    assert arpabet_to_ipa("IH0") == "ɪ"
    assert arpabet_to_ipa("N") == "n"
    assert arpabet_to_ipa("???") is None


def test_single_symbol_resolves():
    vocab = {"ð": 5, "ɪ": 6, "n": 7, "<pad>": 0}
    exact, norm = build_maps(vocab)
    assert arpabet_to_token_ids("DH", exact, norm) == [5]
    assert arpabet_to_token_ids("IH1", exact, norm) == [6]


def test_diphthong_splits_into_two_tokens():
    # AY -> aɪ ; vocab has the parts but not the combined symbol
    vocab = {"a": 1, "ɪ": 2, "<pad>": 0}
    exact, norm = build_maps(vocab)
    assert ipa_to_token_ids("aɪ", exact, norm) == [1, 2]


def test_unmappable_returns_none():
    vocab = {"n": 7, "<pad>": 0}
    exact, norm = build_maps(vocab)
    assert arpabet_to_token_ids("ZH", exact, norm) is None  # ʒ not in vocab
