"""ARPAbet (speechocean762) -> IPA -> model-token bridge.

speechocean762 labels phones in ARPAbet; our Stage-1 model emits espeak/IPA. To attach a
label to a feature vector we translate each ARPAbet phone to IPA, then resolve it against the
model vocab (exact match, normalized fallback, or per-character split for diphthongs/affricates).
"""
from __future__ import annotations

from app.g2p import normalize

# Base CMU/ARPAbet (39 phones) -> IPA. Diphthongs/affricates are multi-symbol and get
# resolved by splitting downstream.
ARPABET_TO_IPA: dict[str, str] = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "EH": "ɛ", "ER": "ɚ",
    "EY": "eɪ", "F": "f", "G": "ɡ", "HH": "h", "IH": "ɪ", "IY": "i",
    # ER -> ɚ (r-colored schwa); the model vocab uses ɚ, not ɝ, for American English.
    "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ",
    "OW": "oʊ", "OY": "ɔɪ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v", "W": "w",
    "Y": "j", "Z": "z", "ZH": "ʒ",
}


def strip_stress(arpa: str) -> str:
    """Drop the trailing stress digit ARPAbet puts on vowels (e.g. 'IH1' -> 'IH')."""
    return arpa.rstrip("0123456789").upper()


def arpabet_to_ipa(arpa: str) -> str | None:
    return ARPABET_TO_IPA.get(strip_stress(arpa))


def ipa_to_token_ids(
    ipa: str, exact: dict[str, int], norm: dict[str, int]
) -> list[int] | None:
    """Resolve an IPA symbol to one or more model token ids, or None if unmappable."""
    if ipa in exact:
        return [exact[ipa]]
    if normalize(ipa) in norm:
        return [norm[normalize(ipa)]]
    # diphthong / affricate: try splitting into individual symbols
    ids: list[int] = []
    for ch in ipa:
        if ch in exact:
            ids.append(exact[ch])
        elif normalize(ch) in norm:
            ids.append(norm[normalize(ch)])
        else:
            return None
    return ids or None


def arpabet_to_token_ids(
    arpa: str, exact: dict[str, int], norm: dict[str, int]
) -> list[int] | None:
    ipa = arpabet_to_ipa(arpa)
    return ipa_to_token_ids(ipa, exact, norm) if ipa else None
