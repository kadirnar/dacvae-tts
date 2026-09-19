import re
import unicodedata

import numpy as np
import torch

PAD, BOS, SEP, EOS, BYTE_OFFSET = 0, 1, 2, 3, 4
VOCAB_SIZE = 260
TEXT_VERSIONS = {"unicode-v1", "english-explicit-v2"}


def normalize(text: str, version="unicode-v1", spoken_text=None) -> str:
    if version not in TEXT_VERSIONS:
        raise ValueError(f"Unknown normalization version: {version}")
    if not isinstance(text, str) or (spoken_text is not None and not isinstance(spoken_text, str)):
        raise ValueError("Transcript must be a string")
    text = text if spoken_text is None else spoken_text
    text = unicodedata.normalize("NFKC", text).translate(
        str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "..."})
    )
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise ValueError("Empty transcript")
    if version == "english-explicit-v2" and (re.search(r"\d|[$€£%]|\b(?:Dr|Mr|Mrs|Ms|St|vs|etc)\.", text)):
        raise ValueError(
            "Ambiguous English expression: provide an explicit spoken-form transcript (numbers, dates, units, abbreviations)"
        )
    return text


def tokenize(reference: str, target: str, version="unicode-v1"):
    ref = normalize(reference, version).encode("utf-8") if reference.strip() else b""
    tgt = normalize(target, version).encode("utf-8")
    return tokenize_bytes(ref, tgt)


def tokenize_bytes(reference: bytes, target: bytes):
    """Assemble cached normalized UTF-8 bytes without normalizing every training pair."""
    if not target:
        raise ValueError("Empty target transcript bytes")
    nref = len(reference)
    tokens = np.empty(nref + len(target) + 3, dtype=np.int64)
    tokens[0], tokens[nref + 1], tokens[-1] = BOS, SEP, EOS
    tokens[1 : nref + 1] = np.frombuffer(reference, dtype=np.uint8)
    tokens[nref + 2 : -1] = np.frombuffer(target, dtype=np.uint8)
    tokens[1 : nref + 1] += BYTE_OFFSET
    tokens[nref + 2 : -1] += BYTE_OFFSET
    segments = np.zeros(len(tokens), dtype=np.int64)
    segments[nref + 2 :] = 1
    return torch.from_numpy(tokens), torch.from_numpy(segments)
