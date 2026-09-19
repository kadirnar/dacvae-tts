import re
import unicodedata

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
    ref = list(normalize(reference, version).encode("utf-8")) if reference.strip() else []
    tgt = list(normalize(target, version).encode("utf-8"))
    tokens = [BOS] + [v + BYTE_OFFSET for v in ref] + [SEP] + [v + BYTE_OFFSET for v in tgt] + [EOS]
    segments = [0] * (len(ref) + 2) + [1] * (len(tgt) + 1)
    return torch.tensor(tokens, dtype=torch.long), torch.tensor(segments, dtype=torch.long)
