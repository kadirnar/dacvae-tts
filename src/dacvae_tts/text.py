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


LAYOUTS = {"segments", "joined"}


def tokenize(reference: str, target: str, version="unicode-v1", layout="segments"):
    ref = normalize(reference, version).encode("utf-8") if reference.strip() else b""
    tgt = normalize(target, version).encode("utf-8")
    return tokenize_bytes(ref, tgt, layout)


SPACE = 32 + BYTE_OFFSET


def corrupt_transcript(tokens, segments, rng):
    """Return (tokens, segments) with one target word skipped or repeated, or None if impossible.

    Skip/repeat negatives (RobustSpeechFlow, arXiv:2605.22083) give the model a transcript that is
    wrong by exactly one word; it must then prefer the true transcript on the same audio.
    """
    tokens, segments = tokens.tolist(), segments.tolist()
    target = [i for i, (t, s) in enumerate(zip(tokens, segments)) if s == 1 and t >= BYTE_OFFSET]
    if not target:
        return None
    words, current = [], []
    for i in target:
        if tokens[i] == SPACE:
            if current:
                words.append(current)
            current = []
        else:
            current.append(i)
    if current:
        words.append(current)
    if len(words) < 2:
        return None
    word = rng.randrange(len(words))
    first, last = words[word][0], words[word][-1]
    if rng.random() < 0.5:  # skip: drop the word and one adjacent space
        cut_start = first - 1 if first > 0 and tokens[first - 1] == SPACE else first
        cut_end = (
            last + 1 if cut_start == first and last + 1 < len(tokens) and tokens[last + 1] == SPACE else last
        )
        keep = [i for i in range(len(tokens)) if not cut_start <= i <= cut_end]
        new_tokens = [tokens[i] for i in keep]
        new_segments = [segments[i] for i in keep]
    else:  # repeat: "the the"
        new_tokens = tokens[: last + 1] + [SPACE] + tokens[first : last + 1] + tokens[last + 1 :]
        new_segments = segments[: last + 1] + [1] + segments[first : last + 1] + segments[last + 1 :]
    return torch.tensor(new_tokens, dtype=torch.int64), torch.tensor(new_segments, dtype=torch.int64)


def tokenize_bytes(reference: bytes, target: bytes, layout="segments"):
    """Assemble cached normalized UTF-8 bytes without normalizing every training pair.

    `segments` marks reference and target transcripts ([BOS ref SEP target EOS]). `joined` is one
    sentence stream without a boundary ([BOS ref+" "+target EOS]), the layout a prompt cut from the
    same utterance needs: nobody knows where its transcript ends.
    """
    if not target:
        raise ValueError("Empty target transcript bytes")
    if layout == "joined":
        joined = reference + b" " + target if reference else target
        tokens = np.empty(len(joined) + 2, dtype=np.int64)
        tokens[0], tokens[-1] = BOS, EOS
        tokens[1:-1] = np.frombuffer(joined, dtype=np.uint8)
        tokens[1:-1] += BYTE_OFFSET
        return torch.from_numpy(tokens), torch.ones(len(tokens), dtype=torch.int64)
    if layout != "segments":
        raise ValueError(f"Unknown text layout: {layout}")
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
