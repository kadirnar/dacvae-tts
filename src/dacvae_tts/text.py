import re
import unicodedata

import numpy as np
import torch

PAD, BOS, SEP, EOS, BYTE_OFFSET = 0, 1, 2, 3, 4
SPACE = 32 + BYTE_OFFSET
VOCAB_SIZE = 260
TEXT_VERSIONS = {"unicode-v1", "english-explicit-v2", "turkish-v1", "turkish-v2"}


def normalize(text: str, version="unicode-v1", spoken_text=None) -> str:
    if version not in TEXT_VERSIONS:
        raise ValueError(f"Unknown normalization version: {version}")
    if not isinstance(text, str) or (spoken_text is not None and not isinstance(spoken_text, str)):
        raise ValueError("Transcript must be a string")
    text = text if spoken_text is None else spoken_text
    if version in {"turkish-v1", "turkish-v2"}:
        from .turkish import normalize_turkish

        return normalize_turkish(text, version)
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


def tokenize(reference: str, target: str, version="unicode-v1", layout="segments", units="bytes"):
    ref = normalize(reference, version).encode("utf-8") if reference.strip() else b""
    tgt = normalize(target, version).encode("utf-8")
    return to_units(*tokenize_bytes(ref, tgt, layout), units)


# Character units (model.text_units: chars): one token per character. Ids are ISO-8859-9 (Latin-5, the Turkish
# 8-bit code page) + BYTE_OFFSET: every ASCII character keeps its byte id, so spaces, punctuation, a-z and the
# vocabulary size are those of byte models (embedding shapes, SPACE and the `tokens >= BYTE_OFFSET` "is text"
# checks carry over), and every symbol turkish-v1/v2 lets through (Turkish letters with â î û and capitals) is one
# token at 0xC2-0xFE instead of two UTF-8 bytes: an even length-aware RoPE diagonal, one CTC label per letter and
# a per-character duration rule. A frozen standard, so ids never depend on a table order. No character id lies in
# the UTF-8 continuation range 0x80-0xBF (Latin-5 symbols there map to UNIT_UNKNOWN), which (1) keeps UTF-8
# lead-byte character counts valid on character rows and (2) tells byte rows from character rows (every non-ASCII
# UTF-8 letter has a continuation byte). Other characters fall back to their NFKD base letter (ć -> c), else
# UNIT_UNKNOWN (ASCII SUB). Caches keep their UTF-8 byte ids; `to_units` converts at load time.
UNIT_CODEC = "iso8859_9"
UNIT_UNKNOWN = 0x1A + BYTE_OFFSET
CONTINUATION = (0x80 + BYTE_OFFSET, 0xBF + BYTE_OFFSET)  # UTF-8 continuation bytes as ids (inclusive)
TEXT_UNITS = ("bytes", "chars")


def _unit_id(character):
    for candidate in (character, unicodedata.normalize("NFKD", character)[:1]):
        try:
            value = candidate.encode(UNIT_CODEC)[0] if candidate else None
        except UnicodeEncodeError:
            continue
        if value is not None and not 0x80 <= value <= 0xBF and (value >= 0x20 or candidate in "\t\n"):
            return value + BYTE_OFFSET
    return UNIT_UNKNOWN


def char_units(text):
    """String -> character unit ids (NFC; Latin-5 + BYTE_OFFSET, see UNIT_CODEC). Combining marks left after NFC
    are dropped: "i̇" (a non-Turkish lower() of İ) reads as "i"."""
    return [_unit_id(c) for c in unicodedata.normalize("NFC", text) if unicodedata.category(c) != "Mn"]


def units_text(values, units="bytes"):
    """Text-unit ids (>= BYTE_OFFSET) -> string: UTF-8 bytes, or character units (Latin-5; unknown as U+FFFD)."""
    data = bytes(v - BYTE_OFFSET for v in values)
    if units == "bytes":
        return data.decode("utf-8", errors="ignore")
    return data.decode(UNIT_CODEC).replace("\x1a", "\ufffd")


def to_units(tokens, segments, units="bytes"):
    """Assembled byte-id rows (tokens, segments [S]) -> the same rows in `units`; bytes returns them unchanged.

    Each run of text ids between special tokens (BOS/SEP/EOS) is decoded from UTF-8 and re-encoded as characters;
    the run keeps its segment. Works for both layouts and for multi-utterance (cross) prompts.
    """
    if units == "bytes":
        return tokens, segments
    if units != "chars":
        raise ValueError(f"text units must be one of {TEXT_UNITS}")
    values, marks = tokens.tolist(), segments.tolist()
    out_tokens, out_segments, run = [], [], []

    def flush():
        if run:
            ids = char_units(units_text([values[i] for i in run]))
            out_tokens.extend(ids)
            out_segments.extend([marks[run[0]]] * len(ids))
            run.clear()

    for i, value in enumerate(values):
        if value >= BYTE_OFFSET:
            run.append(i)
        else:
            flush()
            out_tokens.append(value)
            out_segments.append(marks[i])
    flush()
    return torch.tensor(out_tokens, dtype=torch.int64), torch.tensor(out_segments, dtype=torch.int64)


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


ID_DTYPE = np.uint16  # vocabulary 260 fits; this is the on-disk token format of the cache


def encode_ids(utf8: bytes):
    """Tokenize one normalized transcript once, at preparation time: [BOS] byte+4 ... [EOS]."""
    if not utf8:
        raise ValueError("Empty transcript bytes")
    ids = np.empty(len(utf8) + 2, dtype=ID_DTYPE)
    ids[0], ids[-1] = BOS, EOS
    ids[1:-1] = np.frombuffer(utf8, dtype=np.uint8).astype(ID_DTYPE) + BYTE_OFFSET
    return ids


def decode_ids(blob):
    return np.frombuffer(blob, dtype=ID_DTYPE) if blob is not None else None


def assemble(reference_ids, target_ids, layout="segments"):
    """Combine cached per-utterance token ids into one model input; no tokenization happens here.

    `segments` marks reference and target transcripts ([BOS ref SEP target EOS]). `joined` is one
    sentence stream without a boundary ([BOS ref+" "+target EOS]), the layout a prompt cut from the
    same utterance needs: nobody knows where its transcript ends.
    """
    target_ids = np.asarray(target_ids, dtype=np.int64)
    if len(target_ids) < 3:
        raise ValueError("Empty target transcript ids")
    body = target_ids[1:-1]
    reference_body = np.asarray(reference_ids, dtype=np.int64)[1:-1] if reference_ids is not None else None
    if layout == "joined":
        parts = (
            [reference_body, np.array([SPACE], dtype=np.int64), body]
            if reference_body is not None
            else [body]
        )
        tokens = np.concatenate([np.array([BOS], dtype=np.int64), *parts, np.array([EOS], dtype=np.int64)])
        return torch.from_numpy(tokens), torch.ones(len(tokens), dtype=torch.int64)
    if layout != "segments":
        raise ValueError(f"Unknown text layout: {layout}")
    reference_body = reference_body if reference_body is not None else np.empty(0, dtype=np.int64)
    tokens = np.concatenate(
        [
            np.array([BOS], dtype=np.int64),
            reference_body,
            np.array([SEP], dtype=np.int64),
            body,
            np.array([EOS], dtype=np.int64),
        ]
    )
    segments = np.zeros(len(tokens), dtype=np.int64)
    segments[len(reference_body) + 2 :] = 1
    return torch.from_numpy(tokens), torch.from_numpy(segments)


def tokenize_bytes(reference: bytes, target: bytes, layout="segments"):
    """Legacy path for caches that stored normalized UTF-8 bytes instead of token ids."""
    if not target:
        raise ValueError("Empty target transcript bytes")
    return assemble(encode_ids(reference) if reference else None, encode_ids(target), layout)


def join_ids(parts):
    """Cached ids of several utterances -> one transcript [BOS a SPACE b ... EOS] (a multi-utterance prompt)."""
    if not parts:
        raise ValueError("Nothing to join")
    bodies = []
    for ids in parts:
        if len(ids) < 3:
            raise ValueError("Empty transcript ids")
        bodies += [np.asarray(ids[1:-1], dtype=ID_DTYPE), np.array([SPACE], dtype=ID_DTYPE)]
    return np.concatenate([np.array([BOS], dtype=ID_DTYPE), *bodies[:-1], np.array([EOS], dtype=ID_DTYPE)])


# Character CTC labels (model.ctc_targets: chars): blank 0, space 1, then the Turkish lower-case alphabet
# plus q, w, x for loanwords. The circumflex vowels are the same phones as their plain forms.
CTC_CHARS = " abcdefghijklmnopqrstuvwxyzçğıöşü"
CTC_INDEX = {c: i + 1 for i, c in enumerate(CTC_CHARS)}
CHAR_VOCAB_SIZE = len(CTC_CHARS) + 1
CTC_FOLD = str.maketrans({"â": "a", "î": "i", "û": "u"})


def ctc_text(text):
    """Transcript -> CTC character string: Turkish lower case (İ->i, I->ı), letters and single spaces.

    Apostrophes vanish inside a word ("İstanbul'da" -> "istanbulda"); other punctuation and digits
    separate words, as in the Turkish WER normalization. Foreign accented letters keep their base letter.
    """
    from .turkish import tr_lower

    characters = []
    for c in tr_lower(unicodedata.normalize("NFC", text)).translate(CTC_FOLD):
        if c in "'’":
            continue
        if c in CTC_INDEX:
            characters.append(c)
        elif c.isalpha():
            base = unicodedata.normalize("NFKD", c)[0]
            characters.append(base if base in CTC_INDEX else "")
        else:
            characters.append(" ")
    return " ".join("".join(characters).split())


def char_ctc_targets(rows, units="bytes"):
    """Assembled token rows ([B,S] or a list of [S]) -> padded character targets [B,T] and lengths [B].

    The text runs between special tokens (BOS/SEP/EOS/PAD) are decoded (`units`: UTF-8 bytes or character
    units) and joined by a space, so both layouts and multi-utterance prompts give "reference words target
    words", like the byte targets.
    """
    targets = []
    for row in rows:
        runs, current = [], []
        for value in row.tolist() + [PAD]:
            if value >= BYTE_OFFSET:
                current.append(value)
            elif current:
                runs.append(units_text(current, units))
                current = []
        targets.append(torch.tensor([CTC_INDEX[c] for c in ctc_text(" ".join(runs))], dtype=torch.int64))
    lengths = torch.tensor([len(t) for t in targets], dtype=torch.int64)
    padded = torch.zeros(len(targets), max(int(lengths.max()), 1), dtype=torch.int64)
    for i, target in enumerate(targets):
        padded[i, : len(target)] = target
    return padded, lengths
