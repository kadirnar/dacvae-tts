"""Character text units (model.text_units: chars): Latin-5 ids, load-time conversion, model/inference paths."""

import random
import string
import unicodedata
from types import SimpleNamespace

import pytest
import torch

from dacvae_tts.config import Config, ModelConfig
from dacvae_tts.data import LatentDataset, collate
from dacvae_tts.model import FlowTTS
from dacvae_tts.text import (
    BYTE_OFFSET,
    CONTINUATION,
    SEP,
    SPACE,
    UNIT_UNKNOWN,
    char_ctc_targets,
    char_units,
    corrupt_transcript,
    encode_ids,
    normalize,
    to_units,
    tokenize,
    units_text,
)
from dacvae_tts.turkish import LATIN_EXTRA, PUNCTUATION

SENTENCE = "Şimdi İstanbul'a gidiyoruz, ığdır'da güzel çöp şişeleri var."
PROMPT = "Işık yağmurlu, soğuk günde."


def test_every_normalizer_symbol_is_one_distinct_latin5_unit():
    symbols = string.ascii_letters + "".join(sorted(LATIN_EXTRA)) + "".join(sorted(PUNCTUATION)) + " "
    ids = char_units(symbols)
    assert len(ids) == len(symbols) == 82 and len(set(ids)) == len(ids)
    assert all(BYTE_OFFSET <= i < 260 and i != UNIT_UNKNOWN for i in ids)
    assert not any(CONTINUATION[0] <= i <= CONTINUATION[1] for i in ids)
    assert all(char_units(c) == [ord(c) + BYTE_OFFSET] for c in string.printable if c.isprintable())  # ASCII = bytes
    # Pinned: the Turkish letters at their ISO-8859-9 code points (a frozen standard, not a table order).
    pinned = {"ç": 0xE7, "ğ": 0xF0, "ı": 0xFD, "ö": 0xF6, "ş": 0xFE, "ü": 0xFC, "Ç": 0xC7, "Ğ": 0xD0, "İ": 0xDD,
              "Ö": 0xD6, "Ş": 0xDE, "Ü": 0xDC, "â": 0xE2, "î": 0xEE, "û": 0xFB, "Â": 0xC2, "Î": 0xCE, "Û": 0xDB}
    assert {c: char_units(c)[0] - BYTE_OFFSET for c in pinned} == pinned
    assert units_text(ids, "chars") == unicodedata.normalize("NFC", symbols)
    # İ I i ı are four units; "i̇" (a non-Turkish lower() of İ) reads as i; foreign letters fold or are unknown.
    assert len(set(char_units("İIiı"))) == 4 and char_units("i̇") == char_units("i")
    assert char_units("ć") == char_units("c") and char_units("€°«") == [UNIT_UNKNOWN] * 3


@pytest.mark.parametrize("layout", ["joined", "segments"])
def test_character_rows_match_the_byte_rows_text(layout):
    t, s = tokenize(PROMPT, SENTENCE, "turkish-v1", layout)
    c, cs = tokenize(PROMPT, SENTENCE, "turkish-v1", layout, units="chars")
    assert torch.equal(torch.stack(to_units(t, s, "chars")), torch.stack([c, cs]))
    text = normalize(PROMPT, "turkish-v1"), normalize(SENTENCE, "turkish-v1")
    assert len(c) == len(text[0]) + len(text[1]) + 3  # BOS, EOS and the SPACE (joined) or SEP (segments)
    assert len(t) - len(c) == sum(ord(ch) > 127 for ch in "".join(text))  # one token less per two-byte letter
    runs = [v for v in c.tolist() if v >= BYTE_OFFSET]
    assert units_text(runs, "chars").replace(" ", "") == "".join(text).replace(" ", "")
    if layout == "segments":
        assert (c == SEP).sum() == 1 and cs[: len(text[0]) + 2].eq(0).all() and cs[len(text[0]) + 2 :].eq(1).all()
    else:
        assert cs.eq(1).all()
    ascii_only = tokenize("Hello there.", "A plain sentence.", "unicode-v1", layout)
    assert torch.equal(torch.stack(ascii_only), torch.stack(to_units(*ascii_only, "chars")))  # ASCII: bit-identical


def item(units=None, branch="ids", reference=True):
    ref_text, text = normalize(PROMPT, "turkish-v1"), normalize(SENTENCE, "turkish-v1")
    base = {"target": torch.zeros(6, 4), "reference": torch.zeros(3 if reference else 0, 4), "layout": "joined",
            "text": text, "reference_text": ref_text if reference else "", "text_normalization": "turkish-v1"}
    if branch == "ids":
        base.update(token_ids=encode_ids(text.encode()),
                    reference_token_ids=encode_ids(ref_text.encode()) if reference else None)
    elif branch == "bytes":
        base.update(text_bytes=text.encode(), reference_text_bytes=ref_text.encode() if reference else b"")
    return {**base, **({"text_units": units} if units else {})}


def test_collate_converts_every_tokenization_branch_and_refuses_mixed_units():
    rows = [collate([item("chars", branch)])["tokens"][0] for branch in ("ids", "bytes", "text")]
    assert all(torch.equal(rows[0], row) for row in rows[1:])
    assert torch.equal(rows[0], tokenize(PROMPT, SENTENCE, "turkish-v1", "joined", units="chars")[0])
    plain = collate([item(None), item(None, reference=False)])
    assert torch.equal(plain["tokens"][0][: len(tokenize(PROMPT, SENTENCE, "turkish-v1", "joined")[0])],
                       tokenize(PROMPT, SENTENCE, "turkish-v1", "joined")[0])  # no flag: the byte rows, unchanged
    with pytest.raises(ValueError, match="mix text units"):
        collate([item("chars"), item(None)])
    chars = collate([{**item("chars"), "ctc_targets": "chars"}])
    by_bytes = collate([{**item(None), "ctc_targets": "chars"}])
    assert torch.equal(chars["ctc_targets"], by_bytes["ctc_targets"])  # same letters either way


def test_word_negatives_and_ctc_targets_work_on_character_rows():
    tokens, segments = tokenize(PROMPT, SENTENCE, "turkish-v1", "joined", units="chars")
    words = units_text([v for v in tokens.tolist() if v >= BYTE_OFFSET], "chars").split()
    for seed in range(20):
        corrupted = corrupt_transcript(tokens, segments, random.Random(seed))
        changed = units_text([v for v in corrupted[0].tolist() if v >= BYTE_OFFSET], "chars").split()
        assert abs(len(changed) - len(words)) == 1 and set(changed) <= set(words)  # whole words skipped/repeated
    byte_rows = tokenize(PROMPT, SENTENCE, "turkish-v1", "joined")[0]
    assert torch.equal(char_ctc_targets([tokens], "chars")[0], char_ctc_targets([byte_rows], "bytes")[0])
    assert SPACE in tokens.tolist()


def test_character_model_has_the_byte_model_shapes_and_guards_its_input():
    options = dict(latent_dim=4, width=16, heads=2, text_depth=1, text_layout="joined", duration="rule",
                   positions="rope", ctc_layer=1)
    byte_model, char_model = FlowTTS(ModelConfig(**options)), FlowTTS(ModelConfig(**options, text_units="chars"))
    assert {k: v.shape for k, v in byte_model.state_dict().items()} == {
        k: v.shape for k, v in char_model.state_dict().items()}
    old = Config.from_dict({"model": {k: v for k, v in ModelConfig(**options).__dict__.items() if k != "text_units"}})
    assert old.model.text_units == "bytes"
    with pytest.raises(ValueError, match="text_units"):
        ModelConfig(**options, text_units="bpe")
    chars, segments = tokenize(PROMPT, SENTENCE, "turkish-v1", "joined", units="chars")
    prompt, mask = torch.zeros(1, 3, 4), torch.ones(1, 3, dtype=torch.bool)
    with torch.no_grad():
        text, valid, voice = char_model.conditions(prompt, mask, chars[None], segments[None])
    assert text.shape == (1, len(chars), 16) and valid.all()
    byte_rows = tokenize(PROMPT, SENTENCE, "turkish-v1", "joined")
    with pytest.raises(ValueError, match="character-unit model"):
        char_model.conditions(prompt, mask, byte_rows[0][None], byte_rows[1][None])


def test_character_models_keep_frames_per_character_in_the_duration_rule():
    from dacvae_tts.inference import Synthesizer

    def target(units, text):
        tts = Synthesizer.__new__(Synthesizer)
        tts.codec = SimpleNamespace(sample_rate=48000, hop_length=1920)
        tts.model = SimpleNamespace(duration=None)
        tts.text_version, tts.duration_model = "turkish-v1", None
        tts.text_units, tts.rule_unit = units, "chars" if units == "chars" else "bytes"
        return tts.target_frames(100, "Kısa bir cümle.", text)

    assert target("chars", "şüphe çığ") == target("chars", "suphe cig")  # Turkish letters count once
    assert target("bytes", "şüphe çığ") > target("bytes", "suphe cig")  # the byte rule gives them twice the time
    assert target("chars", "şüphe")[1]["duration_rule"] == "reference_frames_per_char"


def test_dataset_items_carry_the_units_flag_only_for_characters(cache):
    plain = LatentDataset(cache, "train")[0]
    chars = LatentDataset(cache, "train", text_units="chars")[0]
    assert "text_units" not in plain and chars["text_units"] == "chars"
    with pytest.raises(ValueError, match="text_units"):
        LatentDataset(cache, "train", text_units="phones")
