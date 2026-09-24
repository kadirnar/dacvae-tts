"""scripts/make_drop_list.py: transcription/DNSMOS scores -> the uid drop list of `merge --drop-uids`."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_drop_list.py"
spec = importlib.util.spec_from_file_location("make_drop_list", SCRIPT)
make_drop_list = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_drop_list)

TEXT = "iki kelimeden uzun bir cümle"
# Rows as scripts/transcribe_corpus.py writes them (quality_score is null for sources without the column).
ROWS = [
    {"uid": "good", "text": TEXT, "cer": 0.02, "wer": 0.1, "dnsmos_ovrl": 3.3, "quality_score": None},
    {"uid": "bad-cer", "text": TEXT, "cer": 0.6, "wer": 0.9, "dnsmos_ovrl": 3.3},
    {"uid": "noisy", "text": TEXT, "cer": 0.0, "wer": 0.0, "dnsmos_ovrl": 2.1},
    {"uid": "short", "text": "tek", "cer": 0.0, "wer": 0.0, "dnsmos_ovrl": 3.3},
    # DNSMOS failed: the CER threshold still applies.
    {"uid": "bad-cer-dnsmos-error", "text": TEXT, "cer": 0.6, "wer": 0.9, "dnsmos_error": "onnx"},
    {"uid": "good-dnsmos-error", "text": TEXT, "cer": 0.0, "wer": 0.0, "dnsmos_error": "onnx"},
    # Whisper/decode failed: the DNSMOS threshold still applies.
    {"uid": "noisy-asr-error", "text": TEXT, "error": "too short", "dnsmos_ovrl": 1.9},
    {"uid": "decode-error", "text": TEXT, "error": "corrupt mp3"},
    {"uid": "nan-cer", "text": TEXT, "cer": float("nan"), "dnsmos_ovrl": 3.3},
]


def run(tmp_path, rows, *extra):
    scores, output = tmp_path / "scores.jsonl", tmp_path / "drop.json"
    scores.write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = make_drop_list.main(["--scores", str(scores), "--output", str(output), *extra])
    return set(json.loads(output.read_text())), summary


def test_thresholds_apply_per_score_and_unscored_rows_are_dropped(tmp_path):
    dropped, summary = run(tmp_path, ROWS)
    assert dropped == {row["uid"] for row in ROWS} - {"good"}
    assert summary["reasons"] == {"cer": 2, "dnsmos": 2, "short": 1, "unscored": 3}
    assert summary["unscored_rows"] == 5
    assert summary["missing_scores"] == {"cer": 3, "dnsmos_ovrl": 3}
    # --drop-errors (the old opt-in) is accepted and is the default behaviour.
    assert run(tmp_path, ROWS, "--drop-errors")[0] == dropped


def test_keep_unscored_still_applies_the_scores_a_row_has(tmp_path):
    dropped, summary = run(tmp_path, ROWS, "--keep-unscored")
    # CER 0.6 with a DNSMOS error and OVRL 1.9 with a Whisper error used to be kept.
    assert dropped == {"bad-cer", "noisy", "short", "bad-cer-dnsmos-error", "noisy-asr-error"}
    assert summary["unscored_rows"] == 5 and "unscored" not in summary["reasons"]
    with pytest.raises(SystemExit):
        run(tmp_path, ROWS, "--keep-unscored", "--drop-errors")


def test_scores_without_dnsmos_still_filter_on_cer(tmp_path, capsys):
    # transcribe_corpus.py without --dnsmos: every row lacks dnsmos_ovrl, which used to drop nothing at all.
    rows = [{k: v for k, v in row.items() if k != "dnsmos_ovrl"} for row in ROWS[:4]]
    dropped, summary = run(tmp_path, rows, "--keep-unscored")
    assert dropped == {"bad-cer", "short"} and summary["missing_scores"] == {"dnsmos_ovrl": 4}
    dropped, summary = run(tmp_path, rows)
    assert dropped == {row["uid"] for row in rows} and "no row has dnsmos_ovrl" in capsys.readouterr().err


def test_optional_thresholds_need_their_scores_only_when_set(tmp_path):
    rows = [{"uid": "a", "text": TEXT, "cer": 0.05, "wer": 0.4, "dnsmos_ovrl": 3.0, "quality_score": None}]
    assert run(tmp_path, rows)[0] == set()
    assert run(tmp_path, rows, "--max-wer", "0.3")[1]["reasons"] == {"wer": 1}
    dropped, summary = run(tmp_path, rows, "--min-quality", "60")
    assert dropped == {"a"} and summary["reasons"] == {"unscored": 1}
    assert summary["missing_scores"] == {"quality_score": 1}
