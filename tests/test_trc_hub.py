"""The experiments repo landing page (scripts/trc/hub_index.py) and the per-folder scores.json it is built from."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "trc"))
from hub_index import describe, final, page, pooled  # noqa: E402
from push_snapshot import readme, scores  # noqa: E402

REPO = "VoiceHub/dacvae-tts-tr-combined"


def summary(wer, count=495):
    return {"count": count, "wer": wer, "cer": wer / 2, "sim_o": 0.5, "dnsmos_ovrl": 3.0, "utmos": 2.5, "extra": 1}


def state():
    return {"title": "arm", "notes": "", "steps": {
        "5000": {"checkpoint": "slim", "summary": summary(0.9, 96), "audio": True},
        "20000": {"checkpoint": "full", "summary": summary(0.10), "summary_s1000": summary(0.12), "audio": True}}}


def test_scores_keep_the_landing_page_fields_and_the_second_seed():
    data = scores(state())
    assert set(data["steps"]["20000"]["summary"]) == {"count", "wer", "cer", "sim_o", "dnsmos_ovrl", "utmos"}
    assert data["steps"]["20000"]["summary_s1000"]["wer"] == 0.12
    assert "summary_s1000" not in data["steps"]["5000"]


def test_final_is_the_last_full_evaluation_and_seeds_pool_as_the_mean():
    steps = scores(state())["steps"]
    assert final(steps) == 20000
    assert final({"5000": steps["5000"]}) is None
    assert abs(pooled(steps["20000"], "wer") - 0.11) < 1e-12
    assert pooled(steps["5000"], "wer") == 0.9


def test_page_lists_every_run_and_links_into_its_folder():
    text = page(REPO, ["x-repa", "comparison"], {"x-repa": scores(state())}, "## header")
    assert "## header" in text
    assert f"(https://huggingface.co/{REPO}/tree/main/x-repa)" in text
    assert "| 20000 | 2 | 11.00 | 5.50 |" in text
    assert "90.0 / 45.0" in text  # the quick-set trajectory
    assert f"[`comparison`](https://huggingface.co/{REPO}/tree/main/comparison)" in text


def test_describe_leaves_out_the_shared_options():
    text = describe("x-swiglu")
    assert "ffn_activation=swiglu" in text and "cross_prompt_prob" not in text and "grad_checkpoint" not in text


def test_folder_readme_links_audio_inside_the_folder():
    text = readme("arm", REPO, state(), "", "x-repa")
    assert f"https://huggingface.co/{REPO}/tree/main/x-repa/eval/step-0020000/audio" in text
    assert "20000 (sampling seed 1000)" in text
