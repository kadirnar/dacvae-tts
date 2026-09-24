import copy
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pytest

from dacvae_tts.comparison import (
    cluster_bootstrap,
    compare_evaluations,
    detect_cluster,
    detect_key,
    length_bucket,
    markdown_report,
    verdict,
)


def freya_rows(speakers=8, per_speaker=12, seed=0, draw=None):
    """eval_sentences.py-style rows with a strong per-speaker difficulty, i.e. correlated errors in a cluster.

    `seed` fixes texts, lengths and speaker difficulties; `draw` re-samples only the scores, i.e. another
    system from the same distribution (no true difference)."""
    rng = np.random.default_rng(seed)
    noise = np.random.default_rng([seed, 0 if draw is None else draw + 1])
    rows = []
    for s in range(speakers):
        difficulty = rng.uniform(0.0, 0.3)
        for u in range(per_speaker):
            words = int(rng.integers(2, 15))
            edits = int(noise.binomial(words, difficulty))
            substitutions = edits // 2
            rows.append(
                {
                    "id": f"{s:02d}{u:02d}",
                    "text": " ".join(["kelime"] * words),
                    "speaker": f"spk{s}",
                    "prompt": f"prompt-{s:02d}.wav",
                    "register": "formal" if u % 2 else "casual",
                    "words": words,
                    "word_edits": edits,
                    "word_substitutions": substitutions,
                    "word_deletions": edits - substitutions,
                    "word_insertions": 0,
                    "chars": 6 * words,
                    "char_edits": 2 * edits,
                    "wer": edits / words,
                    "cer": 2 * edits / (6 * words),
                    "speaker_similarity": float(0.9 + noise.normal(0, 0.01)),
                    "dnsmos_ovrl": float(3.0 + noise.normal(0, 0.1)),
                    "clipped_fraction": 0.001,
                    "rtf": 0.3,
                }
            )
    return rows


def shifted(rows, edits=0, similarity=0.0, dnsmos=0.0):
    out = copy.deepcopy(rows)
    for row in out:
        row["word_edits"] += edits
        row["word_insertions"] += edits
        row["char_edits"] += edits
        row["wer"], row["cer"] = row["word_edits"] / row["words"], row["char_edits"] / row["chars"]
        row["speaker_similarity"] += similarity
        row["dnsmos_ovrl"] += dnsmos
    return out


def test_identical_systems_tie_with_zero_difference():
    rows = freya_rows()
    report = compare_evaluations({"a": rows, "b": copy.deepcopy(rows)}, samples=500)
    metrics = report["comparisons"]["b"]["metrics"]
    assert {"cer", "wer", "wer_mean", "speaker_similarity", "dnsmos_ovrl"} <= metrics.keys()
    for entry in metrics.values():
        assert entry["delta"] == 0 and entry["ci"][0] <= 0 <= entry["ci"][1] and entry["verdict"] == "tie"


def test_same_distribution_noise_is_a_tie():
    systems = {"a": freya_rows(), "b": freya_rows(draw=1)}  # independent scores, identical distribution
    metrics = compare_evaluations(systems, samples=2000)["comparisons"]["b"]["metrics"]
    for name in ("cer", "wer", "speaker_similarity", "dnsmos_ovrl"):
        low, high = metrics[name]["ci"]
        assert metrics[name]["delta"] != 0 and low < 0 < high and metrics[name]["verdict"] == "tie"


def test_known_shift_gives_the_right_sign_and_verdict():
    rows = freya_rows()
    worse = shifted(rows, edits=1, similarity=0.02, dnsmos=-0.3)
    report = compare_evaluations({"base": rows, "worse": worse}, samples=2000)
    metrics = report["comparisons"]["worse"]["metrics"]
    expected = len(rows) / sum(r["words"] for r in rows)  # one extra edit per utterance
    assert metrics["wer"]["delta"] == pytest.approx(expected)
    assert metrics["wer"]["ci"][0] > 0 and metrics["wer"]["verdict"] == "loss"
    assert metrics["cer"]["verdict"] == "loss"
    assert metrics["speaker_similarity"]["ci"][0] > 0 and metrics["speaker_similarity"]["verdict"] == "win"
    assert metrics["dnsmos_ovrl"]["ci"][1] < 0 and metrics["dnsmos_ovrl"]["verdict"] == "loss"
    assert metrics["rtf"]["verdict"] == "tie"
    # Swapping the baseline flips every sign.
    back = compare_evaluations({"worse": worse, "base": rows}, samples=2000)["comparisons"]["base"]["metrics"]
    assert back["wer"]["verdict"] == "win" and back["wer"]["delta"] == pytest.approx(-expected)


def test_verdict_is_direction_aware():
    assert verdict(-2.0, -0.1, "lower") == "win"
    assert verdict(-2.0, -0.1, "higher") == "loss"
    assert verdict(0.1, 2.0, "higher") == "win"
    assert verdict(-1.49, 0.0, "lower") == "tie"  # interval touching 0 is not resolved
    assert verdict(0.1, 0.2, None) == "higher"
    assert verdict(None, None, "lower") == "n/a"


def test_results_are_deterministic_under_a_seed():
    systems = {"a": freya_rows(), "b": shifted(freya_rows(), edits=1)}
    one = compare_evaluations(systems, samples=300, seed=3, stratify=["length"])
    two = compare_evaluations(copy.deepcopy(systems), samples=300, seed=3, stratify=["length"])
    other = compare_evaluations(systems, samples=300, seed=4)
    assert json.dumps(one) == json.dumps(two)
    assert one["systems"]["a"]["metrics"]["wer"]["ci"] != other["systems"]["a"]["metrics"]["wer"]["ci"]


def test_cluster_bootstrap_resamples_whole_clusters():
    # Cluster x: 4 edits / 8 words over two items; cluster y: 0 / 8. Two clusters drawn with replacement can
    # only give xx = 0.5, xy = 0.25 or yy = 0 (with probabilities 1/4, 1/2, 1/4), never an item-level mix.
    numerators, denominators = [2, 2, 0, 0], [4, 4, 3, 5]
    estimate, draws = cluster_bootstrap(numerators, denominators, ["x", "x", "y", "y"], samples=4000, seed=0)
    assert estimate[0] == pytest.approx(0.25)
    values, counts = np.unique(draws[:, 0], return_counts=True)
    assert values.tolist() == [0.0, 0.25, 0.5]
    assert counts / counts.sum() == pytest.approx([0.25, 0.5, 0.25], abs=0.03)
    # Utterance-level resampling mixes items and therefore produces other values.
    _, utterance = cluster_bootstrap(numerators, denominators, range(4), samples=4000, seed=0)
    assert len(np.unique(utterance[:, 0])) > 3
    # A mean is a ratio with denominator 1 (0/0 marks a missing value), columns share the resamples.
    estimate, draws = cluster_bootstrap(
        [[1.0, 2.0], [3.0, 0.0]], [[1, 1], [1, 0]], ["x", "y"], samples=200, seed=0
    )
    assert estimate.tolist() == [2.0, 2.0]
    assert set(np.unique(draws[:, 0])) <= {1.0, 2.0, 3.0}
    assert np.isnan(draws[draws[:, 0] == 3.0, 1]).all()  # only cluster y drawn: no value of the second metric
    with pytest.raises(ValueError):
        cluster_bootstrap([1.0], [1.0, 2.0], ["x"])


def test_cluster_bootstrap_matches_a_naive_loop():
    # Same picks as the engine (one chunk): a draw is the ratio of sums over the picked clusters' members.
    rng = np.random.default_rng(7)
    clusters = rng.choice(["a", "b", "c", "d", "e"], size=40)
    num = rng.integers(0, 5, size=(40, 3)).astype(float)
    den = rng.integers(1, 9, size=(40, 3)).astype(float)
    estimate, draws = cluster_bootstrap(num, den, clusters, samples=300, seed=11)
    names = list(dict.fromkeys(clusters))  # codes follow first appearance
    picks = np.random.default_rng(11).integers(0, len(names), size=(300, len(names)))
    for draw, row in zip(draws, picks):
        members = np.concatenate([np.flatnonzero(clusters == names[c]) for c in row])
        assert draw == pytest.approx(num[members].sum(0) / den[members].sum(0))
    assert estimate == pytest.approx(num.sum(0) / den.sum(0))


def test_cluster_bootstrap_is_fast_enough_for_freya_scale():
    rng = np.random.default_rng(0)
    words = rng.integers(2, 15, size=495).astype(float)
    edits = rng.binomial(words.astype(int), 0.05).astype(float)
    started = time.perf_counter()
    cluster_bootstrap(np.stack([edits] * 8, 1), np.stack([words] * 8, 1), np.arange(495) % 24, samples=5000)
    cluster_bootstrap(np.stack([edits] * 8, 1), np.stack([words] * 8, 1), range(495), samples=5000)
    assert time.perf_counter() - started < 5


def test_speaker_clusters_widen_the_interval():
    report = compare_evaluations({"a": freya_rows(speakers=12)}, samples=3000, utterance_ci=True)
    wer = report["systems"]["a"]["metrics"]["wer"]
    clustered, utterance = wer["ci"][1] - wer["ci"][0], wer["ci_utterance"][1] - wer["ci_utterance"][0]
    assert report["config"]["cluster"] == "speaker" and clustered > 1.2 * utterance


def test_missing_failed_and_extra_fields_are_tolerated():
    rows = freya_rows()
    for row in rows:
        row["utmos"], row["sim_o"] = 3.5, 0.6
    other = shifted(rows)
    for row in other:
        del row["dnsmos_ovrl"], row["utmos"]
        row["unknown_metric"], row["rtf"] = "ignored", None
    other[0] = {**{k: other[0][k] for k in ("id", "text", "speaker")}, "error": "boom"}
    del other[1]["speaker_similarity"]
    report = compare_evaluations({"a": rows, "b": other}, samples=200)
    a, b = report["systems"]["a"], report["systems"]["b"]
    assert {"dnsmos_ovrl", "utmos", "sim_o", "rtf"} <= a["metrics"].keys()
    assert not {"dnsmos_ovrl", "utmos", "rtf"} & b["metrics"].keys()
    assert b["failed"] == 1 and b["items"] == len(rows) - 1
    assert b["metrics"]["speaker_similarity"]["n"] == len(rows) - 2
    paired = report["comparisons"]["b"]
    assert paired["unpaired"] == {"baseline": 1, "system": 0}
    assert paired["metrics"]["speaker_similarity"]["n"] == len(rows) - 2
    assert "dnsmos_ovrl" not in paired["metrics"] and "sim_o" in paired["metrics"]
    assert any("no scored partner" in note for note in report["notes"])
    text = markdown_report(report)
    assert "UTMOS" in text and "| `b` |" in text
    # clip_fraction (eval_sentences full band) is only used when no clipped_fraction exists anywhere.
    legacy = [{k: v for k, v in r.items() if k != "clipped_fraction"} | {"clip_fraction": 0.5} for r in rows]
    report = compare_evaluations({"a": legacy}, samples=50)
    assert report["systems"]["a"]["metrics"]["clipped_fraction"]["value"] == 0.5
    assert report["config"]["aliases"]["clipped_fraction"] == "clip_fraction"


def test_seed_replicates_are_averaged_per_utterance():
    seed_a, seed_b = freya_rows(), freya_rows(draw=5)
    report = compare_evaluations({"sys": [seed_a, seed_b]}, samples=200)
    summary = report["systems"]["sys"]
    per_seed = [sum(r["word_edits"] for r in s) / sum(r["words"] for r in s) for s in (seed_a, seed_b)]
    assert summary["replicates"] == 2 and summary["items"] == len(seed_a)
    assert summary["rows"] == 2 * len(seed_a)
    assert summary["metrics"]["wer"]["value"] == pytest.approx(np.mean(per_seed))
    # The same replicates in one file, told apart by a seed field.
    flat = [dict(r, seed=1) for r in seed_a] + [dict(r, seed=2) for r in seed_b]
    same = compare_evaluations({"sys": flat}, samples=200)["systems"]["sys"]
    assert same["metrics"]["wer"]["value"] == pytest.approx(np.mean(per_seed))
    with pytest.raises(ValueError, match="duplicate"):
        compare_evaluations({"sys": seed_a + seed_b}, samples=10)


def test_key_and_cluster_detection():
    evaluate_rows = [
        {"uid": "u1", "reference_uid": "r1", "speaker": "s1", "seed": 1, "text": "a"},
        {"uid": "u2", "reference_uid": "r2", "speaker": "s2", "seed": 1, "text": "b"},
    ]
    assert detect_key(evaluate_rows) == ("uid", "reference_uid")
    assert detect_cluster(evaluate_rows) == "speaker"
    monitor_rows = [{"uid": f"u{i}", "prompt_uid": f"p{i}", "text": "a"} for i in range(2)]
    assert detect_key(monitor_rows) == ("uid", "prompt_uid") and detect_cluster(monitor_rows) == "prompt_uid"
    freya = [{"id": "0", "prompt": "prompt-00.wav"}, {"id": "1", "prompt": "prompt-01.wav"}]
    assert detect_key(freya) == ("id",) and detect_cluster(freya) == "prompt"
    assert detect_cluster([{"id": "0", "speaker": "s"}, {"id": "1", "speaker": "s"}]) is None
    assert detect_key([{"text": "a"}]) == ("text",)
    with pytest.raises(ValueError, match="pairing key"):
        detect_key([{"checkpoint": "step-1.pt", "wer": 0.1}])  # monitor.jsonl summaries, not utterances
    report = compare_evaluations({"a": freya_rows()}, cluster="none", samples=50)
    assert report["config"]["cluster"] is None and report["systems"]["a"]["clusters"] == len(freya_rows())
    assert any("Utterance-level" in note for note in report["notes"])
    report = compare_evaluations({"a": freya_rows()}, cluster="prompt", key=["text", "id"], samples=50)
    assert report["config"]["key"] == ["text", "id"] and report["config"]["cluster"] == "prompt"


def test_mismatched_pairs_are_rejected():
    rows = freya_rows()
    revoiced = copy.deepcopy(rows)
    for row in revoiced:
        row["speaker"] = "other-" + row["speaker"]
    with pytest.raises(ValueError, match="different clusters"):
        compare_evaluations({"a": rows, "b": revoiced}, samples=10)
    # Clustering on prompt files that keep their names across voice draws pairs other voices: flagged.
    report = compare_evaluations({"a": rows, "b": revoiced}, cluster="prompt", samples=10)
    assert any(f"{len(rows)} paired utterances have another `speaker`" in note for note in report["notes"])
    retexted = copy.deepcopy(rows)
    retexted[3]["text"] = "başka"
    with pytest.raises(ValueError, match="text changed"):
        compare_evaluations({"a": rows, "b": retexted}, samples=10)
    with pytest.raises(ValueError, match="Unknown metrics"):
        compare_evaluations({"a": rows}, metrics=["wer", "mos"], samples=10)
    with pytest.raises(ValueError, match="Baseline"):
        compare_evaluations({"a": rows}, baseline="z", samples=10)
    with pytest.raises(ValueError, match="no successful"):
        compare_evaluations({"a": rows, "b": [{"id": "0", "error": "x"}]}, samples=10)


def test_systems_scored_differently_are_refused():
    rows = freya_rows()
    identity = {"asr_backend": "faster-whisper", "asr_model": "large-v3", "language": "tr",
                "metric_normalization": "turkish-v1", "decoding": {"beam_size": 5}, "compute_type": "float16",
                "device": "cuda", "speaker_model": "microsoft/wavlm-base-plus-sv", "protocol_options": None}

    def scored(rows, **changes):
        out = copy.deepcopy(rows)
        for row in out:
            row["evaluator"] = {**identity, **changes}
        return out

    same = compare_evaluations({"a": scored(rows), "b": scored(shifted(rows, edits=1))}, samples=10)
    assert not any("scor" in note for note in same["notes"])
    # A full Evaluator identity (versions, hashes, the protocol's details) compares by its compact fields.
    full = scored(rows)
    for row in full:
        row["evaluator"].update(faster_whisper_version="1.1.0", dnsmos_sha256="abc")
    compare_evaluations({"a": scored(rows), "b": full}, samples=10)
    band_limited = scored(rows, protocol_options={"band_limit_8k": True})
    with pytest.raises(ValueError, match="scored differently.*protocol_options"):
        compare_evaluations({"a": scored(rows), "b": band_limited}, samples=10)
    with pytest.raises(ValueError, match='metric_normalization: a="turkish-v1", b="turkish-v2"'):
        compare_evaluations({"a": scored(rows), "b": scored(rows, metric_normalization="turkish-v2")}, samples=10)
    with pytest.raises(ValueError, match="compute_type"):  # CPU int8 vs CUDA float16 Whisper
        compare_evaluations({"a": scored(rows), "b": scored(rows, compute_type="int8", device="cpu")}, samples=10)
    mixed = scored(rows)
    mixed[0]["evaluator"]["device"] = "cpu"  # replicates or rescored rows within one system disagree
    with pytest.raises(ValueError, match="device"):
        compare_evaluations({"a": scored(rows), "b": mixed}, samples=10)
    allowed = compare_evaluations({"a": scored(rows), "b": band_limited}, samples=10, allow_scorer_mismatch=True)
    assert any("Scorer mismatch allowed" in note for note in allowed["notes"])
    unknown = compare_evaluations({"a": scored(rows), "b": rows}, samples=10)  # results written before identities
    assert any("`b` carry no scorer identity" in note for note in unknown["notes"])


def test_stratified_reports():
    assert [length_bucket({"words": n}) for n in (1, 5, 6, 9, 10, 30)] == [
        "1-5 words", "1-5 words", "6-9 words", "6-9 words", "10+ words", "10+ words"
    ]
    assert length_bucket({"text": "bir iki üç"}) == "1-5 words"
    rows = freya_rows()
    report = compare_evaluations(
        {"a": rows, "b": shifted(rows, edits=1)}, stratify=["length", "register"], samples=300
    )
    length = report["strata"]["length"]
    assert list(length) == [b for b in ("1-5 words", "6-9 words", "10+ words") if b in length]
    assert sum(entry["systems"]["a"]["items"] for entry in length.values()) == len(rows)
    for entry in length.values():
        assert set(entry["systems"]["a"]["metrics"]) == {"cer", "wer"}
        assert entry["comparisons"]["b"]["metrics"]["wer"]["delta"] > 0
    assert list(report["strata"]["register"]) == ["casual", "formal"]
    text = markdown_report(report)
    assert "### By `length`" in text and "### By `register`" in text and "Δ WER % vs `a`" in text


def test_markdown_report_tables():
    rows = freya_rows()
    report = compare_evaluations(
        {"base": rows, "worse": shifted(rows, edits=1, similarity=0.02)}, samples=500, utterance_ci=True
    )
    json.dumps(report, allow_nan=False)  # strict JSON: no NaN/inf, no numpy scalars
    text = markdown_report(report)
    lines = text.splitlines()
    assert "Bootstrap over `speaker` clusters: B = 500, seed 0, 95% percentile intervals" in lines[0]
    header = next(line for line in lines if line.startswith("| system | n |"))
    columns = ("CER %", "WER %", "WER % (utt. mean)", "S/D/I %", "SIM", "DNSMOS OVRL", "clipped %", "RTF")
    assert all(f"| {column} |" in header for column in columns)
    assert "WER % utt.-level CI" in header and "UTMOS" not in header and "seeds" not in header
    wer = report["systems"]["base"]["metrics"]["wer"]
    cell = f"{100 * wer['value']:.2f} [{100 * wer['ci'][0]:.2f}, {100 * wer['ci'][1]:.2f}]"
    assert cell in text
    assert "### Paired differences vs `base`" in text
    paired = [line for line in lines if line.startswith("| `worse` |")]
    assert any("| WER % |" in line and "**loss**" in line for line in paired)
    assert any("| SIM |" in line and "**win**" in line for line in paired)
    assert any("| RTF |" in line and "| tie |" in line for line in paired)
    # Every table row has as many cells as its header.
    for block in text.split("\n\n"):
        table = [line for line in block.splitlines() if line.startswith("|")]
        if table:
            assert len({line.count(" | ") for line in table if not line.startswith("| ---")}) == 1
    assert "Notes:" in text and "Only 8 `speaker` clusters" in text


def test_compare_evals_script(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location(
        "compare_evals", Path(__file__).resolve().parents[1] / "scripts" / "compare_evals.py"
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    rows = freya_rows()
    for name, data in (("base", rows), ("new", shifted(rows, edits=1))):
        (tmp_path / name).mkdir()
        (tmp_path / name / "results.jsonl").write_text("\n".join(json.dumps(r) for r in data) + "\n")
    (tmp_path / "seed2.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    script.main(
        [
            str(tmp_path / "base"),
            f"new={tmp_path / 'new'},{tmp_path / 'seed2.jsonl'}",
            "--bootstrap", "300", "--stratify", "length",
            "--markdown", str(tmp_path / "out" / "report.md"),
            "--output", str(tmp_path / "out" / "report.json"),
        ]
    )
    printed = capsys.readouterr().out
    assert printed == (tmp_path / "out" / "report.md").read_text()
    assert "| `base` |" in printed and "| `new` |" in printed and "seeds" in printed
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert report["config"]["baseline"] == "base" and len(report["config"]["sources"]["new"]) == 2
    assert report["systems"]["new"]["replicates"] == 2
    for name, normalization in (("v1", "turkish-v1"), ("v2", "turkish-v2")):
        (tmp_path / name).mkdir()
        lines = [json.dumps({**r, "evaluator": {"metric_normalization": normalization}}) for r in rows]
        (tmp_path / name / "results.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(SystemExit, match="scored differently"):
        script.main([str(tmp_path / "v1"), str(tmp_path / "v2"), "--bootstrap", "10"])
    script.main([str(tmp_path / "v1"), str(tmp_path / "v2"), "--bootstrap", "10", "--allow-scorer-mismatch"])
    assert "Scorer mismatch allowed" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        script.main([str(tmp_path / "missing")])
    with pytest.raises(SystemExit):
        script.main([str(tmp_path / "base"), "--baseline", "nope", "--bootstrap", "10"])
