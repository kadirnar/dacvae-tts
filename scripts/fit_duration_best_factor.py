"""Refit the duration predictor on the duration factor that decodes best (DMOSpeech 2's idea without RL).

DMOSpeech 2 (arXiv 2507.14988, Seed-TTS-eval en): ground-truth durations give 1.821 WER, the F5 rate rule 2.028, a
predictor supervised on corpus durations 3.750 (worse than the rule) and the same predictor optimized for WER/SIM
(GRPO) 1.752. The target of a duration model is the length the TTS model speaks a text best at, not the length the
corpus speaker happened to use. Without RL: same-speaker (prompt, text) pairs of a latent cache are synthesized at K
duration factors around a base mode (one batch per pair: Synthesizer.synthesize_many with duration_factors), every
candidate is scored by a pluggable scorer (CER of a Whisper transcript, optionally minus sim_weight x WavLM speaker
similarity), the best factor per pair is kept (mean over --seeds noises; ties go to the factor closest to 1) and the
ridge predictor (duration.fit, same features) is refitted on target frames = base frames x best factor.

The output loads with DurationPredictor.load: Synthesizer(..., duration_model=PATH) with --duration-mode predictor
(or auto, whose slow prompts use the predictor). Per-pair records are appended to a JSONL, so an interrupted run
resumes where it stopped and --refit-only refits from the records on a CPU.

  python scripts/fit_duration_best_factor.py --checkpoint runs/tr-nano/step-0060000.pt --cache data/tr55/merged \
      --output outputs/duration_best_factor.json --pairs 600 --factors 1.0,0.85,0.92,1.08,1.15 --base rule
"""

import argparse
import json
import math
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from dacvae_tts.duration import FEATURES, DurationPredictor, fit, speaking_rate


def parse_factors(text):
    """'1.0,0.9,1.1' -> [1.0, 0.9, 1.1]; finite, positive and distinct."""
    factors = [float(value) for value in str(text).split(",") if value.strip()]
    if not factors or len(set(factors)) != len(factors) or not all(math.isfinite(f) and f > 0 for f in factors):
        raise ValueError(f"Need distinct finite positive duration factors, got {text!r}")
    return factors


def sample_pairs(data, split, count, seed, min_prompt=50, max_prompt=375):
    """Same-speaker (prompt recording, other recording's text) pairs of `split`, round-robin over shuffled speakers.

    `data` is a LatentDataset of the split (its row order gives the latent index of each prompt). A target text is
    used at most once; prompts are 2-15 s like the evaluation prompts.
    """
    db = sqlite3.connect(f"file:{data.db_path}?mode=ro", uri=True)
    rows = db.execute("SELECT id, uid, speaker, text, frames FROM samples WHERE split=? ORDER BY speaker, id",
                      (split,)).fetchall()
    index = {int(v): i for i, v in enumerate(data.ids)}
    by_speaker = defaultdict(list)
    for row in rows:
        by_speaker[row[2]].append(row)
    rng = random.Random(seed)
    speakers = sorted(s for s, items in by_speaker.items()
                      if len(items) >= 2 and any(min_prompt <= r[4] <= max_prompt for r in items))
    rng.shuffle(speakers)
    used, pairs, turn = set(), [], 0
    while len(pairs) < count and speakers:
        speaker = speakers[turn % len(speakers)]
        prompts = [r for r in by_speaker[speaker] if min_prompt <= r[4] <= max_prompt]
        free = [r for r in by_speaker[speaker] if r[1] not in used and any(p[1] != r[1] for p in prompts)]
        if not free:
            speakers.remove(speaker)  # every usable target of this speaker is taken
            continue
        target = rng.choice(free)
        prompt = rng.choice([r for r in prompts if r[1] != target[1]])
        used.add(target[1])
        pairs.append({"pair": len(pairs), "speaker": speaker, "prompt_uid": prompt[1], "prompt_index": index[int(prompt[0])],
                      "prompt_frames": int(prompt[4]), "prompt_text": prompt[3], "uid": target[1], "text": target[3],
                      "frames": int(target[4])})
        turn += 1
    return pairs


def search_pair(tts, voice, text, factors, scorer, base="rule", seeds=1, seed=42, **synthesis):
    """Synthesize `text` at every factor (x `seeds` noises) in one batch, score it and pick the best factor."""
    results, _ = tts.synthesize_many([text], voice, candidates=len(factors) * seeds, duration_factors=factors,
                                     duration_mode=base, seed=seed, **synthesis)
    candidates = results[0]
    prompt = None
    if getattr(scorer, "needs_prompt", False):  # the prompt's waveform, for speaker similarity
        import torch

        with torch.inference_mode():
            prompt = tts.codec.decode(voice.latents.to(tts.device).float() * tts.std + tts.mean).float().cpu().numpy()
    scores = scorer(text, [c["audio"] for c in candidates], tts.codec.sample_rate, prompt)
    per_factor, frames = defaultdict(list), {}
    for candidate, score in zip(candidates, scores):
        per_factor[candidate["duration_factor"]].append(score["score"])
        frames[candidate["duration_factor"]] = int(candidate["frames"])
    means = {f: float(np.mean(v)) for f, v in per_factor.items()}
    best = min(means, key=lambda f: (means[f], abs(math.log(f)), f))
    return {
        "best_factor": best,
        "target_frames": frames[best],
        "base_frames": frames.get(1.0),
        "factor_frames": {str(f): frames[f] for f in factors},
        "factor_scores": {str(f): means[f] for f in factors},
        "candidates": [{"factor": c["duration_factor"], **s} for c, s in zip(candidates, scores)],
    }


def read_records(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def search(tts, pairs, load_latents, scorer, factors, base="rule", records_path=None, seeds=1, seed=42, **synthesis):
    """search_pair over `pairs`; each record is appended to `records_path` at once and existing ones are reused."""
    from dacvae_tts.inference import VoiceReference
    from dacvae_tts.text import normalize

    records = read_records(records_path) if records_path else []
    for record in records:
        if "error" not in record and (record["base"] != base or set(record["factor_frames"]) != {str(f) for f in factors}):
            raise ValueError(f"{records_path} was searched with another base mode or factor set; use a new --records")
    done = {r["pair"] for r in records}
    for pair in pairs:
        if pair["pair"] in done:
            continue
        # Normalized as Synthesizer.target_frames normalizes them, so the refit sees what inference will predict on.
        pair = {**pair, "prompt_text": normalize(pair["prompt_text"], tts.text_version),
                "text": normalize(pair["text"], tts.text_version)}
        voice = VoiceReference(load_latents(pair), pair["prompt_text"], "cache", {})
        try:
            record = {**pair, "prompt_frames": len(voice.latents), "base": base,
                      **search_pair(tts, voice, pair["text"], factors, scorer, base, seeds, seed + pair["pair"], **synthesis)}
        except ValueError as error:  # target outside .25-30 s at some factor, empty metric text, ...
            record = {**pair, "error": str(error)}
        records.append(record)
        if records_path:
            with open(records_path, "a") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if len(records) % 25 == 0:
            print(f"{len(records)}/{len(pairs)} pairs", flush=True)
    return records


def _mae_log(predicted, truth):
    return float(np.mean(np.abs(np.log(np.asarray(predicted, float) / np.asarray(truth, float))))) if truth else None


def refit(records, ridge=1e-3, holdout=0.0):
    """Ridge refit on (prompt frames, prompt text, text, best-factor frames); returns (predictor, report)."""
    good = sorted((r for r in records if "error" not in r), key=lambda r: r["pair"])
    if not good:
        raise ValueError("No scored pairs to fit")
    held = good[: int(len(good) * holdout)]
    train = good[len(held):]
    predictor = fit([(r["prompt_frames"], r["prompt_text"], r["text"], r["target_frames"]) for r in train], ridge)
    factors = np.array([r["best_factor"] for r in good])
    rates = np.array([speaking_rate(r["prompt_frames"], r["prompt_text"]) for r in good])
    report = {
        "pairs": len(good), "failed": len(records) - len(good), "fit_pairs": len(train), "holdout_pairs": len(held),
        "best_factor_counts": {str(f): int((factors == f).sum()) for f in sorted(set(factors.tolist()))},
        "mean_log_best_factor": float(np.mean(np.log(factors))),
        "changed_fraction": float(np.mean(factors != 1.0)),
    }
    for label, mask in (("fast_prompts", rates > 17), ("normal_prompts", (rates >= 13) & (rates <= 17)),
                        ("slow_prompts", rates < 13)):
        if mask.any():
            report[f"{label}_mean_log_best_factor"] = float(np.mean(np.log(factors[mask])))
            report[f"{label}_pairs"] = int(mask.sum())
    for name, rows in (("fit", train), ("holdout", held)):
        with_base = [r for r in rows if r.get("base_frames")]
        if with_base:  # how far the base mode and the refit are from the best-factor lengths
            truth = [r["target_frames"] for r in with_base]
            report[f"{name}_base_mae_log"] = _mae_log([r["base_frames"] for r in with_base], truth)
            report[f"{name}_refit_mae_log"] = _mae_log(
                [predictor.predict(r["prompt_frames"], r["prompt_text"], r["text"]) for r in with_base], truth)
    predictor.metadata.update(report, weights_by_feature=dict(zip(FEATURES, predictor.weights)), frame_rate=25,
                              note="log(best-factor frames) ridge regression; scripts/fit_duration_best_factor.py")
    return predictor, report


class WhisperScorer:
    """score = CER of a Whisper transcript (lower is better), minus sim_weight x WavLM speaker similarity."""

    def __init__(self, model, device, language, speaker_model=None, sim_weight=0.0, normalization="turkish-v1"):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from eval_sentences import HFWhisper

        self.whisper = HFWhisper(model, device, language)
        self.normalization, self.sim_weight = normalization, sim_weight
        self.needs_prompt = bool(speaker_model) and sim_weight > 0
        if self.needs_prompt:
            import torch
            from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

            from dacvae_tts.metrics import Evaluator

            self.evaluator = Evaluator.__new__(Evaluator)  # only the speaker model (as in eval_sentences.score_hf)
            self.evaluator.device = torch.device(device)
            self.evaluator.extractor = AutoFeatureExtractor.from_pretrained(speaker_model)
            self.evaluator.speaker = AutoModelForAudioXVector.from_pretrained(speaker_model).to(device).eval()

    def _embedding(self, audio, sample_rate):
        import torch
        from scipy.signal import resample_poly

        clip = resample_poly(audio, 16, sample_rate // 1000).astype(np.float32)
        with torch.inference_mode():
            return self.evaluator.embedding(torch.from_numpy(clip))

    def __call__(self, text, audios, sample_rate, prompt=None):
        from dacvae_tts.metrics import error_counts

        hypotheses = self.whisper.transcribe(audios, sample_rate)
        reference = self._embedding(prompt, sample_rate) if self.needs_prompt else None
        scores = []
        for audio, hypothesis in zip(audios, hypotheses):
            counts = error_counts(text, hypothesis, self.normalization)
            score = {"cer": counts["cer"], "wer": counts["wer"], "hypothesis": hypothesis, "score": counts["cer"]}
            if reference is not None:
                score["similarity"] = float((self._embedding(audio, sample_rate) * reference).sum())
                score["score"] = counts["cer"] - self.sim_weight * score["similarity"]
            scores.append(score)
        return scores


def main():
    from dacvae_tts.inference import DURATION_MODES

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint")
    parser.add_argument("--cache")
    parser.add_argument("--output", required=True, help="Predictor JSON (DurationPredictor.load format)")
    parser.add_argument("--records", help="Per-pair JSONL (default: <output>.records.jsonl); reused to resume")
    parser.add_argument("--refit-only", action="store_true", help="Only refit from --records (no synthesis, CPU)")
    parser.add_argument("--split", default="train", help="Cache split the pairs come from")
    parser.add_argument("--pairs", type=int, default=600)
    parser.add_argument("--factors", default="1.0,0.85,0.92,1.08,1.15",
                        help="Duration factors searched per pair (1.0 first: the base length)")
    parser.add_argument("--seeds", type=int, default=1, help="Noises per factor; the factor's score is their mean")
    parser.add_argument("--base", choices=DURATION_MODES, default="rule", help="Duration mode the factors scale")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--sway", type=float, default=-1.0)
    parser.add_argument("--max-rows", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scorer-model", default="openai/whisper-large-v3-turbo", help="HF Whisper of the CER scorer")
    parser.add_argument("--language", default="tr")
    parser.add_argument("--speaker-model", default="microsoft/wavlm-base-plus-sv")
    parser.add_argument("--sim-weight", type=float, default=0.0,
                        help="> 0 adds speaker similarity to the score: CER - weight x SIM")
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--holdout", type=float, default=0.1, help="Fraction of pairs kept out of the fit (report)")
    args = parser.parse_args()
    factors = parse_factors(args.factors)
    records_path = Path(args.records) if args.records else Path(args.output).with_suffix(".records.jsonl")
    if args.refit_only:
        records = read_records(records_path)
    else:
        if not args.checkpoint or not args.cache:
            parser.error("--checkpoint and --cache are required unless --refit-only")
        from dacvae_tts.data import LatentDataset
        from dacvae_tts.inference import Synthesizer

        tts = Synthesizer(args.checkpoint, device=args.device)
        data = LatentDataset(args.cache, args.split, pairing="within", layout="joined")
        pairs = sample_pairs(data, args.split, args.pairs, args.seed)
        scorer = WhisperScorer(args.scorer_model, args.device, args.language, args.speaker_model, args.sim_weight)
        records_path.parent.mkdir(parents=True, exist_ok=True)
        records = search(tts, pairs, lambda pair: data.row(pair["prompt_index"])["latents"], scorer, factors,
                         args.base, records_path, seeds=args.seeds, seed=args.seed, steps=args.steps,
                         guidance=args.guidance, sway=args.sway, max_rows=args.max_rows)
    predictor, report = refit(records, args.ridge, args.holdout)
    predictor.metadata.update(checkpoint=args.checkpoint, cache=args.cache, split=args.split, base=args.base,
                              factors=factors, seeds=args.seeds, steps=args.steps, guidance=args.guidance,
                              scorer=args.scorer_model, sim_weight=args.sim_weight, records=str(records_path))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    predictor.save(args.output)
    assert DurationPredictor.load(args.output).weights == predictor.weights
    assert all(math.isfinite(w) for w in predictor.weights)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
