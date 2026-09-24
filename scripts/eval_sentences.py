"""Synthesize a fixed sentence list (e.g. Freya-TR-Eval) with held-out prompts and score WER/CER/SIM/DNSMOS.

Prompts are cross-utterance validation cases from the latent cache (unseen speakers, as in monitor.py), used in
rotation over the sentences; each sentence is therefore unseen text on an unseen voice. Whisper large-v3 + the
Turkish metric normalization score intelligibility; DNSMOS OVRL scores audio quality (16 kHz proxy).

  python scripts/eval_sentences.py --checkpoint runs/tr-nano/step-0060000.pt --cache data/tr55/merged \
      --sentences data/eval/freya_tr_eval.jsonl --output outputs/freya-nano --prompts 24 --guidance 3 --steps 32
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from monitor import select_cases  # noqa: E402

from dacvae_tts.inference import Synthesizer, VoiceReference  # noqa: E402
from dacvae_tts.metrics import Evaluator, summarize  # noqa: E402


def load_sentences(path, limit=0):
    path = Path(path)
    rows = []
    if path.suffix == ".jsonl":
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows.append({"id": str(row.get("id", len(rows))), "text": row["text"]})
    else:
        for i, line in enumerate(path.read_text().splitlines()):
            if line.strip():
                rows.append({"id": str(i), "text": line.strip()})
    return rows[:limit] if limit else rows


class HFWhisper:
    """Batched transformers Whisper (fp16) used to rank best-of-N candidates."""

    def __init__(self, name, device, language):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.processor = WhisperProcessor.from_pretrained(name)
        self.model = WhisperForConditionalGeneration.from_pretrained(name, dtype=torch.float16).to(device).eval()
        self.device, self.language = device, language

    @torch.inference_mode()
    def transcribe(self, audios, sample_rate):
        from scipy.signal import resample_poly

        clips = [resample_poly(a, 16000 // 1000, sample_rate // 1000).astype(np.float32) for a in audios]
        features = self.processor(clips, sampling_rate=16000, return_tensors="pt").input_features
        ids = self.model.generate(features.to(self.device, torch.float16), language=self.language, task="transcribe",
                                  num_beams=1, max_new_tokens=220)
        return [t.strip() for t in self.processor.batch_decode(ids, skip_special_tokens=True)]


def best_of_n(tts, selector, text, voice, path, seconds, index, args, sampler):
    """Synthesize N candidates in one batch; keep the lowest selector CER (ties: lowest WER, then first)."""
    from dacvae_tts.metrics import error_counts

    results, metadata = tts.synthesize_many(
        [text], voice, candidates=args.candidates, seconds=seconds, duration_scale=args.duration_scale,
        duration_mode=args.duration_mode, steps=args.steps, guidance=args.guidance, seed=args.seed + index,
        sway=args.sway, guidance_until=args.guidance_until, noise_scale=args.noise_scale,
        duration_factors=args.duration_factors, **sampler,
    )
    candidates = results[0]
    hypotheses = selector.transcribe([c["audio"] for c in candidates], tts.codec.sample_rate)
    scores = [error_counts(text, h, "turkish-v1") for h in hypotheses]
    best = min(range(len(candidates)), key=lambda k: (scores[k]["cer"], scores[k]["wer"], k))
    audio = candidates[best]["audio"]
    sf.write(path, audio, tts.codec.sample_rate, subtype="FLOAT")
    meta = {**metadata, "selected": best, "candidate_cer": [s["cer"] for s in scores], "candidate_hypotheses": hypotheses,
            "audio_seconds": len(audio) / tts.codec.sample_rate}
    factor = {"selected_factor": candidates[best]["duration_factor"]} if args.duration_factors else {}
    meta.update(factor)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return {"audio_seconds": meta["audio_seconds"], "rtf": metadata["request_seconds"] / max(meta["audio_seconds"], 1e-6),
            "selected": best, "candidate_cer": meta["candidate_cer"], **factor}


def audio_stats(path):
    """Full-band level statistics of one synthesis: samples at the decoder's tanh ceiling and integrated loudness."""
    import pyloudnorm

    audio, rate = sf.read(str(path), dtype="float64", always_2d=True)
    audio = audio.mean(1)
    loudness = pyloudnorm.Meter(rate).integrated_loudness(audio) if len(audio) >= 0.4 * rate else float("nan")
    return {"peak": float(np.abs(audio).max()), "clip_fraction": float(np.mean(np.abs(audio) >= 0.999)),
            "lufs": float(loudness) if np.isfinite(loudness) else -70.0}


def score_hf(rows, out, args):
    """Batched GPU scoring: transformers Whisper (greedy) for WER/CER, WavLM-SV similarity and DNSMOS per clip."""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    from dacvae_tts.codec import read_audio
    from dacvae_tts.metrics import DNSMOS, error_counts

    device = args.device
    name = args.asr_model if "/" in args.asr_model else f"openai/whisper-{args.asr_model}"
    processor = WhisperProcessor.from_pretrained(name)
    whisper = WhisperForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float16).to(device).eval()
    evaluator = Evaluator.__new__(Evaluator)  # only the speaker model, on the GPU
    evaluator.device = torch.device(device)
    from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

    evaluator.extractor = AutoFeatureExtractor.from_pretrained(args.speaker_model)
    evaluator.speaker = AutoModelForAudioXVector.from_pretrained(args.speaker_model).to(device).eval()
    dnsmos = DNSMOS(args.dnsmos) if args.dnsmos else None
    good = [r for r in rows if "error" not in r]
    audios = {r["audio"]: read_audio(r["audio"], 16000) for r in good}
    hypotheses = {}
    for start in range(0, len(good), args.asr_batch):
        chunk = good[start : start + args.asr_batch]
        features = processor([audios[r["audio"]].numpy() for r in chunk], sampling_rate=16000, return_tensors="pt").input_features
        with torch.inference_mode():
            ids = whisper.generate(features.to(device, torch.float16), language=args.language, task="transcribe", num_beams=1, max_new_tokens=220)
        for r, text in zip(chunk, processor.batch_decode(ids, skip_special_tokens=True)):
            hypotheses[r["audio"]] = text.strip()
    prompt_embeddings = {}
    scored = []
    for r in rows:
        if "error" in r:
            scored.append(r)
            continue
        audio = audios[r["audio"]]
        try:
            counts = error_counts(r["text"], hypotheses[r["audio"]], "turkish-v1")
        except ValueError as error:
            scored.append({**r, "error": str(error)})
            continue
        if r["prompt"] not in prompt_embeddings:
            prompt_embeddings[r["prompt"]] = evaluator.embedding(read_audio(out / r["prompt"], 16000))
        similarity = float((evaluator.embedding(audio) * prompt_embeddings[r["prompt"]]).sum())
        record = {**r, **counts, "hypothesis": hypotheses[r["audio"]], "speaker_similarity": similarity, "asr_backend": "hf-greedy"}
        if dnsmos is not None:
            record.update(dnsmos(audio.numpy()))
        scored.append(record)
    return scored


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--sentences", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--prompts", type=int, default=24, help="Number of held-out prompt voices used in rotation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--guidance-until", type=float, default=1.0)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--sway", type=float, default=-1.0)
    parser.add_argument("--duration-scale", type=float, default=1.0)
    parser.add_argument("--chars-per-second", type=float, default=0.0,
                        help="If > 0: fixed speaking rate; output seconds = normalized characters / rate instead of the prompt-rate rule")
    parser.add_argument("--duration-mode", choices=["rule", "clamp", "syllable", "predictor", "auto", "articulation"],
                        default="rule")
    parser.add_argument("--articulation-options", type=json.loads,
                        help='JSON overrides of the articulation rule, e.g. \'{"comma_pause": 0.2, "stop_pause": 0.4}\'')
    parser.add_argument("--duration-factors", type=lambda v: [float(f) for f in v.split(",")],
                        help="Duration-diverse best-of-N, e.g. 1.0,0.9,1.1: candidate k gets factor k mod n (needs "
                        "--candidates >= n; list 1.0 first, selector ties keep the first candidate)")
    parser.add_argument("--duration-model", help="Predictor JSON of the predictor/auto modes (default: the packaged "
                        "duration_tr.json), e.g. the output of scripts/fit_duration_best_factor.py")
    parser.add_argument("--guidance-from", type=float, default=0.0)
    parser.add_argument("--cfg-rescale", type=float, default=0.0)
    parser.add_argument("--apg-eta", type=float, default=1.0)
    parser.add_argument("--apg-norm", type=float, default=0.0)
    parser.add_argument("--apg-momentum", type=float, default=0.0)
    parser.add_argument("--speaker-guidance", type=float, default=None)
    parser.add_argument("--candidates", type=int, default=1,
                        help="Best-of-N: generate N candidates per sentence and keep the one the selector ASR transcribes best")
    parser.add_argument("--selector", default="openai/whisper-large-v3-turbo", help="HF Whisper used for best-of-N selection")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--language", default="tr")
    parser.add_argument("--asr-model", default="large-v3")
    parser.add_argument("--asr-device", default="cuda")
    parser.add_argument("--dnsmos")
    parser.add_argument("--speaker-model", default="microsoft/wavlm-base-plus-sv")
    parser.add_argument(
        "--asr-backend", choices=["faster-whisper", "hf"], default="faster-whisper",
        help="hf = transformers Whisper on the GPU with cross-clip batching (~30x faster than CPU faster-whisper; greedy decoding)",
    )
    parser.add_argument("--asr-batch", type=int, default=24)
    parser.add_argument("--rescore", action="store_true", help="Skip synthesis when the WAVs already exist; only score")
    args = parser.parse_args()
    if args.duration_factors and (args.candidates < 2 or len(args.duration_factors) > args.candidates):
        parser.error("--duration-factors is a best-of-N option: needs --candidates >= max(2, number of factors)")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    sentences = load_sentences(args.sentences, args.limit)
    data, cases = select_cases(args.cache, args.prompts, args.seed)
    wavs_exist = all((out / f"{s['id']}.wav").exists() for s in sentences)
    reuse = args.rescore and ((out / "results.jsonl").exists() or wavs_exist)
    tts = None if reuse else Synthesizer(args.checkpoint, device=args.device)
    if tts is not None:
        tts.articulation_options = args.articulation_options  # None: the defaults of duration.articulation_seconds
        if args.duration_model:
            tts.duration_model = args.duration_model  # loaded by the predictor/auto modes
    prompts = []
    for number, case in enumerate(cases):
        latents = data.row(case["prompt_index"])["latents"]
        wav = out / f"prompt-{number:02d}.wav"
        if not wav.exists() and tts is not None:
            sf.write(wav, tts.codec.decode(latents.to(tts.device) * tts.std + tts.mean).numpy(), tts.codec.sample_rate)
        prompts.append((VoiceReference(latents, case["prompt_text"], "cache", {}), wav, case["speaker"]))
    rows = []
    started = time.time()
    sampler = dict(guidance_from=args.guidance_from, cfg_rescale=args.cfg_rescale, apg_eta=args.apg_eta,
                   apg_norm=args.apg_norm, apg_momentum=args.apg_momentum, speaker_guidance=args.speaker_guidance)
    if tts is None and (out / "results.jsonl").exists():  # rescore: reuse the synthesis rows of the previous pass
        previous = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines() if line.strip()]
        rows = [{k: v for k, v in r.items() if k in {"id", "text", "speaker", "prompt", "audio", "audio_seconds", "rtf", "error", "selected_factor"}} for r in previous]
        sentences = []
    elif tts is None:  # rescore an interrupted pass: rebuild the rows from the WAVs and their JSON sidecars
        for index, sentence in enumerate(sentences):
            _, prompt_wav, speaker = prompts[index % len(prompts)]
            path = out / f"{sentence['id']}.wav"
            meta = json.loads(path.with_suffix(".json").read_text()) if path.with_suffix(".json").exists() else {}
            rows.append({**sentence, "speaker": speaker, "prompt": prompt_wav.name, "audio": str(path),
                         "audio_seconds": meta.get("audio_seconds", sf.info(str(path)).duration), "rtf": meta.get("rtf")})
        sentences = []
    selector = HFWhisper(args.selector, args.device, args.language) if args.candidates > 1 and sentences else None
    for index, sentence in enumerate(sentences):
        voice, prompt_wav, speaker = prompts[index % len(prompts)]
        path = out / f"{sentence['id']}.wav"
        try:
            seconds = None
            if args.chars_per_second > 0:
                from dacvae_tts.text import normalize

                seconds = max(0.5, len(normalize(sentence["text"], "turkish-v1")) / args.chars_per_second)
            if selector is None:
                result = tts.synthesize(
                    sentence["text"], reference=voice, output=path, seconds=seconds, steps=args.steps, guidance=args.guidance,
                    seed=args.seed + index, sway=args.sway, guidance_until=args.guidance_until,
                    noise_scale=args.noise_scale, duration_scale=args.duration_scale, duration_mode=args.duration_mode,
                    **sampler,
                )
                extra = {"audio_seconds": result.metadata["audio_seconds"], "rtf": result.metadata["rtf"]}
            else:
                extra = best_of_n(tts, selector, sentence["text"], voice, path, seconds, index, args, sampler)
        except ValueError as error:
            rows.append({**sentence, "speaker": speaker, "error": str(error)})
            continue
        rows.append({**sentence, "speaker": speaker, "prompt": prompt_wav.name, "audio": str(path), **extra})
    if selector is not None:
        del selector
    print(f"synthesized {len(rows)} in {time.time() - started:.0f}s", flush=True)
    del tts
    torch.cuda.empty_cache()
    if args.asr_backend == "hf":
        scored = score_hf(rows, out, args)
    else:
        evaluator = Evaluator(args.asr_model, args.dnsmos, args.speaker_model, args.asr_device, language=args.language)
        scored = []
        for row in rows:
            if "error" in row:
                scored.append(row)
                continue
            score = evaluator.score(row["audio"], row["text"], out / row["prompt"])
            scored.append({**row, **{k: v for k, v in score.items() if k != "evaluator"}})
    for row in scored:
        if "error" not in row:
            row.update(audio_stats(row["audio"]))
    (out / "results.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in scored) + "\n")
    good = [r for r in scored if "error" not in r]
    summary = summarize(good) if good else {"count": 0}
    summary.update(
        failed=len(scored) - len(good),
        per_sentence_wer_mean=float(np.mean([r["wer"] for r in good])) if good else None,
        sentences_wer_zero=sum(r["wer"] == 0 for r in good),
        files_clipping=float(np.mean([r["clip_fraction"] > 0 for r in good])) if good else None,
        clipped_sample_fraction=float(np.mean([r["clip_fraction"] for r in good])) if good else None,
        median_lufs=float(np.median([r["lufs"] for r in good])) if good else None,
        checkpoint=args.checkpoint, steps=args.steps, guidance=args.guidance, guidance_until=args.guidance_until,
        noise_scale=args.noise_scale, sway=args.sway, duration_scale=args.duration_scale, asr_model=args.asr_model,
        asr_backend=args.asr_backend, chars_per_second=args.chars_per_second, duration_mode=args.duration_mode,
        candidates=args.candidates, selector=args.selector if args.candidates > 1 else None, **sampler,
        selection_changed=sum(r.get("selected", 0) != 0 for r in good) if args.candidates > 1 else None,
        sentences=str(args.sentences), prompts=len(prompts),
    )
    if args.duration_factors or args.articulation_options or args.duration_model:
        summary.update(duration_factors=args.duration_factors, articulation_options=args.articulation_options,
                       duration_model=args.duration_model,
                       selected_factors={str(f): sum(r.get("selected_factor") == f for r in good)
                                         for f in args.duration_factors or []})
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
