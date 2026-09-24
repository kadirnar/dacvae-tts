"""Synthesize a fixed sentence list (e.g. Freya-TR-Eval) with held-out prompts and score WER/CER/SIM/DNSMOS.

Prompts are cross-utterance validation cases from the latent cache (unseen speakers, as in monitor.py), used in
rotation over the sentences; each sentence is therefore unseen text on an unseen voice. Whisper large-v3 + the
Turkish metric normalization score intelligibility; DNSMOS OVRL scores audio quality (16 kHz proxy).

  python scripts/eval_sentences.py --checkpoint runs/tr-nano/step-0060000.pt --cache data/tr55/merged \
      --sentences data/eval/freya_tr_eval.jsonl --output outputs/freya-nano --prompts 24 --guidance 3 --steps 32

Evaluation protocol v2 (issue #3, opt-in; see dacvae_tts.eval_protocol). `speaker_similarity` keeps its old meaning
(wavlm-base-plus-sv vs the codec-decoded prompt, a development metric). `--sim-o` adds the seed-tts-eval SIM:
`sim_r` against the codec-decoded prompt WAV and, with `--prompt-audio`, `sim_o` against the ORIGINAL prompt
recording (the published SIM-o). The originals come from the dataset: this script writes OUTPUT/cases.json, then

  python scripts/export_case_audio.py --repo ORG/DATASET --cases outputs/freya-nano/cases.json \
      --cache data/tr55/merged --output data/tr55/eval-audio
  python scripts/eval_sentences.py ... --rescore --protocol-v2 --prompt-audio data/tr55/eval-audio \
      --dnsmos models/sig_bak_ovr.onnx [--band-limit-8k] [--utmosv2]

`--band-limit-8k` is the ASR input of the FreyaTTS scoring protocol (8 kHz resample before ASR only). Their text
scoring differs from ours (apostrophes become spaces, CER counts spaces): `--freya-metric` adds `freya_wer` /
`freya_cer` under that convention next to the usual `wer` / `cer`; use both for Freya-TR-Eval tables.
UTMOS comes from the protocol switches: `--utmos` (or `--utmos utmos22`) scores every output with UTMOS22-strong
into `utmos`, `--utmosv2` (or `--utmos utmosv2`) with UTMOSv2 into `utmosv2`; `--select-utmos` is only the best-of-N
selector's UTMOS.
"""

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from monitor import select_cases  # noqa: E402

from dacvae_tts.cli import add_output_quality_args  # noqa: E402
from dacvae_tts.eval_protocol import (  # noqa: E402
    ProtocolScorer,
    add_protocol_args,
    protocol_from_args,
    utmos_models,
)
from dacvae_tts.inference import OUTPUT_OPTIONS, WINDOW_OPTIONS, Synthesizer, VoiceReference  # noqa: E402
from dacvae_tts.metrics import (  # noqa: E402
    METRIC_NORMALIZATIONS,
    Evaluator,
    default_metric_normalization,
    freya_error_counts,
    row_identity,
    summarize,
)
from dacvae_tts.quality import (  # noqa: E402
    METRIC_FAMILY,
    CandidateScorer,
    load_utmos,
    parse_select_by,
    warn_judge_overlap,
)
from dacvae_tts.speakers import read_speaker_list  # noqa: E402


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


def speaker_embedder(name, device):
    """callable(16 kHz waveform) -> L2-normalized x-vector of a transformers AudioXVector model (SIM selection)."""
    from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

    embedder = Evaluator.__new__(Evaluator)
    embedder.device = torch.device(device)
    embedder.extractor = AutoFeatureExtractor.from_pretrained(name)
    embedder.speaker = AutoModelForAudioXVector.from_pretrained(name).to(device).eval()
    return lambda audio: embedder.embedding(torch.as_tensor(audio))[0]


def best_of_n(tts, scorer, text, voice, path, seconds, index, args, sampler, prompt_audio=None):
    """Synthesize N candidates in one batch; keep the best under --select-by (default cer,wer: the lowest selector
    CER, ties the lowest WER, then the first candidate)."""
    select = partial(scorer.select, reference=prompt_audio) if "sim" in scorer.rule.metrics else scorer.select
    results, metadata = tts.synthesize_many(
        [text], voice, candidates=args.candidates, seconds=seconds, duration_scale=args.duration_scale,
        duration_mode=args.duration_mode, steps=args.steps, guidance=args.guidance, seed=args.seed + index,
        sway=args.sway, guidance_until=args.guidance_until, noise_scale=args.noise_scale, selector=select,
        duration_factors=args.duration_factors, **sampler,
    )
    candidates = results[0]
    best = metadata["selected"][0]
    scores = [c["selection"] for c in candidates]
    audio = candidates[best]["audio"]
    sf.write(path, audio, tts.codec.sample_rate, subtype="FLOAT")
    chosen = {k: candidates[best][k] for k in ("latent_moments", "pre_tanh_gain", "pre_tanh_level") if k in candidates[best]}
    meta = {**metadata, **chosen, "selected": best, "candidate_scores": scores,
            "candidate_cer": [s.get("cer") for s in scores], "candidate_hypotheses": [s.get("hypothesis") for s in scores],
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


def original_prompt_finder(args, prompts, cases):
    """prompt WAV name -> original recording (export_case_audio.py naming) or None; never the codec prompt."""
    if not args.prompt_audio:
        return lambda name: None
    uids = {wav.name: case["prompt_uid"] for (_, wav, _), case in zip(prompts, cases)}

    def find(name):
        if name not in uids:  # rows of a previous pass with a different prompt set
            return None
        path = Path(args.prompt_audio) / (uids[name].replace("/", "_").replace(":", "_") + ".wav")
        return path if path.exists() else None

    missing = sorted(name for name in uids if find(name) is None)
    if missing:
        print(f"warning: {len(missing)} prompts have no original recording in {args.prompt_audio}; "
              f"their rows get no sim_o", flush=True)
    return find


def score_hf(rows, out, args, protocol=None, originals=None):
    """Batched GPU scoring: transformers Whisper (greedy) for WER/CER, WavLM-SV similarity and DNSMOS per clip.

    With a protocol, ASR sees the protocol's input (trailing-silence trim / 8 kHz band limit) and each row gets the
    protocol extras; decoding stays greedy (already deterministic, `--asr-deterministic` concerns faster-whisper).
    Returns (rows, protocol identity or None).
    """
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
    scorer = ProtocolScorer(protocol, device, dnsmos) if protocol is not None else None
    normalization = args.metric_normalization or default_metric_normalization(args.language)
    identity = row_identity({  # the same compact identity the faster-whisper rows carry
        "asr_backend": "hf-greedy", "asr_model": name, "language": args.language, "metric_normalization": normalization,
        "decoding": {"num_beams": 1, "max_new_tokens": 220}, "compute_type": "float16", "device": str(device),
        "speaker_model": args.speaker_model, "protocol_options": scorer.identity["options"] if scorer else None,
    })
    good = [r for r in rows if "error" not in r]
    audios = {r["audio"]: read_audio(r["audio"], 16000) for r in good}
    asr_inputs = {k: scorer.asr_audio(a) if scorer else (a, {}) for k, a in audios.items()}
    hypotheses = {}
    for start in range(0, len(good), args.asr_batch):
        chunk = good[start : start + args.asr_batch]
        features = processor([asr_inputs[r["audio"]][0].numpy() for r in chunk], sampling_rate=16000, return_tensors="pt").input_features
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
            counts = error_counts(r["text"], hypotheses[r["audio"]], normalization)
        except ValueError as error:
            scored.append({**r, "error": str(error)})
            continue
        if r["prompt"] not in prompt_embeddings:
            prompt_embeddings[r["prompt"]] = evaluator.embedding(read_audio(out / r["prompt"], 16000))
        similarity = float((evaluator.embedding(audio) * prompt_embeddings[r["prompt"]]).sum())
        record = {**r, **counts, "hypothesis": hypotheses[r["audio"]], "speaker_similarity": similarity, "asr_backend": "hf-greedy",
                  "evaluator": identity}
        if args.freya_metric:
            record.update(freya_error_counts(r["text"], hypotheses[r["audio"]], normalization))
        if dnsmos is not None:
            record.update(dnsmos(audio.numpy()))
        if scorer is not None:
            original = originals(r["prompt"]) if originals else None
            record.update(asr_inputs[r["audio"]][1])
            record.update(scorer.score(r["audio"], audio, r["text"], hypotheses[r["audio"]], normalization,
                                       original, out / r["prompt"]))
            if original:
                record["prompt_original"] = str(original)
        scored.append(record)
    return scored, (scorer.identity if scorer is not None else None)


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
    add_output_quality_args(parser)
    parser.add_argument("--candidates", type=int, default=1,
                        help="Best-of-N: generate N candidates per sentence and keep the best one under --select-by")
    parser.add_argument("--selector", default="openai/whisper-large-v3-turbo", help="HF Whisper used for best-of-N selection")
    parser.add_argument(
        "--select-by", default="cer,wer",
        help="Best-of-N rule over cer, wer, dnsmos(_sig/_bak), utmos, sim, clip: 'cer,wer' ranks lexicographically "
        "(default, the ASR selector), 'cer:10,dnsmos:1' by weighted sum; dnsmos needs --dnsmos. Selecting with the "
        "judge's model family (Whisper ASR, the same DNSMOS/UTMOS) is an oracle upper bound, not a gain")
    parser.add_argument("--select-utmos", choices=["utmos22", "utmosv2"], default="utmos22", help="UTMOS used by --select-by utmos")
    parser.add_argument(
        "--select-speaker-model", default="microsoft/unispeech-sat-base-plus-sv",
        help="Speaker model of --select-by sim; must differ from the SIM judge --speaker-model (arXiv 2607.08256)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--language", default="tr")
    parser.add_argument(
        "--metric-normalization", choices=METRIC_NORMALIZATIONS,
        help="WER/CER text normalization of both --asr-backend paths (default: turkish-v1 for --language tr, else "
             "english-unicode-v2)",
    )
    parser.add_argument(
        "--freya-metric", action="store_true",
        help="Also score WER/CER the Freya-TR-Eval way (apostrophes -> spaces, CER with spaces) as freya_wer/freya_cer",
    )
    parser.add_argument("--asr-model", default="large-v3")
    parser.add_argument("--asr-device", default="cuda")
    parser.add_argument("--dnsmos")
    parser.add_argument("--speaker-model", default="microsoft/wavlm-base-plus-sv")
    parser.add_argument(
        "--exclude-speakers", help="Labels never used as prompts, e.g. leakage.json of speaker_clusters.py"
    )
    parser.add_argument(
        "--asr-backend", choices=["faster-whisper", "hf"], default="faster-whisper",
        help="hf = transformers Whisper on the GPU with cross-clip batching (~30x faster than CPU faster-whisper; greedy decoding)",
    )
    parser.add_argument("--asr-batch", type=int, default=24)
    parser.add_argument("--rescore", action="store_true", help="Skip synthesis when the WAVs already exist; only score")
    parser.add_argument(
        "--prompt-audio",
        help="Directory of ORIGINAL prompt recordings (export_case_audio.py --cases OUTPUT/cases.json); with a "
        "protocol v2 speaker metric the rows get sim_o against them (sim_r is always vs the codec-decoded prompt)",
    )
    add_protocol_args(parser)
    args = parser.parse_args()
    protocol = protocol_from_args(args)
    if args.duration_factors and (args.candidates < 2 or len(args.duration_factors) > args.candidates):
        parser.error("--duration-factors is a best-of-N option: needs --candidates >= max(2, number of factors)")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    sentences = load_sentences(args.sentences, args.limit)
    exclude = read_speaker_list(args.exclude_speakers) if args.exclude_speakers else ()
    data, cases = select_cases(args.cache, args.prompts, args.seed, exclude=exclude)
    (out / "cases.json").write_text(json.dumps(cases, indent=1, ensure_ascii=False))  # input of export_case_audio.py
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
                   apg_norm=args.apg_norm, apg_momentum=args.apg_momentum, speaker_guidance=args.speaker_guidance,
                   **{name: getattr(args, name) for name in (*WINDOW_OPTIONS, *OUTPUT_OPTIONS)})
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
    rule, selector, selection_bias = parse_select_by(args.select_by), None, None
    if args.candidates > 1 and sentences:
        from dacvae_tts.metrics import DNSMOS

        needs = {METRIC_FAMILY[m] for m in rule.metrics}
        selector = CandidateScorer(
            rule,
            transcriber=HFWhisper(args.selector, args.device, args.language) if "asr" in needs else None,
            dnsmos=DNSMOS(args.dnsmos) if "dnsmos" in needs and args.dnsmos else None,
            utmos=load_utmos(args.select_utmos, args.device) if "utmos" in needs else None,
            speaker=speaker_embedder(args.select_speaker_model, args.device) if "speaker" in needs else None,
        )
        judge_asr = args.asr_model if "whisper" in args.asr_model else f"whisper-{args.asr_model}"
        selection_bias = warn_judge_overlap(
            rule,
            {"asr": args.selector, "dnsmos": args.dnsmos, "utmos": args.select_utmos, "speaker": args.select_speaker_model},
            {"asr": judge_asr, "dnsmos": args.dnsmos, "utmos": ",".join(utmos_models(protocol)) or None,
             "speaker": args.speaker_model},
        )
    prompt_audio = {}
    for index, sentence in enumerate(sentences):
        voice, prompt_wav, speaker = prompts[index % len(prompts)]
        path = out / f"{sentence['id']}.wav"
        if selector is not None and "sim" in rule.metrics and prompt_wav not in prompt_audio:
            prompt_audio[prompt_wav] = sf.read(str(prompt_wav), dtype="float32")[0]
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
                extra = best_of_n(tts, selector, sentence["text"], voice, path, seconds, index, args, sampler,
                                  prompt_audio.get(prompt_wav))
        except ValueError as error:
            rows.append({**sentence, "speaker": speaker, "error": str(error)})
            continue
        rows.append({**sentence, "speaker": speaker, "prompt": prompt_wav.name, "audio": str(path), **extra})
    if selector is not None:
        del selector
    print(f"synthesized {len(rows)} in {time.time() - started:.0f}s", flush=True)
    del tts
    torch.cuda.empty_cache()
    originals = original_prompt_finder(args, prompts, cases)
    if args.asr_backend == "hf":
        scored, protocol_identity = score_hf(rows, out, args, protocol, originals)
    else:
        evaluator = Evaluator(args.asr_model, args.dnsmos, args.speaker_model, args.asr_device,
                              metric_normalization=args.metric_normalization, language=args.language, protocol=protocol)
        protocol_identity = evaluator.identity.get("protocol")
        scored = []
        for row in rows:
            if "error" in row:
                scored.append(row)
                continue
            original = originals(row["prompt"])
            score = evaluator.score(row["audio"], row["text"], out / row["prompt"], original_prompt=original,
                                    codec_prompt=out / row["prompt"])
            extra = {"prompt_original": str(original)} if protocol is not None and original else {}
            if args.freya_metric:
                extra.update(freya_error_counts(row["text"], score["hypothesis"], evaluator.metric_normalization))
            scored.append({**row, **score, "evaluator": evaluator.row_identity, **extra})
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
        metric_normalization=args.metric_normalization or default_metric_normalization(args.language),
        candidates=args.candidates, selector=args.selector if args.candidates > 1 else None, **sampler,
        selection_changed=sum(r.get("selected", 0) != 0 for r in good) if args.candidates > 1 else None,
        select_by=args.select_by if args.candidates > 1 else None, selection_judge_overlap=selection_bias,
        utmos_model=utmos_models(protocol) or None,  # the means are `utmos` / `utmosv2` (summarize)
        sentences=str(args.sentences), prompts=len(prompts),
    )
    if protocol is not None:
        summary.update(protocol=protocol_identity, prompt_audio=args.prompt_audio)
    if args.duration_factors or args.articulation_options or args.duration_model:
        summary.update(duration_factors=args.duration_factors, articulation_options=args.articulation_options,
                       duration_model=args.duration_model,
                       selected_factors={str(f): sum(r.get("selected_factor") == f for r in good)
                                         for f in args.duration_factors or []})
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
