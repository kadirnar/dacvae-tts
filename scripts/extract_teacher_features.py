"""Precompute teacher targets for the training-only speech-REPA and TLA-SA losses (issue #10).

Writes a sidecar store next to the latent cache (format: dacvae_tts/teacher.py); the cache is never
modified. Audio comes from each row's original file when it still exists, otherwise from decoding the
stored latents with the cache's DACVAE checkpoint (rows prepared from embedded Parquet audio). Heavy
dependencies (transformers, SpeechBrain, dacvae) are imported only by the subcommand that needs them.

  # speech-REPA frames: multilingual SSL teacher (default mHuBERT-147, last layer), 50 Hz -> 25 fps,
  # PCA 768 -> 256 fitted on 1000 utterances; then set model.repa_dim: 256
  python scripts/extract_teacher_features.py frames --cache data/cache-tr \\
      --output data/cache-tr/teacher/mhubert147-l12-pca256 --layer 12 --pca-dim 256 --device cuda

  # the same on 8 GPUs: fit the PCA once, one process per GPU, then merge the partitions
  python scripts/extract_teacher_features.py fit-pca --cache C --output S --layer 12 --pca-dim 256 --device cuda
  for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i python scripts/extract_teacher_features.py frames \\
      --cache C --output S --layer 12 --device cuda --shard-index $i --num-shards 8 & done; wait
  python scripts/extract_teacher_features.py merge --output S --cache C

  # TLA-SA speaker embeddings: use a speaker model other than the evaluation's WavLM-large ECAPA
  # (SIM-o), so that a SIM gain is not metric gaming; SpeechBrain ECAPA (192-d) needs `pip install
  # speechbrain`; `--embedder hf-xvector` (transformers only) is WavLM-based and therefore correlated
  python scripts/extract_teacher_features.py speakers --cache data/cache-tr \\
      --output data/cache-tr/teacher/ecapa-speechbrain --device cuda

Every run is resumable (rows already in a partition are skipped) and `--splits train` is enough for
training; add val/test only if you want to inspect the targets there. `--embedder pkg.module:factory`
plugs in any embedder: factory(model_id, device) returning an object with `sample_rate` and
`__call__(waveform) -> [E]` (and optionally `model_id`, recorded in the store); without `--model` it is
called as factory(device=...), so each factory falls back to its own default model.
"""

import argparse
import importlib
import json
from pathlib import Path

from dacvae_tts.teacher import CacheAudio, extract_frames, extract_speakers, fit_cache_pca, merge_parts

DEFAULT_SSL = "utter-project/mHuBERT-147"  # HuBERT base, 12 layers x 768, 147 languages, 16 kHz, 50 Hz
ALTERNATIVE_SSL = "facebook/w2v-bert-2.0"  # 24 layers x 1024, 4.5M h / 143 languages, 50 Hz
DEFAULT_SPEAKER = "speechbrain/spkrec-ecapa-voxceleb"


class HuggingFaceSSL:
    """Hidden states of one layer of a Hugging Face speech encoder (HuBERT, wav2vec 2.0, w2v-BERT 2.0).

    Utterances run one at a time: HuBERT-base checkpoints use group-normalized convolutions, whose
    statistics padding would change. `layer` counts transformer layers (0 = input embeddings,
    -1 = last); A-DMA found the last SSL layer better than an average of layers.
    """

    def __init__(self, model_id=DEFAULT_SSL, layer=-1, device="cpu", frame_rate=50.0):
        import torch
        from transformers import AutoFeatureExtractor, AutoModel

        self.torch, self.device = torch, torch.device(device)
        self.extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval().requires_grad_(False)
        depth = int(self.model.config.num_hidden_layers)
        self.layer = depth if layer < 0 else layer
        if not 0 <= self.layer <= depth:
            raise ValueError(f"{model_id} has layers 0..{depth}")
        self.sample_rate = int(getattr(self.extractor, "sampling_rate", 16000))
        self.frame_rate, self.dim = frame_rate, int(self.model.config.hidden_size)

    def __call__(self, waveform):
        inputs = self.extractor(waveform, sampling_rate=self.sample_rate, return_tensors="pt")
        with self.torch.inference_mode():
            output = self.model(
                **{k: v.to(self.device) for k, v in inputs.items()}, output_hidden_states=True
            )
        return output.hidden_states[self.layer][0].float().cpu()


class SpeechBrainEmbedder:
    """SpeechBrain ECAPA-TDNN (VoxCeleb) utterance embeddings, 192-d; a non-WavLM training target."""

    sample_rate = 16000

    def __init__(self, model_id=DEFAULT_SPEAKER, device="cpu"):
        import torch

        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:  # SpeechBrain < 1.0
            from speechbrain.pretrained import EncoderClassifier
        savedir = Path.home() / ".cache" / "speechbrain" / model_id.replace("/", "--")
        self.torch, self.model_id = torch, model_id
        self.model = EncoderClassifier.from_hparams(
            source=model_id, savedir=str(savedir), run_opts={"device": str(device)}
        )

    def __call__(self, waveform):
        with self.torch.inference_mode():
            return self.model.encode_batch(self.torch.from_numpy(waveform)[None]).reshape(-1).float().cpu()


class HuggingFaceXVector:
    """transformers x-vector heads (e.g. microsoft/wavlm-base-plus-sv); no extra dependency, but WavLM."""

    sample_rate = 16000

    def __init__(self, model_id="microsoft/wavlm-base-plus-sv", device="cpu"):
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

        self.torch, self.device, self.model_id = torch, torch.device(device), model_id
        self.extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = AutoModelForAudioXVector.from_pretrained(model_id).to(self.device).eval()

    def __call__(self, waveform):
        inputs = self.extractor(waveform, sampling_rate=self.sample_rate, return_tensors="pt")
        with self.torch.inference_mode():
            return self.model(**{k: v.to(self.device) for k, v in inputs.items()}).embeddings[0].float().cpu()


EMBEDDERS = {"speechbrain": SpeechBrainEmbedder, "hf-xvector": HuggingFaceXVector}


def load_frame_teacher(args):
    return HuggingFaceSSL(args.model, args.layer, args.device, args.teacher_frame_rate)


def load_embedder(args):
    if ":" in args.embedder:
        module, name = args.embedder.split(":", 1)
        factory = getattr(importlib.import_module(module), name)
    elif args.embedder in EMBEDDERS:
        factory = EMBEDDERS[args.embedder]
    else:
        raise ValueError(
            f"Unknown embedder {args.embedder}; use {sorted(EMBEDDERS)} or package.module:factory"
        )
    # Each factory has its own default model: SpeechBrain's ECAPA id would break hf-xvector.
    return factory(args.model, args.device) if args.model else factory(device=args.device)


def load_decoder(cache, checkpoint, device):
    """`Codec.decode` of the cache's own DACVAE checkpoint, verified against the cache metadata."""
    from dacvae_tts.codec import Codec

    meta = json.loads((Path(cache) / "metadata.json").read_text())
    codec = Codec(checkpoint or meta["checkpoint"], device)
    for key in ("sample_rate", "hop_length", "latent_dim"):
        if getattr(codec, key) != meta[key]:
            raise ValueError(f"Codec {key}={getattr(codec, key)} does not match the cache ({meta[key]})")
    return codec.decode


class LazyDecoder:
    """Loads DACVAE on the first row that has no original audio file."""

    def __init__(self, cache, checkpoint, device):
        self.cache, self.checkpoint, self.device, self.decode = cache, checkpoint, device, None

    def __call__(self, latents):
        if self.decode is None:
            self.decode = load_decoder(self.cache, self.checkpoint, self.device)
        return self.decode(latents)


def audio_source(args):
    return CacheAudio(args.cache, args.audio_source, LazyDecoder(args.cache, args.codec, args.device))


def frames(args):
    """A pca.pt in the store root (fit-pca) is always applied, so shards need no --pca-dim."""
    output = Path(args.output)
    pca_path = output / "pca.pt"
    if args.pca_dim and pca_path.exists():
        import torch

        fitted = torch.load(pca_path, weights_only=True)["components"].shape[1]
        if fitted != args.pca_dim:
            raise ValueError(f"{pca_path} projects to {fitted} dims, not --pca-dim {args.pca_dim}")
    if args.pca_dim and not pca_path.exists() and args.num_shards > 1:
        raise ValueError("Sharded extraction with PCA: run the fit-pca subcommand once first")
    teacher, audio = load_frame_teacher(args), audio_source(args)
    if args.pca_dim and not pca_path.exists():
        fit_pca(args, teacher, audio)
    describe = {"model": args.model, "layer": teacher.layer, "audio_source": args.audio_source}
    return extract_frames(
        args.cache,
        output,
        teacher,
        audio,
        args.splits,
        args.shard_index,
        args.num_shards,
        describe,
        progress=not args.quiet,
    )


def fit_pca(args, teacher=None, audio=None):
    if not args.pca_dim:
        raise ValueError("fit-pca needs --pca-dim")
    teacher, audio = teacher or load_frame_teacher(args), audio or audio_source(args)
    pca = fit_cache_pca(
        args.cache, args.output, teacher, audio, args.pca_dim, args.pca_rows, args.seed, args.splits
    )
    kept = float(pca["explained"].sum())
    return {
        "pca": str(Path(args.output) / "pca.pt"),
        "rows": pca["rows"],
        "frames": pca["frames"],
        "explained": kept,
    }


def speakers(args):
    embedder = load_embedder(args)
    describe = {
        "embedder": args.embedder,
        "model": getattr(embedder, "model_id", args.model),  # the model the embedder actually loaded
        "audio_source": args.audio_source,
    }
    return extract_speakers(
        args.cache,
        args.output,
        embedder,
        audio_source(args),
        args.splits,
        args.shard_index,
        args.num_shards,
        describe,
        progress=not args.quiet,
    )


def merge(args):
    return merge_parts(args.output, args.cache, args.allow_missing)


def parser():
    root = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = root.add_subparsers(dest="command", required=True)

    def common(name, function, help):
        command = commands.add_parser(name, help=help)
        command.set_defaults(function=function)
        command.add_argument("--cache", required=True, help="Merged latent cache directory")
        command.add_argument("--output", required=True, help="Store directory, e.g. CACHE/teacher/NAME")
        command.add_argument("--splits", type=lambda s: tuple(s.split(",")), default=("train",))
        command.add_argument("--device", default="cpu")
        command.add_argument("--audio-source", choices=("auto", "original", "decode"), default="auto")
        command.add_argument("--codec", default=None, help="DACVAE checkpoint; default: the cache's")
        command.add_argument("--quiet", action="store_true")
        return command

    def ssl(command):
        command.add_argument(
            "--model", default=DEFAULT_SSL, help=f"SSL teacher (alternative: {ALTERNATIVE_SSL})"
        )
        command.add_argument("--layer", type=int, default=-1, help="Transformer layer; -1 = last")
        command.add_argument("--teacher-frame-rate", type=float, default=50.0)
        command.add_argument("--pca-dim", type=int, default=0, help="Project to this width; 0 keeps all")
        command.add_argument("--pca-rows", type=int, default=1000, help="Utterances used to fit the PCA")
        command.add_argument("--seed", type=int, default=0)

    def sharded(command):
        command.add_argument("--shard-index", type=int, default=0)
        command.add_argument("--num-shards", type=int, default=1)

    ssl(common("fit-pca", fit_pca, "Fit the frame PCA once (before a sharded frames run)"))
    command = common("frames", frames, "speech-REPA frame features")
    ssl(command)
    sharded(command)
    command = common("speakers", speakers, "TLA-SA utterance speaker embeddings")
    command.add_argument(
        "--embedder", default="speechbrain", help="speechbrain, hf-xvector or pkg.mod:factory"
    )
    command.add_argument(
        "--model",
        default=None,
        help=f"Speaker model id (default: the embedder's own, {DEFAULT_SPEAKER} for speechbrain)",
    )
    sharded(command)
    command = commands.add_parser("merge", help="Combine the partitions of a sharded run")
    command.set_defaults(function=merge)
    command.add_argument("--output", required=True)
    command.add_argument("--cache", default=None, help="Verify that every selected cache row is covered")
    command.add_argument("--allow-missing", action="store_true")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    result = args.function(args)
    print(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    main()
