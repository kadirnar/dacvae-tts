# Architecture decision record — 18 September 2026

This is a new, untrained research implementation. No quality, cloning success,
WER/CER/DNSMOS gain, or real-time performance has been established yet.

## Evidence and choices

| Primary source | Finding relevant to this project | Decision |
|---|---|---|
| [DACVAE source](https://github.com/facebookresearch/dacvae) and [checkpoint card](https://huggingface.co/facebook/dacvae-watermarked) | Continuous variational codec; public encode samples the posterior; decoder includes watermarking. | Frozen codec adapter, deterministic posterior-mean cache, train-set channel statistics; probe actual latent dimension, hop and sample rate. Preserve upstream decoder behavior. |
| [E2 TTS](https://arxiv.org/abs/2406.18009) | Conditional flow matching and audio infilling support reference-conditioned speech without forced alignments. | Parallel latent flow generation; reference audio conditioning. |
| [F5-TTS](https://aclanthology.org/2025.acl-long.313/) | Text refinement and a nonuniform integration schedule improve a flow TTS baseline. | Convolutional text encoder, explicit text cross-attention, optional sway schedule. This is an adaptation, not an F5 reproduction. |
| [F5R-TTS](https://arxiv.org/abs/2504.02407) | Metric rewards improve intelligibility and speaker similarity; GRPO requires a probabilistic policy construction. | Start with offline preference learning. Do not pretend deterministic ODE velocities are policy log probabilities. |
| [Diffusion-DPO](https://arxiv.org/abs/2311.12908) | Preference learning can compare denoising errors against a frozen reference. | Experimental flow-error preference surrogate, shared noise/time, reference model, winner anchor and real-data replay. Not an exact likelihood DPO or GRPO implementation. |
| [DMOSpeech 2](https://arxiv.org/abs/2507.14988) | Duration matters for metric optimization and low-step generation. | Train a duration head; evaluate duration multipliers separately from sampler steps and guidance. Distill only after the teacher is good. |
| [Microsoft DNSMOS](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS) | Non-intrusive quality scoring uses specific preprocessing and calibrated models. | Implement standard non-personalized SIG/BAK/OVRL evaluation with the official ONNX model; never substitute a made-up MOS proxy. |
| [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) | Dispatches to supported fused attention kernels. | Native SDPA, BF16, fused AdamW on CUDA, length buckets, cached codec latents, gradient accumulation/checkpointing, optional compile and DDP. |

## Proposed model

English UTF-8 byte text avoids an external tokenizer or pretrained language model.
Text normalization is deliberately conservative: prepare spoken-out numbers,
abbreviations and unusual symbols in the dataset. Prompt and target transcripts
have distinct segment embeddings. A convolutional text encoder provides contextual
keys/values to cross-attention in each diffusion-transformer block.

Reference and target latents are concatenated in time. Only target frames are
noised and supervised; the reference remains fixed throughout sampling. A pooled
reference encoder adds global voice conditioning through adaptive layer norms.
Training pairs **different complete utterances from the same speaker**, never a
cropped waveform with an unchanged transcript. This encourages transfer of speaker
identity across linguistic content. No pretrained speaker encoder is used inside
the TTS model. Reference transcripts are required internally; the audio-only API can
obtain them with optional ASR (see the subsequent [implementation audit](architecture-audit.md)).

Two adjacent latent frames are packed reversibly into a single transformer token.
Packing halves sequence length without discarding channels. Padding masks apply
inside text convolutions, attention, pooling, and losses. Frame-level prompt masks
survive packing, including a patch that straddles the prompt boundary.

Tiny: width 256, 8 blocks, 4 heads (13.53M trainable parameters).
Small: width 384, 12 blocks, 6 heads (44.36M trainable parameters).
Both include their text, reference and duration networks. Exact counts are emitted
by `dacvae-tts inspect`; the frozen codec is additional and can dominate deployment
size (the tested codec has 107.67M parameters, 48 kHz output and 25 latent frames/s).
Start with 16–32 flow steps; 4–8 steps are an experimental post-distillation
target. No streaming claim: synthesis uses full-sequence attention.

Duration prediction estimates target log frames per text byte from target text,
reference voice and reference speaking rate. Explicit duration remains available.
This simple head and byte alignment are risks for difficult English pronunciation;
compare them against a phoneme/alignment frontend if held-out WER stalls.

## Data contract and experiments

The user confirmed a 4M-row English corpus, eight GPUs (model may vary), and frozen
pretrained DACVAE. Use only this corpus for TTS training, statistics, preferences, and
teacher/student distillation. Frozen DACVAE is a representation dependency; its
pretraining is external. If “from scratch” includes the codec, supply a separately
trained compatible DACVAE checkpoint. This project does not train DACVAE itself.
ASR, DNSMOS and speaker verification models are optional external evaluators, not
TTS initializations. Their use as reward judges must be distinguished from a
strict prohibition on all externally pretrained models.

Require real speaker IDs, at least two utterances per retained speaker, accurate
English transcripts, and utterance-level clips. Default splits are speaker-disjoint.
Keep independent prompts and targets inside each split. Also create a seen-speaker,
utterance-disjoint diagnostic set when enough recordings are available. Do not
estimate normalization statistics from validation/test. Single-speaker data cannot
establish general zero-shot cloning across identities.

First audit transcript quality, duplicates, clipping, silence and speaker diversity;
then score original and codec-reconstructed audio as a representation diagnostic,
not a guaranteed mathematical upper bound on speech metrics.
Overfit a small training subset before scaling. Sweep tiny/small, packing 1/2,
steps 8/16/32, guidance 1/1.5/2, and duration scale .9/1/1.1 on validation only.

## Post-training sequence

1. Establish pretraining baseline using held-out speakers and texts, fixed prompts,
   seeds, duration policy and evaluator versions. Report corpus WER/CER (total edit
   counts divided by reference counts), mean DNSMOS SIG/BAK/OVRL, speaker cosine,
   generation failure rate, latency/RTF and peak GPU memory. Define cloning success
   from a speaker-verifier threshold calibrated on real same/different-speaker pairs
   plus intelligibility checks; cosine alone is not a success rate.
2. Generate multiple candidates using **training-split** texts/prompts only.
   Rank with WER, CER, calibrated DNSMOS OVRL and speaker similarity. Reject a winner
   if it degrades intelligibility or voice similarity; skip near ties. Keep all raw
   metrics so one composite score cannot hide a regression.
3. Fine-tune with the reference-relative flow-error preference surrogate and winner
   reconstruction anchor, while replaying real training speech. Begin with a low LR,
   monitor an independent ASR model when possible, and select checkpoints using
   validation metrics and blinded listening. Rewards are non-differentiable and are
   computed offline. DNSMOS can reward over-smoothing; never optimize it alone.
4. Optional offline trajectory distillation: a frozen in-domain teacher produces
   full trajectories; a student learns each coarse segment's average velocity from
   the teacher's on-policy states. Progressively halve the step count and rerun the
   entire quality comparison. The teacher must originate from this corpus.
5. Promote only checkpoints that lower WER/CER and raise DNSMOS without hurting
   speaker similarity, natural prosody, latency or subgroup performance. Use paired
   bootstrap intervals over utterances (and speaker-level resampling for cloning).
   A held-out failure is a failed experiment, not a result to conceal.

The implementation enables these experiments. Actual improvements require the
dataset, a trained baseline, generated candidates, evaluation and listening tests.
