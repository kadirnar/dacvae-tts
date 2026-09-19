# Tensor contracts and execution traces

B=batch, L=padded audio frames, C=codec channels, P=packing factor, D=transformer
width, S=padded text tokens, K=ceil(L/P). L includes reference and target frames.
All masks are boolean, all text IDs int64. Audio and masks share a device.

| Boundary | Shape / meaning |
|---|---|
| Original audio → codec | finite mono float [samples]; channel averaging and rational resampling upstream |
| Posterior means / cached utterance | [utterance_frames,C]; disk float16; load float32, normalize with training-only [C] mean/std |
| Collated `latents` | [B,L,C], reference then target then padding |
| `prompt` | [B,L,C], clean reference values, zero elsewhere |
| `valid`, `prompt_mask` | [B,L]; valid frames and reference frames independently |
| Target mask | `valid & ~prompt_mask`; every example needs at least one target frame |
| Tokens / segments | [B,S]; PAD=0, BOS=1, SEP=2, EOS=3, byte values offset by 4; reference segment=0, target=1 |
| Text encoder / text valid | [B,S,D] / [B,S] |
| Reference frame MLP / pooled summary | [B,L,D] / [B,D] |
| Gaussian noise / flow state / true velocity | [B,L,C] |
| Unpacked input features | [B,L,2C+1]: state, separate reference latents, reference indicator |
| Packed features / validity | [B,K,P(2C+1)] / [B,K], validity is `any` across P frames |
| Input projection / positional signal | [B,K,D] / [K,D] |
| Time t / time embedding | [B] / [B,D], distinct from audio position |
| Adaptive condition | time embedding + reference summary, [B,D] |
| Packed velocity projection | [B,K,PC] |
| Unpacked velocity | [B,L,C], remove only explicitly added pack padding |
| Duration baseline input / output | [B,2D+1] / [B], predicted log frames per target byte |
| Decoding | selected target [Ltgt,C] → mono [Ltgt × hop_length] |

At C=128,P=2: **514 input features → D hidden features → 256 velocity values**.
Reference summary dimension is D (Tiny 256, Small 384). It is compatible with the
D-dimensional time embedding; it is not assumed to equal C.

## One training batch

1. `prepare.py`: mono-average, resample to the codec rate, retain complete utterances;
   reject invalid/silent/out-of-duration rows. No gain normalization or silence trimming.
   Extract deterministic posterior means with frozen DACVAE; store float16 shards.
2. `merge`: enforce identity/split rules, deduplicate, remove singleton speakers,
   recompute retained **training-frame-only** mean/std. Std floor is 0.001; loading
   rejects nonfinite or nonpositive statistics.
3. `LatentDataset`: deterministic epoch-dependent pairing with a distinct recording
   of the same speaker. Read [Lref,C] and [Ltgt,C], normalize, tokenize full transcripts.
4. `collate`: concatenate reference/target, pad across B, construct frame masks and
   text segments. Padding is excluded from counts, reference pooling and durations.
5. `flow_loss`: sample independent standard Gaussian epsilon and uniform t per item.
   Form `xt=(1-t)*epsilon+t*z`, then overwrite reference frames with clean z. True
   velocity is `z-epsilon`. Joint conditioning dropout probability defaults to 0.1.
6. Generator sanitizes invalid state values and non-reference prompt values **before**
   projection. Pack, add sinusoidal packed positions, apply DiT blocks, unpack.
   A token straddling an odd reference boundary remains valid; its components keep
   separate frame masks. Fully padded tokens are excluded as attention keys.
7. Flow error uses target frames only. Duration receives the full un-dropped text and
   reference: supervision comes from original valid counts, never from dropped inputs.
8. Default loss is mean of per-utterance channel/frame MSE plus 0.1 times mean SmoothL1
   duration loss. Opt-in `train.flow_reduction: frame` uses
   `sum(mask*(pred-true)^2)/(C*sum(mask))`; longer targets then have greater weight.
   Duration remains utterance-weighted. DDP denominators include all ranks and the
   complete accumulation window, including unequal microbatch sizes.

Duration counts exclude special/padding tokens; punctuation and spaces are real bytes
and count in the baseline. Reference rate is **codec frames per transcript byte**;
the codec frame rate is **25 frames per second**, a different quantity. Silence
affects the former because the baseline does not trim it. Empty target text fails.

## One inference request

`Synthesizer.synthesize(text, ref_audio=path)` reads a complete reference, obtains
normalized codec means and (unless supplied) uses optional frozen English ASR to
infer the transcript. That ASR is a convenience frontend, not a pretrained component
inside the TTS generator. The prepared latents/transcript can be reused.

Tokenize both texts with checkpoint normalization version. Predict target frames or
use `seconds`; `duration_scale` applies in either case. Validate target .25–30s,
reference .5–30s and at most 2,048 combined text tokens. Frame rounding means durations
are quantized to the codec hop. Current waveform API is one request at a time; lower
level `sample()` supports B>1. No streaming implementation exists.

Start with Gaussian target frames and clean reference frames, compute invariant text
features and pooled reference once, then integrate the velocity field. Decode only
`valid & ~reference`. Reference hidden states **inside DiT** are recomputed each step:
full attention can make them depend on the evolving target. Reference feature caching
does not justify caching these hidden states.

## CFG and integration

Conditioned paths are reference values in x, separate reference channels, the global
summary, reference-transcript features and target-text features. Joint null conditioning
zeros **all** these payloads and the reference-position indicator. It retains the
valid-audio mask, text-valid mask/length, total sequence shape and flow time. This is
structurally conditioned, not mathematically unconditional. Text-valid positions
remain available with zero feature vectors, avoiding an all-masked text attention row.
Learned projection biases can still contribute to null predictions.

Training and inference share that representation. With fixed state outside reference,
lengths, masks and t, changing removed payloads does not change null predictions. No
independent text-only or speaker-only CFG is trained.

`v = v_null + g*(v_conditioned-v_null)`. At g=1, only the conditioned branch executes.
Otherwise each Euler step evaluates both branches. Default 16 steps and g=1.5 means
**32 branch evaluations and 32 actual generator calls**. Feature encoding is separate.

For u in a uniform [0,1] grid, sway points are
`t = u + sway*(cos(pi*u/2)-1+u)`, with supported sway in [-1,0]. Custom lower-level
grids must have steps+1 finite strictly increasing values, start at 0 and end at 1.
Malformed grids are rejected. Each target update uses **t[k+1]-t[k]**. Reference
frames are restored exactly and padding is zero at every step. Analytic zero/constant
fields agree across different valid grids. Audio position is the packed token index,
independent of t; positions extrapolate analytically without a learned table limit,
but long-utterance speech quality is unproven.

## Diagnostics and experimental contracts

Logs contain flow/duration losses, target frames, gradient norms and flow sum/count
by five t intervals and four target-length buckets (see `diagnostics.py`). With
`diagnostics_every > 0`, record activation maxima/finite checks and component gradients.
Separate flow/duration gradient norms are available in a single-process diagnostic
run; DDP skips `autograd.grad` and reports that explicitly. Zero gradients in early
steps are expected from zero initialization; monitor persistent inactivity over time.

Optional temporal reference networks mask before/after convolution. Attention pooling
and mean/std pooling both return [B,D], including zero output for an empty null
reference. Optional duration text statistics add three features, producing [B,2D+4].
These are new architectures, not quality-validated upgrades.
