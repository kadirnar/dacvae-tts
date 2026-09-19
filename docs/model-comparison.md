# DAC and DACVAE model comparison

Research date: 19 September 2026. This is a comparison of verified public model
families, not an exhaustive inventory of community fine-tunes or unpublished
systems. The research introduced no model-code changes. Published measurements
below are author-reported; these models were not independently benchmarked here.

## Codec scope

| Category | Representation | Interpretation |
|---|---|---|
| [Descript DAC](https://github.com/descriptinc/descript-audio-codec) | Residual-vector-quantized discrete codes, with continuous internal features | Used by autoregressive, masked-token, and some continuous-latent generators. |
| [Meta DACVAE](https://github.com/facebookresearch/dacvae) | Continuous variational latents | The local checkpoint has 128 channels at 25 Hz for 48 kHz audio. |
| Custom DAC-derived codecs | Modified quantizers, semantic features, rates, or VAE bottlenecks | Related architecture does not imply interchangeable weights or latent spaces. |

## Continuous-latent speech models

| Model | Codec and architecture | Comparison with this repository |
|---|---|---|
| [Echo-TTS](https://jordandarefsky.com/blog/2025/echo/) | Approximately 2.4B; flow matching over an 80-dimensional continuous representation derived from Fish S1-DAC, not Meta DACVAE. Byte text encoder and separate reference transformer; no reference transcript required. | Strong architectural neighbor, but different codec and much greater capacity. Uses separate text/reference dropout and guidance. |
| [Irodori-TTS v3](https://huggingface.co/Aratako/Irodori-TTS-500M-v3) | Approximately 500M; rectified-flow DiT, reference transformer, duration predictor; Japanese 32-dimensional Semantic-DACVAE and pretrained Japanese text embeddings. | Similar generation paradigm, different language, codec, and text initialization. |
| [Irodori v4/v4.1](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small) | Semantic-DACVAE; richer text/caption conditioning. v4.1 trains duration prediction separately after generator convergence. | Relevant to duration and conditioning comparisons; Japanese results do not establish English quality. |
| [Alignment-Free Text-Audiobox / Text-AB](https://arxiv.org/html/2609.03992v1) | Main model 3B; 128-dimensional, 25 Hz DACVAE at 48 kHz; flow-matching DiT, pretrained mT5, audio context. Pretraining uses 480,000 hours. | Closest major published representation comparison. Avoids explicit token-duration prediction, which is not equivalent to eliminating all inference-length decisions. |
| [DMOSpeech, original](https://proceedings.mlr.press/v267/li25ay.html) | Custom DAC-derived VAE; diffusion generation with distribution-matching distillation, adversarial training, and direct intelligibility/speaker metric optimization. | Closest post-training methodology comparison; results are not established for this repository's codec/objective. |
| [LAION JAX DACVAE-Echo](https://github.com/LAION-AI/jax-dacvae-echotts), [scaled Echo](https://github.com/LAION-AI/scaled-echo-tts), [Vocalino/OpenEcho](https://github.com/christophschuhmann/dacvae-echo) | Meta DACVAE adaptations of reference-conditioned flow generation. JAX implementation reports roughly 1.3B; scaled implementation has approximately 800M/3B/8B presets. | Close implementation relatives. Available code and training scaling measurements do not establish speech quality. |
| [3arab-TTS v2](https://huggingface.co/sherif1313/3arab-TTS-500M-v2) | Approximately 553M; rectified-flow DiT with text/reference transformers and Arabic 32-dimensional DACVAE. Author reports approximately 700 training hours. | Language-specific comparison; not independently verified evidence for English cloning quality or equivalent data efficiency. |

## Discrete DAC speech models

| Model | Codec / generation | Voice conditioning and comparison |
|---|---|---|
| [Parakeet, Darefsky et al.](https://jordandarefsky.com/blog/2024/parakeet/) | Original 44.1 kHz DAC; 3B autoregressive encoder-decoder with delayed codebooks; approximately 100,000 hours. | Conversational speech research predecessor. Demonstrations explicitly include selected best-of-multiple samples. Unrelated to NVIDIA Parakeet ASR. |
| [Parler Mini](https://huggingface.co/parler-tts/parler-tts-mini-v1.1) / [Large](https://huggingface.co/parler-tts/parler-tts-large-v1) | 44.1 kHz DAC; approximately 0.9B / 2.2B autoregressive models. | Text-described voices/styles and known voices, rather than native arbitrary-reference cloning. |
| [Dia 1.6B](https://github.com/nari-labs/dia) | 44.1 kHz DAC; autoregressive encoder-decoder. | Dialogue and audio prompting with matching transcript context. Speaker turn tags are not necessarily fixed identity lookups. |
| [Zonos 0.1](https://github.com/Zyphra/Zonos) | DAC; approximately 1.6B Transformer or Mamba2/Transformer hybrid. | Phonemes plus reference speaker embeddings, with optional audio prefix. Embedding path does not require a reference transcript. |
| [Zonos 2](https://www.zyphra.com/our-work/zonos2) | DAC; autoregressive MoE, 8B total and approximately 900M active parameters; removes CFG. | Active parameters do not represent the total deployment footprint. |
| [OuteTTS 1.0 1B](https://huggingface.co/OuteAI/Llama-OuteTTS-1.0-1B) | IBM DAC speech variant, two codebooks; Llama-based AR generation; approximately 150 audio tokens/second. | Reference audio creates custom speaker profiles. Reports approximately 60,000 training hours. Earlier versions are not automatically DAC models. |
| [Fish S1 / S1-mini](https://huggingface.co/fishaudio/s1-mini) | Modified Fish DAC; dual AR, 4B / approximately 0.5B. | Reference prompting; mini is distilled. Over two million training hours and RLHF are reported, unlike a scratch-only baseline. |
| [Fish S2-Pro](https://huggingface.co/fishaudio/s2-pro) | Modified DAC, 10 codebooks at approximately 21 Hz; 4B slow backbone plus 400M fast decoder. | Reference cloning and streaming-oriented inference; reports over ten million training hours. |
| [MaskGCT](https://arxiv.org/html/2409.00750v3) | Approximately 1.1B; two-stage non-autoregressive masked-token generation. Custom DAC-derived acoustic tokenizer replaces the decoder with Vocos; separate semantic tokenizer. | Demonstrates that DAC does not imply AR. Semantic conditioning introduces additional pretrained dependencies. |
| [Higgs Audio v2 tokenizer](https://github.com/boson-ai/higgs-audio/blob/main/boson_multimodal/audio_processing/higgs_audio_tokenizer.py) | X-Codec-derived semantic/acoustic tokenizer with DAC encoder/decoder components, used for LLM-based audio generation. | Broader DAC-derived category, not direct original DAC checkpoint use; includes semantic-teacher dependencies. |

## Reference-audio interfaces

| Interface | Examples | Inputs |
|---|---|---|
| No reference transcript | Echo, Irodori, Zonos speaker-embedding path | Target text and reference audio. |
| Transcript-conditioned prompting | Dia; [Fish documented workflow](https://github.com/fishaudio/fish-speech/blob/main/docs/en/inference.md) | Target text, reference audio, and reference transcript. |
| Audio-only wrapper over transcript conditioning | This repository | Target text and reference audio; optional ASR supplies the reference transcript internally. |

The local interface requires no speaker-ID lookup, but remains transcript-conditioned
internally. See [architecture audit](architecture-audit.md).

## Other audio tasks

These systems inform codec or generation comparisons, not TTS cloning rankings.

| Model | Task / relationship |
|---|---|
| [Movie Gen Audio](https://arxiv.org/html/2410.13720v1) | Large flow-matching transformer for soundtracks, effects, and music; Meta DACVAE. |
| [DiTVC](https://pixl.cs.princeton.edu/pubs/Wang_2025_OVC/Wang-WASPAA-2025.pdf) | DACVAE reference-conditioned voice conversion; source speech supplies content and timing. |
| [MultiFoley](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_Video-Guided_Foley_Sound_Generation_with_Multimodal_Controls_CVPR_2025_paper.pdf) | Video-guided Foley with multimodal controls and custom DAC-VAE. |
| [HunyuanVideo-Foley](https://arxiv.org/html/2508.16930v1) | Multimodal diffusion/flow; custom DAC-VAE. Published 128-dimensional variant runs at 50 Hz, unlike this repository's 25 Hz codec. |
| [MOSS-SoundEffect v2](https://github.com/OpenMOSS/MOSS-TTS/blob/main/moss_soundeffect_v2/README.md) | Text-to-sound effects; DiT flow matching with DACVAE. |
| [VampNet](https://zenodo.org/records/10265299/files/000042.pdf) | Coarse-to-fine masked-token music generation with a DAC-derived tokenizer. |
| [MaskVAT](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11856.pdf) | Non-autoregressive masked-token video-to-audio using DAC. |
| [DiscoDiff](https://tik-db.ee.ethz.ch/file/98421a1293c8389a17dded4c09addaba/) | Coarse-to-fine text-to-music diffusion over continuous DAC features. |
| [The Rhythm in Anything / TRIA](https://interactiveaudiolab.github.io/assets/papers/oreilly2025ismir.pdf) | Approximately 43M masked-DAC-token drum generator. Similar size to local Small, but much narrower task. |

## Version-specific exclusions

- [Dia 2](https://github.com/nari-labs/dia2) uses Mimi.
- [DMOSpeech 2](https://arxiv.org/html/2507.14988v1) uses mel spectrograms and Vocos.
- [DiTTo-TTS](https://arxiv.org/html/2406.11427v2) primarily uses Mel-VAE; DAC models are ablations.
- [Stable Audio Open](https://arxiv.org/html/2407.14358v2) has its own continuous autoencoder. Shared convolutional/Snake components do not establish use of a DAC or Meta DACVAE checkpoint.
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) has its own speech tokenizer. Decoder resemblance alone does not establish direct DAC/DACVAE checkpoint use.

## Comparative evidence and limitations

DiTTo's reported codec ablation illustrates the difference between reconstruction
and generation quality:

| DiTTo configuration | WER (lower is better) | Codec reconstruction PESQ (higher is better) |
|---|---:|---:|
| Mel-VAE | 2.93 | 2.95 |
| DAC 24 kHz | 7.21 | 4.37 |
| DAC 44 kHz | 14.58 | 3.74 |

Better reconstruction PESQ did not yield better TTS WER in these configurations.
Further schedule tuning improved the DAC results; these are not universal codec
rankings or measurements of Meta DACVAE. [Source](https://arxiv.org/html/2406.11427v2).

Irodori reports Joyo standard CER improving from 5.35 +/- 0.18 in v4 to
4.69 +/- 0.02 in v4.1, while noting weaker short-reference similarity than v3.
Content accuracy and identity transfer need not improve together.
[Source](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small).

No common evaluation was identified that covers these families with identical
English texts, references, ASR normalization, speaker encoder, DNSMOS implementation,
and candidate-selection policy. There is no defensible overall WER/CER/DNSMOS or
voice-cloning-success ranking from the surveyed evidence alone.

## Post-training and runtime comparisons

| Approach | Example | Interpretation |
|---|---|---|
| Distillation plus direct metric optimization | Original DMOSpeech | Reports distilling a 128-step teacher to four steps with intelligibility and speaker objectives. Not validated on this repository. |
| Teacher distillation and preference alignment | Fish S1-mini | Benefits from a larger teacher and corpus; not scratch-only evidence. |
| Task-specific supervised fine-tuning | Text-AB | Dubbing/dialogue specialization changes quality and transfer behavior. |
| Multiple candidates and automatic reranking | Text-AB | Selected-output improvements incur additional cost; distinct from single-sample quality. |

Sources: [DMOSpeech](https://proceedings.mlr.press/v267/li25ay.html),
[Fish S1-mini](https://huggingface.co/fishaudio/s1-mini),
[Text-AB](https://arxiv.org/html/2609.03992v1).

Echo reports generating 30 seconds in approximately 1.45 seconds on an A100.
This is not an end-to-end comparison on matched hardware, lengths, and sampling
settings. Network size, sampling steps, CFG branches, reference processing, and
codec decoding all affect latency. [Source](https://jordandarefsky.com/blog/2025/echo/).

## Position of this repository

| Dimension | Local baseline |
|---|---|
| Trainable parameters | Tiny 13.53M; Small 44.36M. |
| Including frozen codec | Approximately 121M / 152M. |
| Pretrained training component | DACVAE only; text, reference, duration, and generator initialize from scratch. |
| Reference conditioning | Full sequence plus framewise MLP and mean pooling. |
| Corpus information | Four million rows; total audio hours and speaker distribution unspecified. |
| Quality evidence | Synthetic correctness checks, not a trained speech checkpoint or measured cloning quality. |

Four million rows correspond to approximately 5,556 hours at five seconds per row,
or 11,111 hours at ten seconds per row. These are illustrative conversions, not
estimates of the actual corpus. See [architecture audit](architecture-audit.md) for
the verified local implementation.

Text-AB and LAION adaptations are closest in codec family; Echo and Irodori are
strong architecture comparisons; original DMOSpeech is the closest post-training
comparison. The much smaller scratch-trained generator distinguishes this project.
Its intelligibility and unseen-speaker fidelity remain unmeasured.
