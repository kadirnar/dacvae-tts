"""Find one voice under many diarization labels: cluster per-label speaker-embedding centroids across
episodes, report val/test labels that leak into train and labels that mix voices.

Turkish speaker labels are per-episode diarization labels (`<episode>_speaker_k`); a recurring host gets a
new label in every episode, and the label-hash split can put that voice in both train and test. For up to
--per-label utterances of every label this script takes the original recording when its path still exists,
else decodes the cached DACVAE latents (the whole Turkish cache stores embedded audio), resamples to 16 kHz,
embeds with a pluggable speaker model, L2-normalizes, averages per label and runs average-linkage
clustering of the label centroids at a cosine threshold. Outputs in --output:

  clusters.json            label -> cluster id (0 = largest cluster)
  leakage.json             val/test labels with centroid cosine >= --leak-threshold to some train label,
                           highest first; its keys feed `make-cases --exclude-speakers` and
                           `monitor.py/eval_sentences.py --exclude-speakers`
  inconsistent_labels.json labels whose utterances disagree with their own leave-one-out centroid
                           (likely diarization errors), with the outlier uids
  outlier_uids.json        those uids as a list for `merge --drop-uids`
  split_map.json           label -> split with every cluster (and --split-key group) in one split, for
                           `merge --split-map`
  summary.json             counts, nearest-train cosine quantiles and the largest clusters
  embeddings.npz           per-uid embeddings; a re-run with another threshold/policy does no audio work

Thresholds must be calibrated per embedder and corpus: listen to label pairs near the threshold and look at
the nearest-train quantiles in summary.json. Published cross-utterance speaker gates use 0.6 (HiFiTTS-2),
0.65 (WenetSpeech4TTS) and 0.7 (VoxCPM2) with WavLM/ECAPA embeddings.

  pip install speechbrain   # for the default ECAPA embedder; --embedder wavlm needs nothing new
  python scripts/speaker_clusters.py --cache data/tr55/merged --output outputs/speaker-clusters \\
      --per-label 8 --threshold 0.65 --split-key '^(.+)_speaker_\\d+$' --device cuda
  python scripts/speaker_clusters.py --cache data/tr55/merged --output outputs/speaker-clusters --threshold 0.7
"""

import argparse

from dacvae_tts.speakers import EMBEDDERS, cluster_speakers


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", required=True, help="Merged latent cache (index.sqlite + shards)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--embeddings", help="Embedding cache (default: OUTPUT/embeddings.npz)")
    parser.add_argument("--per-label", type=int, default=8, help="Utterances embedded per label; 0 = all")
    parser.add_argument("--seed", type=int, default=42, help="Utterance subsampling and --split-policy hash seed")
    parser.add_argument(
        "--embedder",
        default="ecapa",
        help=f"ecapa ({EMBEDDERS['ecapa']}, needs speechbrain), wavlm ({EMBEDDERS['wavlm']}) or "
        "package.module:factory returning a 16 kHz waveform -> vector callable",
    )
    parser.add_argument("--embedder-model", help="Override the built-in embedder's model id")
    parser.add_argument("--device", default="cuda", help="Device of the embedder and of the DACVAE decoder")
    parser.add_argument(
        "--source",
        choices=["auto", "audio", "latents"],
        default="auto",
        help="auto: original audio when its path exists, else decoded latents; latents: always decode "
        "(one channel for every row)",
    )
    parser.add_argument("--max-seconds", type=float, default=10.0, help="Embed at most this much of each clip")
    parser.add_argument("--threshold", type=float, default=0.65, help="Average-linkage cosine threshold")
    parser.add_argument("--leak-threshold", type=float, help="Cosine for leakage.json (default: --threshold)")
    parser.add_argument(
        "--outlier-threshold", type=float, default=0.5, help="Utterance vs own-label cosine below which it is an outlier"
    )
    parser.add_argument(
        "--outlier-fraction", type=float, default=0.25, help="Outlier share that flags a label as inconsistent"
    )
    parser.add_argument("--min-utterances", type=int, default=3, help="Consistency is judged from this many up")
    parser.add_argument(
        "--split-key",
        help="Regex extracting the episode/program key (group `key`, else group 1): split_map.json keeps each "
        "key group in one split and summary.json counts clusters that span keys",
    )
    parser.add_argument(
        "--split-policy",
        choices=["keep-train", "hash"],
        default="keep-train",
        help="keep-train: a cluster touching train goes to train (existing checkpoints stay evaluable); "
        "hash: a fresh 98/1/1 split per cluster (retrain)",
    )
    parser.add_argument(
        "--max-component", type=int, default=5000, help="Larger single-linkage components skip average linkage"
    )
    parser.add_argument("--save-every", type=int, default=1000, help="Checkpoint the embedding cache every N rows")
    cluster_speakers(parser.parse_args())


if __name__ == "__main__":
    main()
