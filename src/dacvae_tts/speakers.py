"""Speaker identity beyond diarization labels: split keys and split maps for prepare/merge, and the
cross-episode clustering of per-label speaker embeddings behind `scripts/speaker_clusters.py`.

Web corpora label speakers per episode (`<episode>_speaker_k`): a recurring host gets a new label in every
episode, so hashing labels into splits (`speaker_split`) can put one voice in both train and test and inflate
"unseen speaker" WER and SIM. Both remedies here are opt-in; without them every split stays the label hash.

* A split key (`--split-key REGEX`) hashes a key extracted from the label, such as the episode or program
  prefix, so all labels of an episode/program fall into one split and whole episodes are held out.
* A split map (`merge --split-map FILE`) assigns splits explicitly. `scripts/speaker_clusters.py` builds one
  from embedding clusters so that every label of a cross-episode cluster shares one split.

Cosine thresholds depend on the embedder, the channel and the corpus and must be calibrated (listen to label
pairs around the threshold). Published cross-utterance speaker gates use 0.6 (HiFiTTS-2), 0.65
(WenetSpeech4TTS) and 0.7 (VoxCPM2) with WavLM/ECAPA embeddings. Averaging several utterances removes
noise, so same-speaker label centroids score higher than single utterance pairs: the same number flags more
label pairs here, which errs on the safe side for leakage.
"""

import hashlib
import importlib
import json
import math
import os
import re
import sqlite3
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import numpy as np

from .data import speaker_split

SPLITS = ("train", "val", "test")
EMBEDDING_RATE = 16000
EMBEDDERS = {"ecapa": "speechbrain/spkrec-ecapa-voxceleb", "wavlm": "microsoft/wavlm-base-plus-sv"}


# --------------------------------------------------------------------------- split keys and split maps


def compile_split_key(pattern):
    """Validate a --split-key regex once, at the boundary, so a typo fails before any audio is read."""
    if pattern is None:
        return None
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"Invalid --split-key regex {pattern!r}: {exc}") from exc


def split_group(speaker, split_key=None):
    """The string the split hash uses: the label itself by default, else the part --split-key extracts
    (the named group `key`, else the first group, else the whole match of `re.search`).

    A label the regex does not match is an error rather than a silent fallback to the label: a mistyped
    regex would otherwise quietly reproduce the per-label split it was meant to replace.
    """
    if split_key is None:
        return speaker
    regex = compile_split_key(split_key) if isinstance(split_key, str) else split_key
    match = regex.search(speaker)
    if match is None:
        raise ValueError(f"Speaker label {speaker!r} does not match --split-key {regex.pattern!r}")
    if "key" in regex.groupindex:
        key = match.group("key")
    else:
        key = match.group(1) if regex.groups else match.group(0)
    if not key:
        raise ValueError(f"--split-key {regex.pattern!r} extracts an empty key from {speaker!r}")
    return key


def load_split_map(path):
    """A JSON object {speaker label or split key: split}; values are validated here, not per row."""
    mapping = json.loads(Path(path).read_text())
    if not isinstance(mapping, dict) or not all(isinstance(key, str) for key in mapping):
        raise ValueError("--split-map must be a JSON object {speaker label or split key: split}")
    invalid = sorted({str(value) for value in mapping.values() if value not in SPLITS})
    if invalid:
        raise ValueError(f"--split-map values must be train, val or test; found {invalid[:5]}")
    return mapping


def assign_split(speaker, split, seed, split_key=None, split_map=None):
    """Merge-time split of one label: its --split-map entry, else the entry of its split key, else the
    split-key hash; with neither option matching, the partition's own split is kept."""
    if split_map and speaker in split_map:
        return split_map[speaker]
    if split_key is None:
        return split
    key = split_group(speaker, split_key)
    return split_map[key] if split_map and key in split_map else speaker_split(key, seed)


def check_split_groups(db, split_key):
    """Every label sharing a split key must share one split (an episode/program is never divided)."""
    groups = defaultdict(set)
    for speaker, split in db.execute("SELECT DISTINCT speaker, split FROM samples"):
        groups[split_group(speaker, split_key)].add(split)
    divided = sorted(key for key, splits in groups.items() if len(splits) > 1)
    if divided:
        raise ValueError(
            f"Split-key group appears across splits: {divided[0]} ({len(divided)} groups); "
            "map the key itself in --split-map or build the map with the same --split-key"
        )


def read_speaker_list(path):
    """Speaker labels from a JSON list, the keys of a JSON object (e.g. leakage.json) or one label per line."""
    text = Path(path).read_text()
    if text.lstrip().startswith(("[", "{")):
        value = json.loads(text)
        labels = list(value) if isinstance(value, (list, dict)) else None
        if labels is None or not all(isinstance(label, str) for label in labels):
            raise ValueError(f"{path}: expected a JSON list of labels or an object keyed by label")
    else:
        labels = [line.strip() for line in text.splitlines() if line.strip()]
    return set(labels)


# --------------------------------------------------------------------------- clustering (numpy only)


def normalize_rows(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or not len(vectors) or not np.isfinite(vectors).all():
        raise ValueError("Embeddings must be a nonempty finite [N, D] array")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("Zero embedding vector")
    return vectors / norms


def label_centroids(embeddings, labels):
    """Sorted label names, their L2-normalized mean embeddings and the row indices of each label."""
    embeddings = normalize_rows(embeddings)
    members = defaultdict(list)
    for index, label in enumerate(labels):
        members[label].append(index)
    names = sorted(members)
    centroids = normalize_rows(np.stack([embeddings[members[name]].mean(0) for name in names]))
    return names, centroids, [members[name] for name in names]


def _find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def _union(parent, i, j):
    i, j = _find(parent, i), _find(parent, j)
    if i != j:
        parent[max(i, j)] = min(i, j)


def threshold_components(vectors, threshold):
    """Connected components of the graph `cosine >= threshold` (single linkage), in row blocks so memory
    stays O(block * N) for tens of thousands of labels."""
    vectors = np.asarray(vectors, dtype=np.float32)
    count = len(vectors)
    parent = list(range(count))
    step = max(1, 2**24 // max(count, 1))
    for start in range(0, count, step):
        # float32 blocks: a 1e-6 margin keeps every float64 pair >= threshold inside one component.
        rows, cols = np.nonzero(vectors[start : start + step] @ vectors.T >= threshold - 1e-6)
        rows = rows + start
        for i, j in zip(rows[cols > rows].tolist(), cols[cols > rows].tolist()):
            _union(parent, i, j)
    return np.array([_find(parent, i) for i in range(count)])


def average_linkage(similarity, threshold):
    """UPGMA on a similarity matrix: repeatedly merge the two clusters with the highest mean pairwise
    similarity while it is >= threshold. Unlike single linkage it does not chain A~B~C into one voice when
    A and C are dissimilar. Returns a representative index per row."""
    size = len(similarity)
    matrix = np.array(similarity, dtype=np.float64)
    np.fill_diagonal(matrix, -np.inf)
    counts = np.ones(size)
    assignment = np.arange(size)
    while size > 1:
        i, j = divmod(int(np.argmax(matrix)), size)
        if matrix[i, j] < threshold:
            break
        # Lance-Williams update for the average of all pairwise similarities.
        merged = (counts[i] * matrix[i] + counts[j] * matrix[j]) / (counts[i] + counts[j])
        matrix[i, :], matrix[:, i] = merged, merged
        matrix[j, :], matrix[:, j] = -np.inf, -np.inf
        matrix[i, i] = -np.inf
        counts[i] += counts[j]
        assignment[assignment == j] = i
    return assignment


def cluster_labels(centroids, threshold, max_component=5000):
    """Average-linkage clusters of label centroids at a cosine threshold.

    Average-linkage clusters at threshold t never cross single-linkage components at t (a mean >= t needs
    one pair >= t), so the exact linkage runs per component. A component larger than `max_component`
    (a suspiciously low threshold) keeps its single-linkage grouping instead of an O(n^3) loop; the count
    is returned so the summary can flag it. Cluster ids are ordered by size, then by first member.
    """
    centroids = normalize_rows(centroids)
    if not 0 < threshold <= 1:
        raise ValueError("Cosine threshold must lie in (0, 1]")
    roots = threshold_components(centroids, threshold)
    representative = roots.copy()
    fallbacks = 0
    for root in np.unique(roots):
        members = np.flatnonzero(roots == root)
        if len(members) < 2:
            continue
        if len(members) > max_component:
            fallbacks += 1
            continue
        local = average_linkage(centroids[members] @ centroids[members].T, threshold)
        representative[members] = members[local]
    groups = defaultdict(list)
    for index, rep in enumerate(representative.tolist()):
        groups[rep].append(index)
    order = sorted(groups.values(), key=lambda items: (-len(items), items[0]))
    clusters = np.empty(len(centroids), dtype=np.int64)
    for cluster, items in enumerate(order):
        clusters[items] = cluster
    return clusters, fallbacks


def nearest_train(names, centroids, splits):
    """For each val/test label: its highest centroid cosine to any train label and that label."""
    centroids = normalize_rows(centroids)
    train = np.flatnonzero(np.asarray(splits) == "train")
    held = np.flatnonzero(np.asarray(splits) != "train")
    result = {}
    if not len(train) or not len(held):
        return result
    step = max(1, 2**24 // len(train))
    for start in range(0, len(held), step):
        block = held[start : start + step]
        similarity = centroids[block] @ centroids[train].T
        for offset, (row, column) in enumerate(zip(block.tolist(), similarity.argmax(1).tolist())):
            result[names[row]] = (float(similarity[offset, column]), names[train[column]])
    return result


def leakage(names, centroids, splits, threshold, clusters=None):
    """val/test labels whose centroid cosine to some train label centroid is >= threshold, highest first."""
    split_of = dict(zip(names, splits))
    train_clusters = set()
    if clusters is not None:
        train_clusters = {int(c) for c, s in zip(clusters, splits) if s == "train"}
        cluster_of = dict(zip(names, (int(c) for c in clusters)))
    leaked = {}
    for label, (score, train_label) in nearest_train(names, centroids, splits).items():
        if score >= threshold:
            entry = {"split": split_of[label], "score": round(score, 6), "nearest_train_label": train_label}
            if clusters is not None:
                entry.update(cluster=cluster_of[label], cluster_has_train=cluster_of[label] in train_clusters)
            leaked[label] = entry
    return dict(sorted(leaked.items(), key=lambda item: (-item[1]["score"], item[0])))


def inconsistent_labels(
    embeddings, labels, uids, outlier_threshold=0.5, outlier_fraction=0.25, min_utterances=3
):
    """Labels whose utterances disagree with their own centroid: likely diarization errors (two voices
    under one label). Each utterance is compared with the leave-one-out centroid of the label's other
    utterances, since including itself would inflate the score."""
    if min_utterances < 2:
        raise ValueError("Consistency needs at least two utterances per label")
    embeddings = normalize_rows(embeddings)
    members = defaultdict(list)
    for index, label in enumerate(labels):
        members[label].append(index)
    result = {}
    for label in sorted(members):
        rows = members[label]
        if len(rows) < min_utterances:
            continue
        vectors = embeddings[rows]
        others = vectors.sum(0, keepdims=True) - vectors
        cosine = (vectors * others).sum(1) / np.maximum(np.linalg.norm(others, axis=1), 1e-12)
        outliers = np.flatnonzero(cosine < outlier_threshold)
        if len(outliers) and len(outliers) / len(rows) >= outlier_fraction:
            result[label] = {
                "utterances": len(rows),
                "mean_cosine": round(float(cosine.mean()), 6),
                "min_cosine": round(float(cosine.min()), 6),
                "outlier_fraction": round(len(outliers) / len(rows), 6),
                "outliers": [{"uid": uids[rows[i]], "cosine": round(float(cosine[i]), 6)} for i in outliers],
            }
    return result


def split_map_from_clusters(names, clusters, splits, rows=None, policy="keep-train", seed=42, split_key=None):
    """Give every label of a component one split; a component is a cluster, joined with all labels sharing
    its split key when one is given (so neither a voice nor an episode is divided).

    `keep-train`: a component with any train label goes to train (existing checkpoints stay evaluable on
    the shrunken held-out sets, since no held-out label was ever trained on); otherwise the held-out split
    with more rows wins (test on ties). `hash`: a fresh 98/1/1 split hashing the smallest split key (or
    label) of the component, which reproduces `--split-key` (or the default split) for components that did
    not merge across keys.

    With a split key the map also holds one entry per key, so `merge --split-key --split-map` sends labels
    the analysis never saw (singletons dropped by an earlier merge, failed embeddings) to their episode's
    split instead of the plain key hash, which could otherwise divide the episode.
    """
    if policy not in {"keep-train", "hash"}:
        raise ValueError("split policy must be keep-train or hash")
    rows = rows or {}
    parent = list(range(len(names)))
    first = {}
    for index, (name, cluster) in enumerate(zip(names, clusters)):
        keys = [("cluster", int(cluster))] + ([("key", split_group(name, split_key))] if split_key else [])
        for key in keys:
            if key in first:
                _union(parent, index, first[key])
            else:
                first[key] = index
    components = defaultdict(list)
    for index in range(len(names)):
        components[_find(parent, index)].append(index)
    mapping = {}
    for members in components.values():
        member_splits = [splits[i] for i in members]
        if policy == "hash":
            split = speaker_split(min(split_group(names[i], split_key) for i in members), seed)
        elif "train" in member_splits:
            split = "train"
        else:
            totals = Counter()
            for i in members:
                totals[splits[i]] += rows.get(names[i], 1)
            split = max(("test", "val"), key=lambda s: totals[s])
        for i in members:
            mapping[names[i]] = split
    if split_key:
        for name in names:
            key = split_group(name, split_key)
            if mapping.setdefault(key, mapping[name]) != mapping[name]:
                raise ValueError(f"Split key {key!r} equals a speaker label with another split")
    return dict(sorted(mapping.items()))


# --------------------------------------------------------------------------- audio and embedders


class RowAudio:
    """16 kHz mono waveform of a cache row: the original recording when its path exists (unless
    source='latents'), else the stored latents decoded with the frozen DACVAE. Shards hold the raw float16
    posterior means (LatentDataset normalizes on read), so they are decoded without de-normalization."""

    def __init__(self, cache, source="auto", max_seconds=10.0, device="cpu", decoder=None):
        if source not in {"auto", "audio", "latents"} or not max_seconds > 0:
            raise ValueError("source must be auto, audio or latents; max-seconds must be positive")
        self.cache = Path(cache).resolve()
        self.meta = json.loads((self.cache / "metadata.json").read_text())
        self.channels, self.sample_rate = self.meta["latent_dim"], self.meta["sample_rate"]
        self.hop_length = self.meta["hop_length"]
        self.source, self.max_seconds, self.device = source, max_seconds, device
        self._decoder, self._maps = decoder, OrderedDict()

    def decoder(self):
        if self._decoder is None:  # the codec is loaded only if some row lacks its original audio
            from .codec import Codec, check_compatibility

            codec = Codec(self.meta["checkpoint"], self.device, loudness=self.meta.get("loudness_lufs"))
            check_compatibility(codec.metadata, self.meta)
            self._decoder = codec.decode
        return self._decoder

    def latents(self, row):
        shard = row["shard"]
        if shard not in self._maps:
            self._maps[shard] = np.memmap(shard, dtype="<f2", mode="r").reshape(-1, self.channels)
            if len(self._maps) > 32:
                self._maps.popitem(last=False)
        frames = min(row["frames"], max(1, round(self.max_seconds * self.sample_rate / self.hop_length)))
        z = np.array(self._maps[shard][row["offset"] : row["offset"] + frames], dtype=np.float32)
        if z.shape != (frames, self.channels) or not np.isfinite(z).all():
            raise ValueError(f"Corrupt/truncated latent record: {row['uid']}")
        return z

    def __call__(self, row):
        from scipy.signal import resample_poly

        from .codec import read_audio

        path = Path(row["audio"])
        if self.source != "latents" and path.is_file():
            audio, kind = read_audio(path, EMBEDDING_RATE).numpy(), "audio"
        elif self.source == "audio":
            raise ValueError(f"Original audio missing: {row['audio']}")
        else:
            import torch

            z = self.latents(row)
            wave = np.asarray(self.decoder()(torch.from_numpy(z)), dtype=np.float32)
            wave = wave[: min(len(wave), row["samples"], len(z) * self.hop_length)]
            factor = math.gcd(self.sample_rate, EMBEDDING_RATE)
            audio = resample_poly(wave, EMBEDDING_RATE // factor, self.sample_rate // factor)
            kind = "latents"
        audio = np.asarray(audio[: int(self.max_seconds * EMBEDDING_RATE)], dtype=np.float32)
        if len(audio) < EMBEDDING_RATE // 10 or not np.isfinite(audio).all():
            raise ValueError(f"Too short or nonfinite audio for a speaker embedding: {row['uid']}")
        return audio, kind


def make_embedder(name="ecapa", device="cpu", model=None):
    """A callable 16 kHz float32 waveform -> 1-D speaker embedding. `ecapa` (SpeechBrain, installed
    separately) and `wavlm` (transformers, already a dependency) are built in; `package.module:factory`
    plugs in any other model as `factory(device=..., model=...)`. Heavy imports happen only here."""
    import torch

    if name == "ecapa":
        model = model or EMBEDDERS["ecapa"]
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            try:
                from speechbrain.pretrained import EncoderClassifier
            except ImportError as exc:
                raise ImportError(
                    "--embedder ecapa needs SpeechBrain (pip install speechbrain), or use --embedder wavlm"
                ) from exc
        savedir = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "speechbrain"
        classifier = EncoderClassifier.from_hparams(
            source=model, savedir=str(savedir / model.replace("/", "--")), run_opts={"device": device}
        )

        @torch.inference_mode()
        def embed(audio):
            vector = classifier.encode_batch(torch.from_numpy(audio)[None].to(device))
            return vector.reshape(-1).float().cpu().numpy()

        return embed
    if name == "wavlm":
        from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

        model = model or EMBEDDERS["wavlm"]
        extractor = AutoFeatureExtractor.from_pretrained(model)
        network = AutoModelForAudioXVector.from_pretrained(model).to(device).eval()

        @torch.inference_mode()
        def embed(audio):
            inputs = extractor(audio, sampling_rate=EMBEDDING_RATE, return_tensors="pt", padding=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            return network(**inputs).embeddings[0].float().cpu().numpy()

        return embed
    module, _, factory = name.partition(":")
    if not module or not factory:
        raise ValueError("--embedder must be ecapa, wavlm or package.module:factory")
    return getattr(importlib.import_module(module), factory)(device=device, model=model)


def embedding_identity(args):
    """What the cached vectors depend on; a mismatch refuses reuse instead of mixing embedding spaces."""
    name = getattr(args, "embedder", "ecapa")
    return json.dumps(
        {
            "embedder": name,
            "model": getattr(args, "embedder_model", None) or EMBEDDERS.get(name),
            "source": getattr(args, "source", "auto"),
            "max_seconds": getattr(args, "max_seconds", 10.0),
            "sample_rate": EMBEDDING_RATE,
        },
        sort_keys=True,
    )


def load_embeddings(path, identity):
    path = Path(path)
    if not path.exists():
        return {}
    with np.load(path) as data:
        if str(data["identity"]) != identity:
            raise ValueError(f"{path} holds embeddings of {data['identity']}; use another --embeddings file")
        return {
            str(uid): (vector, str(kind))
            for uid, vector, kind in zip(data["uids"], data["embeddings"], data["sources"])
        }


def save_embeddings(path, identity, table):
    if not table:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    uids = sorted(table)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as stream:
        np.savez(
            stream,
            identity=np.array(identity),
            uids=np.array(uids),
            embeddings=np.stack([table[uid][0] for uid in uids]).astype(np.float32),
            sources=np.array([table[uid][1] for uid in uids]),
        )
    os.replace(temporary, path)  # an interrupted save never truncates the previous cache


# --------------------------------------------------------------------------- cache-level pipeline


def select_rows(db, per_label=8, seed=42):
    """Up to `per_label` rows of every label, chosen by a seeded uid hash (independent of row order)."""
    fields = ("uid", "speaker", "split", "audio", "shard", "offset", "frames", "samples")
    selected, group, previous = [], [], None

    def flush():
        group.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['uid']}".encode()).digest())
        selected.extend(group[:per_label] if per_label else group)
        group.clear()

    for row in db.execute(f"SELECT {','.join(fields)} FROM samples ORDER BY speaker, uid"):
        row = dict(zip(fields, row))
        if row["speaker"] != previous and group:
            flush()
        group.append(row)
        previous = row["speaker"]
    if group:
        flush()
    return selected


def _quantiles(values):
    if not values:
        return None
    levels = (("p50", 0.5), ("p90", 0.9), ("p99", 0.99), ("max", 1.0))
    return {name: round(float(np.quantile(np.asarray(values), q)), 6) for name, q in levels}


def cluster_speakers(args, embed=None, load=None):
    """Embed up to --per-label rows of every label, cluster label centroids across episodes and write
    clusters.json, leakage.json, inconsistent_labels.json, outlier_uids.json, split_map.json and
    summary.json into --output. Embeddings are cached per uid in an .npz, so re-running with another
    threshold, policy or split key does no audio work."""
    cache, output = Path(args.cache).resolve(), Path(args.output).resolve()
    threshold = getattr(args, "threshold", 0.65)
    leak_threshold = getattr(args, "leak_threshold", None) or threshold
    per_label, seed = getattr(args, "per_label", 8), getattr(args, "seed", 42)
    split_key = compile_split_key(getattr(args, "split_key", None))
    if per_label < 0 or not 0 < leak_threshold <= 1:
        raise ValueError("per-label must be nonnegative and the leak threshold in (0, 1]")
    output.mkdir(parents=True, exist_ok=True)
    embeddings_path = Path(getattr(args, "embeddings", None) or output / "embeddings.npz")
    identity = embedding_identity(args)
    db = sqlite3.connect(f"file:{cache / 'index.sqlite'}?mode=ro", uri=True)
    try:
        label_rows, label_split = Counter(), {}
        query = "SELECT speaker, split, count(*) FROM samples GROUP BY speaker, split"
        for speaker, split, count in db.execute(query):
            if label_split.setdefault(speaker, split) != split:
                raise ValueError(f"Speaker appears across splits: {speaker}; repair the cache first")
            label_rows[speaker] += count
        rows = select_rows(db, per_label, seed)
    finally:
        db.close()
    table = load_embeddings(embeddings_path, identity)
    pending = sorted((r for r in rows if r["uid"] not in table), key=lambda r: (r["shard"], r["offset"]))
    failures = []
    if pending:
        from tqdm import tqdm

        device = getattr(args, "device", "cpu")
        name, model = getattr(args, "embedder", "ecapa"), getattr(args, "embedder_model", None)
        embed = embed or make_embedder(name, device, model)
        source, max_seconds = getattr(args, "source", "auto"), getattr(args, "max_seconds", 10.0)
        load = load or RowAudio(cache, source, max_seconds, device)
        save_every = getattr(args, "save_every", 1000)
        for done, row in enumerate(tqdm(pending, desc="Speaker embeddings"), 1):
            try:
                audio, kind = load(row)
                vector = np.asarray(embed(audio), dtype=np.float32).reshape(-1)
                if not np.isfinite(vector).all() or not np.linalg.norm(vector) > 0:
                    raise ValueError("nonfinite or zero embedding")
            except (ValueError, OSError, RuntimeError) as exc:
                failures.append({"uid": row["uid"], "error": str(exc)[:200]})
                if len(failures) == done >= 20:  # a systematic error (codec, model), not a bad clip
                    raise ValueError(f"The first {done} rows all failed; last error: {exc}") from exc
                continue
            table[row["uid"]] = (vector, kind)
            if save_every and done % save_every == 0:
                save_embeddings(embeddings_path, identity, table)
        save_embeddings(embeddings_path, identity, table)
    embedded = [r for r in rows if r["uid"] in table]
    if not embedded:
        raise ValueError("No speaker embeddings; inspect the failures in summary.json")
    vectors = np.stack([table[r["uid"]][0] for r in embedded])
    labels, uids = [r["speaker"] for r in embedded], [r["uid"] for r in embedded]
    names, centroids, _ = label_centroids(vectors, labels)
    splits = [label_split[name] for name in names]
    clusters, fallbacks = cluster_labels(centroids, threshold, getattr(args, "max_component", 5000))
    leaked = leakage(names, centroids, splits, leak_threshold, clusters)
    inconsistent = inconsistent_labels(
        vectors,
        labels,
        uids,
        getattr(args, "outlier_threshold", 0.5),
        getattr(args, "outlier_fraction", 0.25),
        getattr(args, "min_utterances", 3),
    )
    policy = getattr(args, "split_policy", "keep-train")
    mapping = split_map_from_clusters(names, clusters, splits, label_rows, policy, seed, split_key)
    changed = Counter()
    for name in names:
        if mapping[name] != label_split[name]:
            changed[f"{label_split[name]}->{mapping[name]}"] += label_rows[name]
    members = defaultdict(list)
    for name, cluster in zip(names, clusters.tolist()):
        members[cluster].append(name)
    nearest = nearest_train(names, centroids, splits)
    held_out = {}
    for split in ("val", "test"):
        total = sum(s == split for s in splits)
        count = sum(entry["split"] == split for entry in leaked.values())
        held_out[split] = {"labels": total, "leaked": count, "clean": total - count}
    summary = {
        "cache": str(cache),
        "embedding": json.loads(identity),
        "embeddings_file": str(embeddings_path),
        "per_label": per_label,
        "seed": seed,
        "threshold": threshold,
        "leak_threshold": leak_threshold,
        "split_key": split_key.pattern if split_key else None,
        "labels": len(label_split),
        "embedded_labels": len(names),
        "embedded_utterances": len(embedded),
        "embedding_sources": dict(Counter(table[u][1] for u in uids)),
        "embedding_failures": len(failures),
        "failures": failures[:20],
        "clusters": len(members),
        "multi_label_clusters": sum(len(items) > 1 for items in members.values()),
        "largest_cluster_labels": max(map(len, members.values())),
        "single_linkage_fallback_components": fallbacks,
        "held_out": held_out,
        "nearest_train_cosine": _quantiles([score for score, _ in nearest.values()]),
        "inconsistent_labels": len(inconsistent),
        "outlier_utterances": sum(len(entry["outliers"]) for entry in inconsistent.values()),
        "split_map": {
            "policy": policy,
            "labels_changed": sum(mapping[n] != label_split[n] for n in names),
            "rows_changed": dict(sorted(changed.items())),
        },
        "largest_clusters": [
            {
                "cluster": cluster,
                "labels": len(items),
                "splits": dict(Counter(label_split[n] for n in items)),
                "examples": items[:5],
            }
            for cluster, items in sorted(members.items())[:10]
            if len(items) > 1
        ],
        "calibration": "cosine thresholds are embedder/corpus specific: listen to pairs near the threshold "
        "(published gates: HiFiTTS-2 0.6, WenetSpeech4TTS 0.65, VoxCPM2 0.7)",
    }
    if split_key:
        keys = {n: split_group(n, split_key) for n in names}
        summary["cross_key_clusters"] = sum(len({keys[n] for n in items}) > 1 for items in members.values())
    outputs = {
        "clusters.json": {name: int(cluster) for name, cluster in zip(names, clusters)},
        "leakage.json": leaked,
        "inconsistent_labels.json": inconsistent,
        "outlier_uids.json": sorted(o["uid"] for entry in inconsistent.values() for o in entry["outliers"]),
        "split_map.json": mapping,
        "summary.json": summary,
    }
    for name, value in outputs.items():
        (output / name).write_text(json.dumps(value, indent=1, ensure_ascii=False))
    brief = (
        "labels embedded_labels embedded_utterances embedding_failures clusters multi_label_clusters "
        "largest_cluster_labels cross_key_clusters held_out nearest_train_cosine inconsistent_labels split_map"
    )
    print(json.dumps({key: summary[key] for key in brief.split() if key in summary}, indent=1))
    return summary
