import json
from types import SimpleNamespace

import pytest

from dacvae_tts.comparison import paired_comparison
from dacvae_tts.data import LatentDataset
from dacvae_tts.prepare import merge, source_rows


def test_parquet_partitioning_is_complete_and_disjoint(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    for groupsize in (2, 50):
        path = tmp_path / f"data-{groupsize}.parquet"
        pq.write_table(
            pa.Table.from_pylist([{"text": str(i)} for i in range(31)]), path, row_group_size=groupsize
        )
        seen = []
        for rank in range(8):
            seen.extend(row["text"] for row in source_rows(path, rank, 8))
        assert len(seen) == len(set(seen)) == 31
    # Generated IDs include filenames when importing a multi-file corpus.
    rows = [row for rank in range(8) for row in source_rows(tmp_path, rank, 8)]
    assert len({row["id"] for row in rows}) == 62


def test_merge_duplicate_partition_recomputes_statistics(cache, tmp_path):
    output = tmp_path / "merged"
    merge(SimpleNamespace(inputs=[cache, cache], output=output))
    data = LatentDataset(output)
    assert len(data) == 12
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["duplicates_removed"] == 36


def test_comparison_gate_and_alignment():
    before, after = [], []
    for i in range(4):
        row = {
            "uid": str(i),
            "reference_uid": "ref",
            "seed": 1,
            "text": "some words",
            "speaker": str(i),
            "wer": 0.2,
            "cer": 0.1,
            "word_edits": 2,
            "words": 10,
            "char_edits": 3,
            "chars": 30,
            "dnsmos_ovrl": 3.0,
            "speaker_similarity": 0.7,
        }
        before.append(row)
        after.append(
            {
                **row,
                "wer": 0.1,
                "cer": 1 / 30,
                "word_edits": 1,
                "char_edits": 1,
                "dnsmos_ovrl": 3.5,
                "speaker_similarity": 0.75,
            }
        )
    result = paired_comparison(before, after, samples=20)
    assert result["metric_gate_passed"]
    with pytest.raises(ValueError, match="identical"):
        paired_comparison(before, after[:-1], samples=20)
