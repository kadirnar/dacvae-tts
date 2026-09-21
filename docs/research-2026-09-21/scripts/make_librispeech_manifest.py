"""Build the small LibriSpeech manifest used by ab_train.py (explicit speaker-disjoint splits).

usage: python make_librispeech_manifest.py LIBRISPEECH_ALL_DIR OUT_DIR
LIBRISPEECH_ALL_DIR is the `all/` folder of the Hugging Face `openslr/librispeech_asr` Parquet export
(`test.clean/0000.parquet` and `train.clean.100/0000.parquet` are read). Then:

  dacvae-tts prepare --manifest OUT_DIR --output CACHE/part --device cuda --max-seconds 15
  dacvae-tts merge --inputs CACHE/part --output CACHE/merged
"""

import sys

import pyarrow as pa
import pyarrow.parquet as pq

root, out = sys.argv[1], sys.argv[2]
columns = ["id", "audio", "text", "speaker_id", "split", "language"]

test = pq.read_table(f"{root}/test.clean/0000.parquet")
speakers = sorted(set(test.column("speaker_id").to_pylist()))
validation = set(speakers[-8:])  # eight held-out speakers
split = ["val" if s in validation else "train" for s in test.column("speaker_id").to_pylist()]
test = test.append_column("split", pa.array(split)).append_column("language", pa.array(["en"] * len(split)))
pq.write_table(test.select(columns), f"{out}/test_clean.parquet", row_group_size=128)

train = pq.read_table(f"{root}/train.clean.100/0000.parquet")
rows = train.num_rows
train = train.append_column("split", pa.array(["train"] * rows)).append_column(
    "language", pa.array(["en"] * rows)
)
pq.write_table(train.select(columns), f"{out}/train_clean_100_part0.parquet", row_group_size=128)
print("validation speakers:", sorted(validation))
