"""Write a config that differs from a base config in the given fields only, and check that it loads.

  python scripts/gpu/derive_config.py configs/nano_tr_w512.yaml runs/configs/no-negatives.yaml \
      train.contrastive_mode=none
  python scripts/gpu/derive_config.py configs/experiments/tr_w512_wsd.yaml runs/configs/wsd.yaml \
      train.decay_cache=/data/tr55/hq

Values are parsed as YAML (`0.3`, `true`, `null`, `[0.2, 0.4]`, `chars`). The output starts with a comment naming
the base file and the overrides, so the run directory records where its configuration came from.
"""

import sys
from pathlib import Path

import yaml

from dacvae_tts.config import Config


def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    base, output, *overrides = argv
    config = yaml.safe_load(Path(base).read_text())
    for override in overrides:
        key, sep, value = override.partition("=")
        section, dot, field = key.partition(".")
        if not sep or not dot or section not in config:
            sys.exit(f"expected section.field=value with an existing section, got {override!r}")
        config[section][field] = yaml.safe_load(value)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = f"# Derived from {base}; changed: {', '.join(overrides) or 'nothing'}\n"
    output.write_text(header + yaml.safe_dump(config, sort_keys=False))
    Config.load(output)  # fails on an unknown field or an invalid combination
    print(output)


if __name__ == "__main__":
    main(sys.argv[1:])
