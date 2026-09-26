"""Flow-GRPO reward trend (#16): mean rewards per 50 updates and the monitor rows of runs/trc-grpo/grpo-log.jsonl."""

import json

rows = [json.loads(line) for line in open("/workspace/runs/trc-grpo/grpo-log.jsonl")]
steps = [r for r in rows if "reward" in r and "step" in r]
monitor = [r for r in rows if "monitor" in r or any(k.startswith("monitor") for k in r)]


def mean(chunk, key):
    return sum(r["reward"][key] for r in chunk) / len(chunk)


for low in range(0, len(steps), 50):
    chunk = steps[low : low + 50]
    kl = sum(r.get("kl_ref", 0) for r in chunk) / len(chunk)
    print(f"updates {chunk[0]['step']:4d}-{chunk[-1]['step']:4d}  cer {mean(chunk, 'cer'):.4f}  sim {mean(chunk, 'sim'):.4f}"
          f"  dnsmos {mean(chunk, 'dnsmos'):.3f}  utmos {mean(chunk, 'utmos'):.3f}  kl_ref {kl:.4f}")
for row in monitor:
    values = row.get("monitor") or row
    print("monitor", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in values.items()
                      if not isinstance(v, (dict, list))})
