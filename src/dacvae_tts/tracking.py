"""Optional Weights & Biases tracking. Local JSONL logs stay the source of truth; this mirrors them."""

import os


class Tracker:
    """No-op unless `enabled`; then every `log` call also goes to a W&B run.

    The run name is the output directory's name, the config is the resolved training config, and
    `resume="allow"` with a stable id (run name) lets `--resume` continue the same W&B run.
    """

    def __init__(self, enabled, project, run_name, config, group=None, resume_id=None, tags=()):
        self.run = None
        if not enabled:
            return
        try:
            import wandb
        except ImportError as error:
            raise ImportError("W&B tracking needs the optional dependency: pip install wandb") from error
        self.run = wandb.init(
            project=project,
            name=run_name,
            id=resume_id or run_name,
            resume="allow",
            group=group,
            config=config,
            tags=list(tags),
            dir=os.environ.get("WANDB_DIR"),
        )

    def log(self, record, step=None, prefix=""):
        if self.run is None:
            return
        payload = {}
        for key, value in record.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                payload[prefix + key] = value
        if payload:
            self.run.log(payload, step=step)

    def audio(self, name, path, caption, sample_rate, step=None):
        if self.run is None:
            return
        import wandb

        self.run.log({name: wandb.Audio(str(path), caption=caption, sample_rate=sample_rate)}, step=step)

    def finish(self):
        if self.run is not None:
            self.run.finish()
            self.run = None
