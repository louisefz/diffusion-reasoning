"""Optional scalar-only W&B tracking; local training logs remain authoritative."""
import json
import warnings


def scalar_metrics(record):
    result = {"train/step": record["step"]}
    for key, value in record.items():
        if key == "validation":
            for group, scores in value.items():
                for metric, number in scores.items():
                    result[f"validation/{group}/{metric}"] = number
        elif key != "step" and isinstance(value, (int, float)):
            result[f"train/{key}"] = value
    return result


class Tracker:
    def __init__(self, args, metadata, start):
        self.run = None
        self.failed = False
        if args.wandb_mode == "disabled":
            return
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("W&B requested but not installed: python -m pip install -r requirements-wandb.txt") from exc
        # New segment on every process/resume: safe for offline runs and rollbacks.
        # Do not inherit WANDB_RUN_ID, which could append to an unrelated run.
        self.run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=args.wandb_name or f"{args.out.name}-from-{start}",
            group=args.wandb_group, id=wandb.util.generate_id(), resume="never",
            mode=args.wandb_mode, dir=str(args.out.resolve()),
            config={"model": metadata["config"], "parameters": metadata["parameters"],
                    "torch_version": metadata["torch_version"], "start_step": start,
                    "target_step": args.steps, "batch_size": args.batch_size,
                    "data_mode": args.data_mode, "seed": args.seed,
                    "task": args.task,
                    "objective": args.objective,
                    "learning_rate": metadata["effective_lr"],
                    "sampling_steps": args.sampling_steps, "eval_limit": args.eval_limit,
                    "dataset": metadata["manifest"]},
            settings=wandb.Settings(console="off", disable_git=True, save_code=False),
        )
        self.run.define_metric("train/step")
        self.run.define_metric("train/*", step_metric="train/step")
        self.run.define_metric("validation/*", step_metric="train/step")
        info = dict(id=self.run.id, mode=args.wandb_mode, project=args.wandb_project,
                    start_step=start, directory=self.run.dir)
        (args.out/f"wandb_from_{start}.json").write_text(json.dumps(info, indent=2)+"\n")

    def __enter__(self):
        return self

    def log(self, record):
        if self.run is not None and not self.failed:
            try:
                # Custom step axis permits train and validation records at same step.
                self.run.log(scalar_metrics(record))
            except Exception as exc:
                self.failed = True
                warnings.warn(f"W&B logging failed ({type(exc).__name__}); continuing with local metrics.jsonl")

    def __exit__(self, exc_type, exc, tb):
        if self.run is not None:
            try:
                self.run.finish(exit_code=1 if exc_type or self.failed else 0)
            except Exception as error:
                warnings.warn(f"W&B finish failed ({type(error).__name__}); local logs retained")
        return False
