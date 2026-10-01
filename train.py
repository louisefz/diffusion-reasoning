

"""Train answer-only flow model. Run substantial training on a compute node."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time
import torch
from model import Config, FlowModel, batch, losses, sample, direct_logits
from task import OnlineEpisodes, load_rows
from tracking import Tracker


def select_task(rows, task):
    return rows if task == "mixed" else [r for r in rows if r["task"] == task]


def training_rows(stream, train, batch_size, task):
    if stream:
        # Each episode generates a pair; consume twice as many rows for lookup-only.
        return select_task(stream.next_batch(batch_size if task == "mixed" else 2*batch_size), task)
    return random.choices(train, k=batch_size)


@torch.no_grad()
def metrics(model, rows, device, steps, batch_size=64, seed=123, objective="flow"):
    was_training = model.training
    model.eval()
    counts = defaultdict(lambda: [0, 0])
    # Fixed eval randomness without altering training RNG.
    with torch.random.fork_rng(devices=([device.index or 0] if device.type == "cuda" else [])):
        torch.manual_seed(seed)
        for offset in range(0, len(rows), batch_size):
            subset = rows[offset:offset+batch_size]
            condition, labels = batch(subset, model.config, device)
            if objective == "direct":
                logits = direct_logits(model, condition)
            else:
                logits, _ = sample(model, condition, steps)
            correct = logits.argmax(-1).eq(labels).cpu().tolist()
            for r, ok in zip(subset, correct):
                for key in ("overall", f"{r['task']}/d{r['depth']}"):
                    counts[key][0] += int(ok)
                    counts[key][1] += 1
    model.train(was_training)
    return {k: dict(correct=v[0], total=v[1], accuracy=v[0]/v[1]) for k, v in sorted(counts.items())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--self-condition", action="store_true")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--checkpoint-every", type=int, default=None,
                   help="Default: eval interval; independent of validation frequency")
    p.add_argument("--eval-limit", type=int, default=0, help="0 evaluates all validation examples")
    p.add_argument("--sampling-steps", type=int, default=32)
    p.add_argument("--resume", type=Path)
    p.add_argument("--data-mode", choices=["online", "offline"], default=None,
                   help="Default: online for new runs; checkpoint mode for resumes")
    p.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="disabled")
    p.add_argument("--task", choices=["mixed", "lookup"], default=None)
    p.add_argument("--objective", choices=["flow", "direct"], default=None)
    p.add_argument("--wandb-project", default="continuous-reasoning")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-group", default=None)
    a = p.parse_args()
    a.checkpoint_every = a.checkpoint_every if a.checkpoint_every is not None else a.eval_every
    if a.checkpoint_every < 1:
        p.error("checkpoint-every must be positive")
    if min(a.steps, a.batch_size, a.width, a.layers, a.heads, a.eval_every, a.sampling_steps, a.threads) < 1:
        p.error("Counts must be positive")
    if a.width % a.heads:
        p.error("width must be divisible by heads")
    if a.out.exists() and not a.resume:
        p.error("Output exists; use a new output path or --resume")
    torch.set_num_threads(a.threads)
    device = torch.device(a.device)
    if a.device == "cuda" and not torch.cuda.is_available():
        p.error("No CUDA device available; use a GPU compute allocation or --device cpu for smoke tests")
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    manifest = json.loads((a.data / "manifest.json").read_text())
    checkpoint = None
    if a.resume:
        checkpoint = torch.load(a.resume, map_location="cpu", weights_only=False)
        if checkpoint["manifest"] != manifest:
            raise ValueError("Resume dataset differs")
    saved_mode = checkpoint.get("data_mode", "offline") if checkpoint else None
    saved_task = checkpoint.get("task", "mixed") if checkpoint else "mixed"
    saved_objective = checkpoint.get("objective", "flow") if checkpoint else "flow"
    a.objective = a.objective or saved_objective
    if checkpoint and a.objective != saved_objective:
        p.error("Resume must preserve objective")
    a.task = a.task or saved_task
    if checkpoint and a.task != saved_task:
        p.error("Resume must preserve task selection")
    a.data_mode = a.data_mode or saved_mode or "online"
    if checkpoint and a.data_mode != saved_mode:
        p.error("Resume must preserve data mode; start a new run to change it")
    if a.data_mode == "online" and (a.batch_size < 2 or a.batch_size % 2):
        p.error("Online mode needs an even batch size >=2")
    if checkpoint and a.data_mode == "online" and a.batch_size != checkpoint["batch_size"]:
        p.error("Online resume must preserve batch size for exact data continuation")
    splits = ["validation", "test"] if a.data_mode == "online" else ["train", "validation"]
    for split in splits:
        actual = hashlib.sha256((a.data/f"{split}.jsonl").read_bytes()).hexdigest()
        if actual != manifest["splits"][split]["sha256"]:
            raise ValueError(f"Dataset hash mismatch: {split}")
    train = load_rows(a.data/"train.jsonl") if a.data_mode == "offline" else None
    if train is not None:
        train = select_task(train, a.task)
        if not train:
            p.error("No training examples for selected task")
    stream = None
    if a.data_mode == "online":
        # Test table IDs are consulted only for exclusion, never labels/scores.
        heldout = set()
        for split in ["validation", "test"]:
            with (a.data/f"{split}.jsonl").open() as f:
                heldout.update(json.loads(line)["table_id"] for line in f if line.strip())
        stream = OnlineEpisodes(a.seed, manifest["symbols"], manifest["functions"],
                                manifest["train_depths"], heldout)
        if checkpoint:
            stream.load_state_dict(checkpoint["online_state"])
    val = load_rows(a.data/"validation.jsonl")
    val = select_task(val, a.task)
    if not val:
        p.error("No validation examples for selected task")
    if a.eval_limit:
        val = val[:a.eval_limit]
    config = Config(manifest["symbols"], manifest["functions"], manifest["max_depth"],
                    a.width, a.layers, a.heads, a.self_condition, decode_history=False)
    if checkpoint:
        config = Config(**checkpoint["config"])
    model = FlowModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
    start = 0
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng"])
        random.setstate(checkpoint["python_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        start = checkpoint["step"]
    if a.steps <= start:
        p.error("--steps must exceed resumed checkpoint step")
    a.out.mkdir(parents=True, exist_ok=True)
    run = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
               config=asdict(config), parameters=sum(p.numel() for p in model.parameters()),
               torch_version=torch.__version__, manifest=manifest,
               effective_lr=optimizer.param_groups[0]["lr"])
    (a.out/f"run_from_{start}.json").write_text(json.dumps(run, indent=2)+"\n")
    print(json.dumps(dict(parameters=run["parameters"], device=str(device), start=start,
                          data_mode=a.data_mode, batch_size=a.batch_size)), flush=True)
    with Tracker(a, run, start) as tracker, (a.out/"metrics.jsonl").open("a") as log:
        tic = time.monotonic()
        for step in range(start+1, a.steps+1):
            model.train()
            rows = training_rows(stream, train, a.batch_size, a.task)
            condition, labels = batch(rows, config, device)
            optimizer.zero_grad(set_to_none=True)
            loss, fm, ce = losses(model, condition, labels, a.objective)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at {step}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step == start+1 or step % 100 == 0 or step == a.steps:
                record = dict(step=step, loss=loss.item(), fm=fm.item(), ce=ce.item(),
                              grad_norm=float(norm), elapsed_s=time.monotonic()-tic)
                if stream:
                    record.update(episodes_seen=stream.count,
                                  examples_seen=stream.count*(2 if a.task == "mixed" else 1))
                record["lr"] = optimizer.param_groups[0]["lr"]
                record["session_queries_per_s"] = (step-start)*a.batch_size/record["elapsed_s"]
                if device.type == "cuda":
                    record["cuda_peak_allocated_mb"] = torch.cuda.max_memory_allocated(device)/2**20
                print(json.dumps(record), flush=True)
                log.write(json.dumps(record)+"\n")
                log.flush()
                tracker.log(record)
            if step % a.eval_every == 0 or step == a.steps:
                result = dict(step=step, validation=metrics(model, val, device, a.sampling_steps,
                                                          batch_size=a.batch_size, objective=a.objective))
                print(json.dumps(result), flush=True)
                log.write(json.dumps(result)+"\n")
                log.flush()
                tracker.log(result)
            if step % a.checkpoint_every == 0 or step == a.steps:
                state = dict(config=asdict(config), model=model.state_dict(),
                    optimizer=optimizer.state_dict(), step=step, manifest=manifest,
                    data_mode=a.data_mode, batch_size=a.batch_size, task=a.task, objective=a.objective,
                    online_state=stream.state_dict() if stream else None,
                    torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                    cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None)
                target = a.out/f"step_{step}.pt"
                temporary = a.out/f"step_{step}.pt.tmp"
                torch.save(state, temporary)
                temporary.replace(target)
                if step == a.steps:
                    # Link the final checkpoint without duplicating large tensors.
                    final = a.out/"final.pt"
                    if final.is_symlink():
                        final.unlink()
                    elif final.exists():
                        raise FileExistsError(final)
                    final.symlink_to(target.name)


if __name__ == "__main__":
    main()
