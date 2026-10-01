"""Bounded CPU memorization diagnostic, not a generalization experiment."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import torch
from model import Config, FlowModel, batch, losses
from task import generate, validate
from train import metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--max-seconds", type=float, default=60)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--self-condition", action="store_true")
    a = p.parse_args()
    if a.out.exists():
        p.error("Refusing to overwrite diagnostic")
    if min(a.episodes, a.steps, a.max_seconds) <= 0:
        p.error("Budgets must be positive")
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    device = torch.device("cpu")
    config = Config(width=64, layers=2, heads=4, self_condition=a.self_condition)
    rows = generate(a.seed, config.symbols, config.functions, [1, 2, 4, 8],
                    a.episodes, set(), "overfit")
    validate(rows, config.symbols, config.functions)
    a.out.mkdir(parents=True)
    (a.out/"examples.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    model = FlowModel(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    condition, labels = batch(rows, config, device)
    tic = time.monotonic()
    history = []
    reason = "step_budget"
    for step in range(1, a.steps+1):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss, fm, ce = losses(model, condition, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite diagnostic loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        opt.step()
        if step == 1 or step % 100 == 0:
            result = metrics(model, rows, device, 16, batch_size=len(rows), seed=123)
            record = dict(step=step, elapsed_s=time.monotonic()-tic,
                          loss=loss.item(), fm=fm.item(), ce=ce.item(), training_examples=result)
            history.append(record)
            print(json.dumps(record), flush=True)
            if result["overall"]["accuracy"] == 1:
                confirm = metrics(model, rows, device, 16, batch_size=len(rows), seed=456)
                if confirm["overall"]["accuracy"] == 1:
                    reason = "memorized_two_sampling_seeds"
                    break
        if time.monotonic()-tic >= a.max_seconds:
            reason = "wall_clock_budget"
            break
    final = {str(seed): metrics(model, rows, device, 16, batch_size=len(rows), seed=seed)
             for seed in [123, 456, 789]}
    report = dict(config=asdict(config), parameters=sum(p.numel() for p in model.parameters()),
                  episodes=a.episodes, examples=len(rows), seed=a.seed, step=step,
                  elapsed_s=time.monotonic()-tic, stop_reason=reason, history=history,
                  final_training_accuracy=final,
                  interpretation="Memorization only; no held-out reasoning claim")
    (a.out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    torch.save(dict(config=asdict(config), model=model.state_dict(), step=step), a.out/"model.pt")
    print(json.dumps(dict(step=step, stop_reason=reason,
                          accuracy={k:v["overall"]["accuracy"] for k,v in final.items()})), flush=True)


if __name__ == "__main__":
    main()
