import argparse
import json
from pathlib import Path
import torch
from model import Config, FlowModel, batch, sample
from task import load_rows
from train import metrics, select_task


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--trajectory", type=Path)
    p.add_argument("--sampling-steps", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--seed", type=int, default=123)
    a = p.parse_args()
    if a.out.exists() or (a.trajectory and a.trajectory.exists()):
        p.error("Refusing to overwrite evaluation artifacts")
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    device = torch.device(a.device)
    saved = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    objective = saved.get("objective", "flow")
    if a.trajectory and objective == "direct":
        p.error("Direct classifier has no flow trajectory")
    model = FlowModel(Config(**saved["config"])).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    rows = select_task(load_rows(a.split), saved.get("task", "mixed"))
    result = dict(checkpoint=str(a.checkpoint), split=str(a.split), seed=a.seed,
                  sampling_steps=a.sampling_steps,
                  objective=objective,
                  metrics=metrics(model, rows, device, a.sampling_steps, a.batch_size, a.seed, objective))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2)+"\n")
    if a.trajectory:
        row = next((r for r in rows if r["task"] == "compose"), rows[0])
        condition, labels = batch([row], model.config, device)
        logits, trace = sample(model, condition, a.sampling_steps, record=True)
        native = []
        with torch.no_grad():
            for s in trace:
                prediction = model(condition, s["z"].to(device), torch.ones(1, device=device),
                                   mode=1, previous=s["previous"].to(device) if model.config.decode_history else None)
                native.append(prediction.cpu())
        mid = trace[len(trace)//2]
        replay, _ = sample(model, condition, a.sampling_steps, z=mid["z"].to(device),
                          previous=mid["previous"].to(device), start_step=mid["step"])
        torch.testing.assert_close(logits, replay, rtol=1e-5, atol=1e-6)
        a.trajectory.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(example=row, condition=condition.cpu(), label=labels.cpu(),
                        states=trace, native_decode_logits=native, final_logits=logits.cpu(),
                        config=saved["config"], sampling_steps=a.sampling_steps,
                        checkpoint=str(a.checkpoint), replay_verified=True), a.trajectory)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
