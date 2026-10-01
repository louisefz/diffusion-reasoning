"""Generate reproducible, table-disjoint task splits and paired controls."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from task import generate, validate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--symbols", type=int, default=8)
    p.add_argument("--functions", type=int, default=4)
    p.add_argument("--train-episodes", type=int, default=10000)
    p.add_argument("--eval-episodes", type=int, default=500)
    a = p.parse_args()
    if a.symbols < 3 or a.functions < 2 or min(a.train_episodes, a.eval_episodes) < 1:
        p.error("Use >=3 symbols, >=2 functions, and positive episode counts")
    if a.out.exists():
        p.error(f"Refusing to overwrite {a.out}")
    a.out.mkdir(parents=True)
    manifest = dict(seed=a.seed, symbols=a.symbols, functions=a.functions,
                    train_depths=[1, 2, 4, 8], eval_depths=[1, 2, 4, 8, 12, 16],
                    max_depth=16, controls="paired lookup; first operation only", splits={})
    seen = set()
    for k, name in enumerate(["train", "validation", "test"]):
        depths = manifest["train_depths"] if name == "train" else manifest["eval_depths"]
        count = a.train_episodes if name == "train" else a.eval_episodes
        rows = generate(a.seed + k, a.symbols, a.functions, depths, count, seen, name)
        validate(rows, a.symbols, a.functions)
        payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
        (a.out / f"{name}.jsonl").write_text(payload)
        manifest["splits"][name] = dict(episodes=count, examples=len(rows),
            sha256=hashlib.sha256(payload.encode()).hexdigest(),
            groups=dict(Counter(f"{r['task']}/d{r['depth']}" for r in rows)),
            answers=dict(Counter(str(r["answer"]) for r in rows)))
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

