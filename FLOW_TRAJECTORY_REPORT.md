# ELF Flow Trajectory Smoke Experiment

## Run and validation

- Successful Slurm job: `62060730` (`gpu_a100_debug`, A100, exit `0:0`, 31 s).
- Checkpoint: `runs/official-composition-d4-1m-61996829/checkpoint_100000`.
- Data: 30 held-out examples each from `compose/d1`, `compose/d2`,
  `compose/d4`, and matched `lookup/d4` (120 unique examples total).
- Sampler: 16-step logit-normal ODE, CFG 2, self-conditioning CFG 1.
- Recorded tensors:
  - `z`: `[120, 17, 132, 512]`, BF16;
  - `velocity`: `[120, 16, 132, 512]`, BF16.
- The automatic verifier checked all files, 2,040 per-sample/time rows,
  group sizes, unique source IDs, and tensor shapes.

The first attempt (`62059404`) computed all 120 trajectories but failed while
writing the CSV because empty initial-state accumulators were retained. The
finalizer was fixed, a regression test was added, and the corrected run
completed normally.

## Main descriptive results

| Group | Final current-`z` accuracy | Final predicted-endpoint accuracy | First `z >= 90%` | First endpoint `>= 90%` |
|---|---:|---:|---:|---:|
| compose/d1 | 96.7% | 96.7% | 1.0000 | 0.0910 |
| compose/d2 | 100% | 100% | 0.3183 | 0.0910 |
| compose/d4 | 100% | 100% | 0.3403 | 0.0910 |
| lookup/d4 | 96.7% | 100% | 1.0000 | 0.0910 |

At the first post-update state (`t=0.0796`), predicted-endpoint accuracy was
76.7%, 66.7%, 83.3%, and 83.3% for compose d1/d2/d4 and lookup d4. At the
next state (`t=0.0910`) it was already 96.7%, 96.7%, 100%, and 96.7%, while
the native decoder applied to the *current* `z_t` reached only 23.3%, 23.3%,
33.3%, and 33.3%.

Answer-slot velocity norms were similar across groups (roughly 56--60), and
successive velocity directions were highly aligned (minimum group-mean cosine
about 0.953--0.957; later values mostly 0.97--0.998). There is no obvious
extra curvature, larger update magnitude, or delayed transition for d4.

## Interpretation

This smoke experiment gives strong evidence for a separation between two
quantities:

1. The model's denoiser can predict a correct clean endpoint very early.
2. The actual integrated latent `z_t` only becomes compatible with the native
   token decoder later.

It does **not** show that more reasoning depth produces later flow-time
commitment. The ordering is not monotonic in depth: compose/d4 is at least as
early as d1/d2 in several current-`z` measures, and matched lookup/d4 looks
similar. The velocity statistics also mostly overlap. Thus, for this trained
model and task, flow time appears to primarily describe transport/lexical
realization, while the task computation is likely performed inside each
Transformer denoiser evaluation.

The early predicted endpoint must not be called causal commitment: computing
that endpoint invokes the full conditional model and may redo the reasoning.
Likewise, a first threshold crossing is not guaranteed to be stable (for
example compose/d1 endpoint accuracy falls from 100% to 96.7% at the final
state). The saved full tensors now enable the necessary next experiment:
branch/perturb intermediate `z_t`, continue the flow, and measure answer
stability rather than relying only on decodability.
