# ELF Latent Branching Stability Experiment

## Experimental setup

- Successful A100 job: `62073506` (`COMPLETED`, exit `0:0`, 8 min 20 s).
- Checkpoint: `checkpoint_100000` from the d4-heavy ELF-B run.
- Held-out groups: compose d1/d2/d4 and matched lookup d4, 30 examples each.
- Branch flow states: `t = 0.0910, 0.1819, 0.2388, 0.3183`.
- Relative Gaussian perturbation scales: `0, .03, .1, .3, .5, 1.0`.
- Eight random branches per example, time, scale, and intervention.
- Interventions: perturb `z_t` only, previous self-conditioning endpoint only, or both.
- Total: 69,120 continued-flow rollouts.

Perturbations affect generated slots only. A scale of 1.0 means per-coordinate
noise SD equals that component's current RMS. The original clean condition is
preserved during continuation.

## Validation

- All 288 aggregate cells contain 30 examples and 240 rollouts.
- All 69,120 per-branch records are present.
- Every scale-zero branch exactly reproduces its unperturbed answer.
- Smoke jobs `62073452` and `62073468` completed before the full run.

## Result

The conditional flow is extremely robust from the earliest tested state.
Across every time and perturbation scale up to 1.0, the minimum rates were:

| Group | z: min agreement / retention | selfcond: min agreement / retention | both: min agreement / retention |
|---|---:|---:|---:|
| compose d1 | 98.8% / 100% | 97.5% / 100% | 98.3% / 100% |
| compose d2 | 97.1% / 97.1% | 100% / 100% | 97.1% / 97.1% |
| compose d4 | 100% / 100% | 100% / 100% | 100% / 100% |
| lookup d4 | 98.8% / 100% | 97.5% / 100% | 98.3% / 100% |

`retention` is correctness among examples that were correct without the
intervention. There is no systematic increase in stability with flow time and
no ordering d1 < d2 < d4. In fact, compose d4 is perfectly stable everywhere
on this grid.

## Interpretation

This result rejects the simple hypothesis that answer commitment gradually
develops across these ELF flow steps. It also shows that ordinary local random
perturbation is not a sensitive commitment diagnostic here: every future
denoiser call still sees the clean problem condition and can project a heavily
perturbed state back to the correct answer. The measured quantity is therefore
**conditional recoverability**, not proof that the perturbed latent already
stored a causally committed answer.

Together with the earlier result that the predicted endpoint is already
96.7--100% correct at `t=0.091`, the most economical explanation is that the
conditional Transformer recomputes/projects the solution during its forward
pass, while flow time mainly realizes that solution in the latent/token
interface.

## Research decision

The next primary analysis should move inside the 12-layer ELF-B denoiser:

1. record answer information after each Transformer block during the first
   denoiser evaluation;
2. test whether the answer-emergence layer shifts from d1 to d2 to d4;
3. use counterfactual layer-state patching/ablation to separate readable from
   causally used information;
4. if deeper tasks use later layers, add recurrent/looped computation inside
   the denoiser rather than adding more ODE steps.

An optional bridge experiment is matched counterfactual state swapping: swap
generated states between problems with different answers while retaining the
recipient condition, and measure whether the final answer follows the donor
state or the recipient problem.
