# From Cooperative Computation to Persistent Semantic Modes

## Updated question

Does reasoning depth merely delay commitment along ELF flow time, or does it
change the organization and dynamical persistence of the causal computation?

The experiments reject the first, simpler hypothesis.  Depth changes both the
causal circuit and whether its output survives subsequent source-conditioned
flow dynamics.

## 1. The same localized d4 circuit is not shared by shallow tasks

A block-11/head-3 semantic Key/Value intervention strongly steers d4, but has
essentially zero effect for d1 and d2.  This initially appears to contradict a
depth comparison, but earlier layer/component results predicted the reason:
d1/d2 answer computation becomes causally effective in block 10, whereas d4
does so in block 11.

## 2. Shallow computation is a sparse cooperative mode

Block-10 head scans show that heads 1, 2, and 3 form the shallow causal circuit.
No single head is sufficient, but combinations are strongly super-additive.

### d1, edit step 1

| Patched donor heads | Donor-answer rate |
|---|---:|
| h1 | 0% |
| h2 | 0% |
| h3 | 0% |
| h1+h3 | 13% |
| h2+h3 | 36% |
| h1+h2+h3 | 68% |
| all 12 heads | 68% |
| control h4+h5+h6 | 0% |

### d2

For edit step 1, h1+h3 and h2+h3 reach 66% and 71%; h1+h2+h3 reaches
73%, equal to all 12 heads.  For edit step 2 the corresponding rates are 66%,
71%, and 72%.  Size-matched control heads remain at 0%.

On the continuous donor-vs-source logit response, pairwise interaction terms
are also large and positive.  Therefore the effect is not only an argmax
threshold artifact.  The shallow circuit is sparse but genuinely cooperative,
with head 3 as a necessary core and heads 1/2 as enabling partners.

## 3. Unified circuit interventions reveal fast and slow semantic modes

To compare depths fairly, a one-step flow pulse was applied to each task's
causally sufficient circuit:

- d1: block-10 heads 1/2/3
- d2: block-10 heads 1/2/3
- d4: block-11 head 3

All interventions act on the answer-position pre-projection head output and use
size-matched non-causal head controls.

At flow time s=0:

| Depth | Immediate donor-axis response | Endpoint response | Retention | Response-norm gain | Donor answer |
|---:|---:|---:|---:|---:|---:|
| d1 | 0.790 | 0.007 | 0.9% | 0.56 | 0% |
| d2 | 0.895 | 0.055 | 6.1% | 0.88 | 0% |
| d4 | 0.802 | 0.225 | 28.1% | 2.00 | 22.5% |

The immediate responses are comparable, so the result is not explained by a
weaker shallow intervention.  The difference lies in subsequent dynamics:

- d1/d2 circuit output is a transient fast mode.  Later denoiser evaluations
  re-read the source condition and almost completely restore the source answer.
- d4 circuit output couples to a slow semantic mode.  A substantial component
  survives, its response norm grows, and the final discrete answer can change.
- Size-matched control circuits have approximately zero donor-axis response at
  all depths.

The strength sweep reinforces this distinction.  At s=0 and full strength,
d1/d2 have large immediate responses but near-zero endpoints.  For d4, a
nonlinear endpoint response appears above roughly 0.75 strength and rises to
0.225 at full strength.  Thus the critical distinction is persistence after
injection, not only instantaneous circuit sufficiency.

## 4. Physical/dynamical interpretation

The most defensible language is a fast-slow decomposition:

```text
shallow circuit edit -> fast semantic displacement -> source dynamics restores
deep circuit edit    -> slow semantic displacement -> flow retains/amplifies
```

This resembles mode separation and basin crossing, but it should not yet be
called a thermodynamic phase transition.  The model is a deterministic,
conditioned, non-autonomous neural ODE, and no energy or equilibrium measure has
been established.

The result changes the method hypothesis.  More inner loops are useful only if
they help a difficult problem form a persistent slow semantic mode.  A generic
increase in ODE steps or repeated shallow computation is unlikely to help,
because the native dynamics actively erase shallow transient edits.

## Artifacts

- Head synergy job: 62165949
- Unified circuit response job: 62165962
- Head synergy implementation: `official_head_group_synergy.py`
- Circuit response implementation: `official_circuit_response_kernel.py`
- Cross-depth response plot:
  `analysis_artifacts/circuit_depth_response_comparison.png`
- Cross-depth dose plot:
  `analysis_artifacts/circuit_depth_dose_comparison.png`

## Next method-facing experiment

Train or iterate an internal recurrent state and test whether recurrence moves
its output from the transient subspace toward the empirically identified slow
semantic subspace.  Before training a controller, the minimal intervention is
to repeat the causally sufficient circuit computation within one fixed flow
state and measure whether semantic retention, rather than just immediate
accuracy, increases with loop count.

## 5. Sustained forcing and basin entry

The proposed sustained intervention was completed in jobs 62166039 and
62166063.  Each task's causal circuit was replaced by its paired donor circuit
for 1--6 consecutive flow evaluations, followed by native source-conditioned
dynamics.  Size-matched non-causal heads were patched as controls.

When forcing always starts at t=0, longer forcing produces a sharp increase in
endpoint persistence.  However, this is confounded because longer forcing also
leaves fewer native steps for source recovery.  Job 62166063 therefore fixes
the release state at t=.75 and varies only how many consecutive steps before
release are forced.

### Fixed release at t=.75

| Forced steps | d1 donor rate | d2 donor rate | d4 donor rate |
|---:|---:|---:|---:|
| 1 | 0% | 0% | 0% |
| 2 | 4% | 25% | 4.5% |
| 3 | 12% | 61% | 36% |
| 4 | 38% | 70% | 60.5% |
| 5 | 60% | 71% | 76.5% |
| 6 | 68% | 71% | 77.5% |

All size-matched controls remain at 0%.  The donor-axis state at release also
grows monotonically with forcing count, so the effect cannot be explained by
less post-release recovery time.  For d2/d4 above threshold, endpoint response
can exceed response at release: after external forcing stops, native flow keeps
moving toward the donor basin.  This is direct evidence for accumulated state
change followed by autonomous basin-directed dynamics.

The critical forcing count is not monotonic in task depth (approximately 5, 3,
and 4 steps for d1/d2/d4 to exceed 50% donor control).  This is expected because
the circuits have different temporal windows.  A method should therefore stop
based on state stability/value of additional compute, not use a fixed loop
count inferred only from nominal problem depth.

These interventions use an oracle donor circuit.  They establish dynamical
controllability and memory, but do not yet show that an autonomous recurrent
model will discover the correct slow state.  That is the next method test.

## 6. Autonomous recurrence: zero-shot baseline

We implemented a tied-weight recurrence over blocks 10--11. One pass is the
original ELF computation; additional passes reuse the same parameters at the
same flow time before block 12 reads out the velocity. Thus inner semantic
compute changes while the 16-step ODE solver remains fixed.

On all 2,400 held-out examples (job 62166102), untrained recurrence is stable
and slightly beneficial. K=6 changes 3.42% of predictions relative to K=1,
correcting 12 examples and breaking 6. Exact accuracies for composition are:

| Inner loops | d1 | d2 | d4 | d8 | d12 | d16 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100% | 100% | 100% | 16.5% | 13.0% | 10.5% |
| 2 | 100% | 100% | 100% | 17.0% | 13.0% | 11.5% |
| 4 | 100% | 100% | 100% | 17.0% | 13.5% | 12.0% |
| 6 | 100% | 100% | 100% | 17.0% | 14.0% | 12.0% |

This establishes that repeated computation is not an identity, but the effect
is not yet statistically compelling. Job 62166105 therefore fine-tunes the
same checkpoint with K sampled uniformly from 1--4. The decisive test is
whether post-training accuracy scales with K on unseen depths while shallow
composition and lookup controls remain invariant.

## 7. Recurrent fine-tuning pilots

Three controlled variants were evaluated with the same 2,400 examples, random
seed, 16 outer flow steps, and K in {1,2,4,6}.

1. **Random-K fine-tuning** (job 62166105, evaluation 62166120) samples K
   independently of task depth. It improves the K=1 checkpoint slightly, but
   additional loops exchange roughly equal numbers of correct and incorrect
   answers. This objective primarily teaches loop invariance.
2. **Depth-conditioned fine-tuning** (job 62166131, evaluation 62166136) uses
   K=1/2/4 for composition depth 1/2/4 and K=1 for lookup. It creates a local
   scaling effect on unseen d16: 8.0% at K=1 to 14.5% at K=4, with 18 paired
   corrections and 5 regressions. The effect does not generalize uniformly:
   d8 falls from 13.5% to 12.0%, and d12 is nearly flat.
3. **Intermediate-state supervision** (job 62166153, evaluation 62166159)
   applies a norm-controlled cosine readout after every loop and supervises the
   true function-composition state. The auxiliary 8-way loss falls from about
   2.65 to 1.85 (chance cross-entropy log(8)=2.08) without overwhelming the FM
   loss. Nevertheless, endpoint K-scaling remains absent after the 1k-step
   pilot: d8 is 15.5/14.5/14.5/15.5%, d12 is
   15.5/16.0/16.5/15.5%, and d16 is 11.0/11.5/11.5/10.0%.

The negative result is informative. Reusing blocks 10--11 changes predictions
and can produce isolated depth-dependent gains, but it does not by itself form
an iterative reasoner. The original blocks were trained as a position-specific
stage of a feed-forward computation. They have neither a protected recurrent
memory nor an explicit inner-time signal, and an answer-position auxiliary
readout can become more decodable without becoming causally useful to the
final flow.

The next architecture should therefore introduce a small persistent reasoning
state with an inner-time embedding:

```text
m_(k+1) = m_k + F(m_k, z_t, x, t, tau_k)
v_t     = G(z_t, m_K, t)
```

This separates semantic state evolution from repeatedly transforming the full
token sequence. Intermediate-state targets supervise `m_k`; the outer solver
and NFE remain fixed. The required ablations are no-memory, no-inner-time, and
untied-weight controls. Current recurrence comparison plot:
`analysis_artifacts/recurrent_reasoning_pilots.png`.

## 8. Persistent memory and causal bottleneck

We introduced four recurrent memory tokens and an explicit inner-time
embedding. A non-bottleneck version makes intermediate states substantially
more readable (8-way cosine CE 1.45 versus chance 2.08), but still shows no
endpoint scaling because the original token pathway bypasses the memory. A
weak learned residual coupling also has almost no effect; its gate remains
near the 0.018 initialization.

The decisive intervention is a **causal memory bottleneck**. During inner
recurrence, blocks 10--11 may update memory but their direct token updates are
discarded. The final token/velocity state receives the reasoning result only
through the memory residual. After a 2,000-step pilot (job 62166469,
evaluation 62166476), accuracy scales with inner compute:

| Inner loops K | Overall | d2 | d4 | lookup mean |
|---:|---:|---:|---:|---:|
| 1 | 70.0% | 89.0% | 13.5% | 99.9% |
| 2 | 70.2% | 90.0% | 14.5% | 99.9% |
| 4 | 70.7% | 90.5% | 19.5% | 99.9% |
| 6 | 71.0% | 90.5% | 22.5% | 99.9% |

For d4, K=6 versus K=1 corrects 18 examples and breaks none (paired exact
binomial p=7.6e-6). Across all 2,400 examples it corrects 31 and breaks 9
(p=6.8e-4). K=6 was not used during training, so its continued d4 improvement
is a first test-time-compute extrapolation signal. No comparable effect occurs
on unseen d8/d12/d16 yet.

This identifies causal use, rather than memory existence, as the central
design variable:

```text
readable memory + bypass       -> no K scaling
readable memory + causal route -> monotonic d4 scaling
```

Low-LR continuation (job 62166495) makes the compute dependence much stronger:

| Additional training | K | Overall | d2 | d4 | lookup mean |
|---:|---:|---:|---:|---:|---:|
| 1k | 1 / 2 / 4 / 6 | 69.46 / 70.13 / 70.92 / 71.04 | 85.0 / 88.5 / 88.5 / 88.0 | 13.0 / 13.0 / 22.0 / 24.5 | 99.9 |
| 3k | 1 / 2 / 4 / 6 | 69.88 / 70.38 / 71.46 / 71.58 | 90.0 / 94.0 / 93.5 / 93.5 | 13.0 / 12.0 / 28.0 / 28.0 | 99.9 |
| 5k | 1 / 2 / 4 / 6 | 70.25 / 70.96 / 73.83 / 73.96 | 96.0 / 99.0 / 99.0 / 98.5 | 12.5 / 16.0 / 51.5 / 51.5 | 99.9 |

At the 5k checkpoint, K1 to K4 changes d4 by 78 corrections and zero
regressions (one-sided exact binomial p=3.3e-24). Overall it yields 104
corrections and 18 regressions (p=3.4e-16). The effect is selective: trivial
lookup is invariant and unseen d8/d12/d16 remain near chance. Training has
therefore learned a depth-4 iterative transition, not yet a depth-general
algorithm. K4 and K6 have equal aggregate d4 accuracy (with five examples
switching in each direction), suggesting a stable/saturated regime. Extended
K=1..16 job 62167151 tests whether this is genuine convergence or eventual
over-iteration instability. Updated plot:
`analysis_artifacts/recurrent_reasoning_bottleneck_comparison.png`.

## 9. Finite reasoning time and compressed internal states

The extended K sweep (job 62167151) exposes a finite useful-compute regime:

| K | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| d4 accuracy (%) | 12.5 | 16.0 | 34.0 | 51.5 | 51.5 | 47.5 | 46.5 | 45.0 |

Performance accumulates rapidly through K=4, saturates through K=6, and then
declines mildly. This is inconsistent with a generic "more depth is always
better" account. The learned update has a useful finite iteration horizon and
is not globally contractive around the task-optimal state. Plot:
`analysis_artifacts/recurrent_compute_scaling_extended.png`.

A two-dimensional diagnostic (job 62167163) reads the four ground-truth
composition states from memory after each inner step and at every outer-flow
call. At the last call, k1 predicts s1 at 77%, k2 predicts s2 at 98.5%, k3
predicts final s4 at 88%, and k4 predicts s4 at 99.5%. The explicit s3 readout
at k3 is only 18%. This pattern is nearly stationary from t=0 onward.

The appropriate interpretation is therefore not a literal latent chain
`s1 -> s2 -> s3 -> s4`. The recurrent system performs a compressed algorithm:
the first two iterations recover early symbolic states, the third jumps to an
answer-readable state, and the fourth stabilizes it. Readability alone is not
causality, but here the readout is paired with the causal memory bottleneck and
the endpoint K interventions above. Artifacts:
`runs/official-recurrent-memory-dynamics-62167163/`.

## 10. Temporal compute response: reasoning steers, flow transports

Job 62167172 assigns K=4 compute to matched early or late outer-flow calls,
with K=1 elsewhere and all noise, examples, and solver times fixed. For scarce
budgets, early compute is consistently more valuable:

| K4 calls | earliest-window accuracy | latest-window accuracy | paired early vs late |
|---:|---:|---:|---:|
| 1 | 21.0% | 17.0% | +9/-1, p=.0107 |
| 4 | 25.5% | 21.5% | +12/-4, p=.0384 |
| 8 | 30.0% | 25.5% | +13/-4, p=.0245 |
| 12 | 37.0% | 35.5% | +12/-9, p=.332 |

All-K1 and all-K4 baselines are 16.5% and 47.0%. Thus early computation has a
larger marginal effect, but sustained computation still matters.

Job 62167191 then applies exactly one K1-to-K4 impulse at each of the 16 solver
calls. The endpoint accuracy response is
4.5/2.5/2.5/4.0/4.0/3.0/2.0/2.5/1.5/1.5/1.0/1.0/1.0/0.5/0.5/0.5 percentage
points in temporal order. Every changed endpoint is a correction; there are
no regressions. Response magnitude correlates strongly with remaining
integration horizon, Pearson r=.876 (p=8.8e-6) and Spearman rho=.930
(p=1.8e-7).

This supplies the desired dynamics interpretation:

```text
inner recurrent computation -> velocity correction at time t
                             -> integration over the remaining horizon
                             -> larger or smaller endpoint displacement
```

An early semantic control input has more time to be integrated by the flow.
The effect is not exclusively early, but it decays as the remaining transport
horizon shrinks. This is an empirical causal response kernel for reasoning
compute, not merely a probe correlation. Artifacts:
`runs/official-temporal-recurrent-compute-62167172/` and
`runs/official-recurrent-compute-impulse-62167191/`.

Important scope limits remain: these results use 200 held-out d4 examples, a
single trained checkpoint, and the model's logit-normal 16-step schedule; the
largest call time is about .466 before the final Euler jump to 1.0. Replication
over seeds, solver schedules, and a second task is required before making a
general claim about continuous-language-flow reasoning.

Three immediate replications address part of this limitation. Two additional
logit-normal seeds (jobs 62167203/62167204) and one uniform 16-step grid (job
62167202) give mean impulse gains over the earliest versus latest four calls
of 2.75/.75, 1.38/.38, and 4.88/.50 percentage points, respectively (the
original is 3.38/.63). The uniform-grid response covers t=0 through .9375 and
has Spearman rho=.979 with remaining horizon. Exact pointwise curves are
noisier for the 200-example logit-normal replications, but the early-versus-
late ordering holds in every run. Across all four runs and all 64 impulse
locations, no correct endpoint becomes incorrect. Combined plot:
`analysis_artifacts/recurrent_compute_response_kernel_replication.png`.

## 11. Response-kernel-guided compute allocation

The response kernel predicts a concrete method: under a fixed number of inner
updates, spend recurrent compute at early flow calls, where a semantic velocity
correction has a longer remaining integration horizon.  Four independent
200-example evaluations compare early, uniform, and late allocation while
holding the outer 16-step solver and total inner-update count fixed.

| Inner-update budget | Policy | Mean accuracy over 4 runs |
|---:|---|---:|
| 16 | all K1 | 13.38% |
| 32 | uniform | 16.50% |
| 32 | early | **28.50%** |
| 32 | late | 20.75% |
| 48 | uniform | 30.50% |
| 48 | early | **36.63%** |
| 48 | late | 29.25% |
| 64 | all K4 | 47.50% |

At budget 32, early allocation gains 12.0 percentage points over uniform and
7.75 points over late allocation.  At budget 48, the gains are 6.13 and 7.38
points.  Early beats uniform in all eight matched run/budget comparisons and
beats late in seven of eight; the one reversal is small (31.0% versus 32.0%).

This is the first method-facing result: a causal response kernel measured from
the dynamics predicts where extra semantic compute has the highest endpoint
value.  It improves accuracy at identical inner-update cost, rather than
conflating reasoning compute with extra ODE solver steps.  The result is still
limited to function composition; graph reachability and verifier-backed
Countdown/Sudoku pipelines are the next cross-task tests.
