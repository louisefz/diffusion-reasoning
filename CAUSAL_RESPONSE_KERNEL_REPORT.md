# Causal Response Kernel of Reasoning in ELF

## Question

When a localized reasoning representation is changed at flow time `s`, how is
that perturbation propagated by the subsequent continuous dynamics, and when
can it steer the trajectory into a counterfactual answer basin?

## Experimental design

- Model: `runs/official-composition-d4-1m-61996829/checkpoint_100000`
- Data: 200 minimal d4 counterfactual pairs, 50 edits at each composition step
- Flow: eight uniform Euler steps
- Circuit: block 11, head 3
- Interventions:
  - `k_final_table`: donor Key vectors at the final-function table
  - `k_edited_table_control`: donor Key vectors at the edited-function table
  - `v_changed_cell`: donor Value at the causally accessed table cell
  - `v_swap_partner_control`: donor Value at the unaccessed swap partner
- A patch is applied during exactly one velocity evaluation at time `s`.
  All later evaluations use the native source-conditioned model.
- At every later time `t`, the experiment records latent displacement along the
  paired donor-answer axis, response norm and gain, response-direction
  persistence, donor-vs-source decoder logit response, and decoded answer.
- Main run uses patch strengths 0.25 and 1.0.  A follow-up sweep uses strengths
  0.25, 0.4, 0.5, 0.6, 0.75, 0.9, and 1.0 at `s=0, 0.125, 0.25`.

The response coordinate is

```text
R_axis(t, s) =
    <z_patch(t; s) - z_source(t), z_donor(t) - z_source(t)>
    / ||z_donor(t) - z_source(t)||^2.
```

This is an intervention-induced response statistic, not a stochastic
committor and not a claim that ELF follows a physical energy function.

## Main results

### 1. Reasoning-to-answer transport is concentrated early

For a full patch at `s=0`:

| Intervention | Immediate answer-axis response | Final response | Final donor answer |
|---|---:|---:|---:|
| final-table Key | 0.824 | 0.321 | 14.7% |
| edited-table Key | 0.200 | 0.083 | 4.0% |
| accessed-cell Value | 0.863 | 0.417 | 28.0% |
| swap-partner Value | approximately 0 | approximately 0 | 0.0% |

At `s=0.125`, final Value response remains 0.202 with 20% donor answers, but
the final Key response falls to 0.034 with 3.3% donor answers.  At `s=0.25`,
Key is ineffective and Value is weak (final response 0.062, 6% donor answers).
After `s=0.375`, neither intervention changes the final discrete answer.

Thus the circuit has ordered causal windows: routing through Key closes first,
while semantic readout through Value remains effective slightly longer.

### 2. The early pulse is dynamically propagated

The answer-position response norm at `s=0` grows between the immediate
post-pulse state and the endpoint:

- final-table Key: gain 2.91
- accessed-cell Value: gain 3.54

The projection onto the donor-answer axis decreases from its immediate value,
so not every expanding direction remains answer-relevant.  Nevertheless, a
large positive component persists to the endpoint and changes decoded answers.
The correct interpretation is selective amplification plus partial correction,
not uniform exponential instability.

Paired bootstrap comparisons at `s=0` give:

- Key final response minus edited-table response: 0.238,
  95% bootstrap CI [0.172, 0.305], n=150.
- Value accessed-cell response minus swap-partner response: 0.417,
  95% bootstrap CI [0.280, 0.549], n=50.
- Value donor-answer-rate difference: 0.28,
  95% bootstrap CI [0.16, 0.40].

The edited-function Key condition is an alternative plausible routing site,
not a perfectly irrelevant control: an edited function can occur again and is
part of the upstream computation.  The swap-partner Value condition is the
clean position-matched negative control.

### 3. The response is strongly nonlinear in intervention strength

At `s=0`, final-table Key shows a sigmoidal dose response:

| Strength | Immediate response | Final response | Donor answer |
|---:|---:|---:|---:|
| 0.25 | 0.000 | 0.000 | 0.0% |
| 0.40 | 0.014 | 0.000 | 0.0% |
| 0.50 | 0.202 | 0.059 | 2.0% |
| 0.60 | 0.625 | 0.190 | 8.7% |
| 0.75 | 0.805 | 0.286 | 14.7% |
| 0.90 | 0.824 | 0.318 | 14.7% |
| 1.00 | 0.824 | 0.321 | 14.7% |

The accessed-cell Value response is more gradual locally, but its endpoint
effect remains thresholded.  At `s=0`, final Value response is zero through
strength 0.6, rises to 0.077 at 0.75, 0.244 at 0.9, and 0.417 at 1.0.
The donor-answer rate is 0%, 12%, and 28% at strengths 0.75, 0.9, and 1.0.

Delaying the intervention shifts the effective threshold upward and reduces
the attainable response.  This is consistent with a shrinking controllability
window and basin-like correction by the source dynamics.  It is not yet enough
evidence to call the behavior a thermodynamic phase transition.

## Mechanistic interpretation

The combined evidence supports the following causal chain:

```text
early semantic routing/readout
    -> nonlinear rotation of the answer velocity
    -> propagation and selective amplification by later flow dynamics
    -> partial crossing of an answer-basin boundary
    -> stable counterfactual lexical answer
```

The result is stronger than saying that answer information is decodable early:
the localized edit changes the future vector field, creates a persistent
trajectory displacement, and changes the final answer.  It also refines the
earlier claim: weak edits are actively corrected, whereas sufficiently strong
and sufficiently early edits can redirect the trajectory.

## Artifacts

- Full kernel: `runs/official-causal-response-full-62165781/`
- Dose response: `runs/official-causal-dose-response-62165792/`
- Main 2D figure:
  `runs/official-causal-response-full-62165781/causal_response_kernel_strength_1p0.png`
- Dose-response figure:
  `runs/official-causal-dose-response-62165792/causal_dose_response.png`
- Implementation: `official_causal_response_kernel.py`

## Next decisive experiment

The current experiment establishes propagation and nonlinear steering for d4.
The next mechanism test should compare response lifetime and dose threshold
across controlled depths d1, d2, and d4.  A depth-dependent delay or higher
threshold would support a genuine reasoning-dynamics claim.  If deeper tasks
instead show insufficient relaxation before the flow advances, that directly
motivates the fast-slow recurrent method; if they show wrong-basin trapping,
the appropriate method is branching/restart rather than more loops.
