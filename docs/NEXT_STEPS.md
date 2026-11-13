# Next steps after the first completed study — 12 November 2026

13 November update: recovery and validation controls are implemented as opt-in
paths; non-optimizer checks pass. The next action is the user-run synthetic
optimizer gate in [the handoff](operations/SUPERVISED_RECOVERY.md), followed by
full regression and measured parity/performance checks. The criteria below
remain acceptance requirements, not claims that all gates have passed.

The first uncapped study is complete and preserved. The next task is to make
replication reliable and affordable, then test the research questions with
matched controls. This is a staged work plan, not a frozen training protocol or
authorization for the assistant to launch training. No new study is planned or
executed by creating this document.

## 1. Use the completed study to decide what to test

The [completion review](../reports/bilstm_completion_2026-11-12/README.md) verifies
all 18 jobs and analyzes paired saved validation predictions. The
[learning-curve review](../reports/next_steps_2026-11-12/README.md) adds curves
and class-specific observations without any model fitting or held-out scoring.

The useful question is whether waveform SSL and spectral fusion improve sleep
staging under limited annotations, compared with simpler matched controls.
The current evidence supports hypotheses, not a claim that the most elaborate
architecture is consistently best:

| Question | Current evidence | Required next evidence |
|---|---|---|
| Does SSL help with fewer labels? | Smaller-budget concat/attention SSL contrasts have positive exploratory participant intervals; raw SSL is inconclusive | Repeat matched scratch/SSL comparisons across predeclared optimization seeds |
| Does attention improve on concatenation? | No convincing advantage across endpoints/budgets; the smaller-budget pooled ranking reverses under equal-participant averaging | Keep concat as a primary comparator and test the attention contrast directly |
| Does BiLSTM improve on LSTM? | Not isolated: all new conditions use BiLSTM and historical models differ in other ways | Add a matched unidirectional-LSTM comparison only if making a superiority claim |
| Is the fusion physiologically meaningful? | Inputs use named spectral features; attention and gate behavior are not yet mechanistic evidence | Feature-group ablations and descriptive gate/attention analyses, with careful interpretation |

Do not change the selected epoch, switch the primary endpoint after viewing
results, remove an inconvenient baseline, or describe a validation interval as
independent confirmation. Keep pooled checkpoint-selection metrics distinct
from equal-participant macro-F1. The latter intervals condition on one trained
seed and one label draw and have no multiple-comparison or selection correction.

## 2. Implement recovery before another long campaign

The two outages occurred before the interrupted job's first completed
validation checkpoint. Existing best-model checkpoints and job-level retry
preserve completed work, but they are not automatic supervised epoch resume.

Implement an epoch-boundary recovery path in a new source version with:

- Separate latest recovery state and best validation-selected model.
- Model, optimizer and iteration state; completed epoch and next epoch's
  deterministic sampler/RNG state, including dropout randomness.
- Early-stopping and learning-rate-reduction callback state, including best
  values, patience/wait counters and current learning rate.
- Data/split, normalizer, initialization, implementation and package seals.
- Atomic publication after all end-of-epoch callbacks have finished, retaining
  the previous valid checkpoint if power fails during a write.
- Strict compatibility checks and preservation of old attempts. Existing
  interrupted attempts cannot be retroactively given missing recovery state.

Acceptance requires comparing uninterrupted versus interrupted/restarted runs
under the same supported environment, checking optimizer steps, epoch ordering,
callbacks, weights, checkpoint choice and predictions. It also requires a
corrupt/incomplete-write test and rejection of changed data/code/packages.
Mocked/read-only checks can be assistant-run. Tests that perform optimizer
updates remain user-run under current ownership. Do not claim exact recovery
until the relevant tests pass; initially target full epoch boundaries.

## 3. Reduce validation overhead and benchmark total runtime

A completed spectral epoch logged roughly 90 seconds for its training pass but
1,015 seconds overall. This suggests substantial time outside the training
pass, consistent with the current small-batch Python validation loop. It is a
diagnostic observation, not a measured attribution of every remaining second.

Investigate compiled inference for validation, a separate inference batch size,
and reduced Python/metadata overhead. Preserve training batch size, targets,
normalization, loss, checkpoint policy and the full validation cohort. The
current models use layer normalization, but numerical equivalence must still
be checked rather than assumed when changing execution or batching.

Acceptance requires identical enrolled target IDs, finite probabilities and a
declared tight probability tolerance against the original validation path;
compare predicted classes, confusion matrices, loss and macro-F1 as well.
Document any numerical differences and their effects on callback decisions.
Keep the original reference implementation/results available. A speed change
must not silently alter checkpoint selection or stopping behavior.

Benchmark total epoch time including loading, validation and saving, plus peak
memory. Do not extrapolate from a cached forward pass alone. Any optimizer-based
profile must be launched by the user. Resolve cooling concerns and choose local
or cloud hardware using a short measured benchmark before a long rental/run.
No account setup, upload, purchase or new hardware configuration is implied.

Implementation changes require fresh frozen studies. Keep the completed
`runs/bilstm_terminal_2026-11-02` study and its evidence unchanged.

## 4. Freeze a replication study before fitting more models

After the recovery/performance gates pass, write a concrete protocol containing
model definitions, labeled and unlabeled participant pools, source split,
checkpoint/stopping policies, paired contrasts, seeds, endpoints and compute
limits. Preserve full-label controls even if attention is the intended method.

A reasonable starting draft is two additional predeclared optimization seeds,
such as 17 and 123, with draw seed 42 held fixed. This separates optimization
variation from a change in participant label selection. Varying label draws is
a distinct study; do not claim robustness to label selection from one draw.
The seed choices here are a proposal, not a registered final protocol.

For M1–M6 at both budgets and two new seeds, the existing job graph would need
24 supervised fits, two SSL fits and three preparation jobs (29 jobs). Including
M0 for both seeds would make 33 jobs. This describes scope, not a runtime or
cost commitment. Final scope depends on benchmarked resources and the scientific
claims, not on which validation result is most flattering.

The existing seed-42 study remains exploratory provenance. Before pooling it
with a new implementation or platform, establish and document equivalence of
the training/selection procedure. If that cannot be established, report versions
separately and repeat an appropriate matched reference under the new protocol.
Never bypass runtime/code seals or relabel old results as a new study.

If subsequent development changes SSL learning rates, augmentation strength,
the SSL objective, fusion or regularization, treat that as source-validation
development with explicit controls and a tuning budget. Early SSL optima and
saturated view retrieval justify investigation but do not prove the cause.
Do not make multiple method changes together and attribute the result to SSL.

The assistant prepares code, checks and frozen plans. The user receives visible
terminal commands and owns all optimizer-based tests, profiling and training.

## 5. Finalize the evaluation policy before opening held-out scores

Once development/replication decisions are complete, freeze the full evaluated
model family, endpoints, class metrics, paired comparisons, uncertainty methods,
eligible epoch mask and handling of all seeds. Then run the prescribed Sleep-EDF
held-out evaluation without using its outcomes to choose settings or models.

Historical source-test baseline performance was already inspected and must be
disclosed; do not describe that split as never seen during the entire project.
Keep ISRUC predictive scoring unopened: cross-dataset work is still deferred.
The paper's physiological and BiLSTM claims must match the ablations actually
performed. A clear, reproducible finding about label efficiency can be valuable
without claiming that every component wins in every setting.

The immediate implementation order is **recovery, validation speed, controlled
benchmark, frozen replication**. No new full training command is needed today.
