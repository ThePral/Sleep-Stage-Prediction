**Recommended direction: a compact, physiology-informed representation study with a strictly frozen external evaluation.**

Implementation started on 27 October 2026. See [implementation status](IMPLEMENTATION_STATUS.md)
for delivered code, validation results, and the next milestone; this document
retains the research design and acceptance criteria.

Your three approaches can form a coherent paper: combine waveform morphology with explicit spectral summaries, initialize the representation through self-supervised learning, and test whether the resulting benefits survive transfer from Sleep-EDF to ISRUC. Each contribution needs its own controlled comparison.

The practical constraint is a 4 GB local GPU and a preferred timescale of roughly one month **to complete the code implementation, integration, and validation**. Paper writing begins after that implementation period. The month should deliver working support for all three proposed approaches, their comparison experiments, and reproducible evaluation. Start training pilots as components become usable; use measured throughput to schedule larger runs and decide whether cloud compute is needed. The complete publication experiment campaign may overlap implementation and continue afterward, with its status tracked separately from code completion.

This roadmap is based on the current project audit, inspection of your downloaded ISRUC-I inventory, and a focused literature review on 26 October 2026. It proposes experiments; none of the new architectures or training methods has been implemented or evaluated yet. GPU manufacturer and usable framework support remain unconfirmed.

**1. Define the research questions and what would count as evidence.**

A suitable central question is:

> Can explicit spectral physiology improve the label efficiency and external robustness of a compact waveform-based sleep-staging model, and does contrastive pretraining strengthen that benefit?

Use three hypotheses, with outcomes that can also falsify them:

| Hypothesis | Essential comparison | Evidence required |
|---|---|---|
| H1: explicit spectral guidance improves the waveform representation | Raw-only, concatenation fusion, and gated cross-attention under matched preprocessing and temporal context | Repeated source results; parameter/cost controls; paired participant-level differences |
| H2: self-supervised initialization improves label efficiency | The same downstream architecture with random versus pretrained initialization | Matched labeled participants, validation policy, tuning budget, and learning curves |
| H3: the benefit transfers to a genuinely unseen cohort | Frozen source-developed models on ISRUC, including the stronger simple controls | Absolute external performance, paired differences, uncertainty, stage-specific failures |

Make external participant-mean macro-F1 the primary generalization endpoint. Prespecify a small set of primary comparisons, such as pretrained gated fusion versus the same gated model trained from scratch, and gated versus concatenation fusion with the same pretraining. Record all source-selected comparisons before opening target performance results. Report pooled epoch macro-F1 separately, rather than substituting it for the participant-mean endpoint.

The features are deterministic summaries of the same EEG. Their potential contribution is inductive bias, access to useful summaries, and optimization—not a second independent physiological sensor. Describe this as two views of single-channel EEG. Also distinguish physiology-informed features from enforced physiological constraints: the proposed network does not automatically obey sleep-stage transition rules.

**2. Position the contribution against the closest existing work.**

The individual ingredients already have substantial precedent. The paper needs a precise difference and convincing evidence for it.

| Work | Why it matters to your study | How to use it |
|---|---|---|
| [XSleepNet](https://arxiv.org/abs/2007.05492) | Joint sleep representations from raw signals and time-frequency views | Establishes that raw/spectral fusion alone is established; informs a simple fusion baseline |
| [TF-C](https://papers.neurips.cc/paper_files/paper/2022/hash/194b8dac525581c346e30a2cebe9a369-Abstract-Conference.html) | Contrastive representation learning with time-frequency consistency | A close conceptual comparator for cross-view pretraining; its transfer setup should not be equated automatically with zero-shot sleep staging |
| [ContraWR sleep-staging study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10715804/) | Sleep-specific self-supervision, limited-label evaluation, and the problem of false negative pairs | Guides low-label experiments and contrastive sampling diagnostics |
| [MC2SleepNet](https://arxiv.org/abs/2502.17470) | Raw EEG and spectrogram learning with contrastive learning and cross-masking | Particularly close related work; compare inputs, objective, training data, cost, and external protocol before claiming novelty |
| [MultiConsSleepNet](https://pubmed.ncbi.nlm.nih.gov/41336818/) | Within- and across-modality contrastive sleep learning | Helps distinguish your same-channel views from EEG/EOG multimodality |
| [U-Sleep](https://www.nature.com/articles/s41746-021-00440-5) | Evaluation across heterogeneous cohorts and channel configurations | A reference for external evaluation design, rather than a directly matched small-model baseline |

A potentially defensible contribution is the use and evaluation of a very small set of named physiological feature tokens, controlled waveform-to-feature attention, source-only pretraining, and a transparent external test under montage and annotation differences. Whether this is sufficiently novel remains a literature-review and experimental question.

Maintain a related-work table with: input views, self-supervised objective, number of participants, whether target signals entered training, whether target labels entered selection, parameter count, and evaluated datasets. Published results on two datasets do not by themselves establish training on one and zero-shot testing on the other. Add one reproducible published compact model and one feasible SSL method as matched comparators; avoid selecting methods solely by headline accuracy.

**3. Repair the foundation before extending the model. Days 1–4.**

Complete these repairs from `PROJECT_AUDIT_2026-10-26.md` first:

- Preserve physical epoch positions after excluded annotation intervals. Reproduce the SC4002 gap case and compare regenerated features with the existing verified cache.
- Make the sequence helper use row positions consistently, independent of DataFrame index labels.
- Use the corrected participant split throughout and archive the historical recording-level experiments.
- Give each model its matching scaler, ordered feature list, configuration, split manifest, and input hashes.
- Replace header-only cache validation with verified completion and preprocessing fingerprints.
- Set random seeds before constructing every model; save all reported checkpoints and histories.

Re-evaluate the corrected baseline under the new common data contract. The audit independently reproduced its 82.66% canonical accuracy and 65.34% macro-F1 on 69,539 source test windows. Use those values as a historical sanity check; changed preprocessing or target coverage requires fresh baseline evaluation.

**Exit criterion:** a clean command can regenerate a small reference recording, build aligned raw/features/labels, load the correct bundle, and reproduce its predictions. Alignment and provenance checks must pass before a large training job.

**4. Build one canonical raw-plus-features data representation. Days 3–7.**

Create a common epoch table for both datasets with at least:

```text
dataset, participant_id, recording_id, epoch_id
channel_as_recorded, derivation_used, original_sfreq, canonical_sfreq
start_time_seconds, original_start_sample, canonical_start_sample
canonical_end_sample_exclusive, quality_valid, exclusion_reason
raw_stage, mapped_stage, scorer_id, annotation_file
preprocessing_version, raw_file_hash
```

Use stable recording/epoch identifiers to join waveform, features, and labels. Do not infer correspondence from row order. Use an explicit contiguous-run identifier and a separate target identifier for every sequence.

Recommended common preprocessing for the first study:

| Decision | Initial policy |
|---|---|
| Task | Five-class offline sleep staging; classify the final epoch of a past/current context window |
| Sampling | 100 Hz canonical signal; anti-aliased resampling of ISRUC from 200 Hz |
| Epoch | 30 seconds, 3,000 canonical samples; preserve exact annotation/sample alignment |
| Filtering | The same declared 0.3–35 Hz offline filter in both datasets |
| Features | The current ten features, computed from the same filtered epoch used by the waveform branch |
| Spectral estimator | Preserve the declared Welch configuration, including 1,024-sample segment length at 100 Hz, overlap/window/detrending defaults made explicit |
| Spectral normalization | Log-transform the four absolute powers, then fit scaling on source training participants only |
| Waveform normalization | Convert physical units explicitly; use a source-fitted or physically fixed amplitude scale with documented mean removal |
| Quality policy | Signal-based, fixed using source development or physical criteria; report exclusion counts |
| Gaps | Preserve temporal positions; sequences never cross a recording, participant, or invalid-data gap |
| Context | Start with ten epochs to match the current five-minute baseline |

Do not independently standardize every epoch to unit variance by default: that discards amplitude information that your absolute-power features retain. If tested, treat it as a declared normalization ablation. Preserve a clear unit convention: changing volts to microvolts changes absolute power by a factor of 10^12. Existing scalers cannot be casually reused across that change.

Check PSG and annotation time origins, one-based versus zero-based epoch numbering, duration rounding, units, and resampling offsets. Verify selected epochs near the beginning, middle, excluded intervals, and end of representative recordings.

For self-supervised pretraining, construct the source training grid independently of stage labels and apply signal QC. Do not use stage labels to balance, crop, or select the unlabeled pretraining pool. The downstream supervised eligibility mask can exclude movement/unknown labels without compacting time. Record which signal data entered each stage.

The current full-record zero-phase filter uses future samples around each point. Retain an offline claim for this study; a causal real-time implementation is a separate protocol.

**5. Resolve the downloaded ISRUC dataset as an evaluation asset. Complete the specification in week 1.**

Your local data is ISRUC Subgroup I: 100 canonical numeric recordings, recorded at 200 Hz according to the compatibility report. Fpz-Cz is absent. This makes the target evaluation a joint cohort, acquisition, scoring, and montage transfer test.

The local channel inventory supports the following prospective policy:

| Analysis | Proposed channel policy | Expected eligibility from inventory |
|---|---|---|
| Primary | C3 referenced to the recorded contralateral mastoid/auricular reference, retaining actual derivation metadata | All 100 appear covered through stored or reconstructible derivations; verify samples and reference semantics |
| Sensitivity | F3 with the recorded contralateral reference | 99 appear covered; participant 8 lacks the frontal channel |

Do not rename C3-A2, C3-M2, or F3 derivations to Fpz-Cz. Auricular and mastoid reference naming must be documented rather than silently treated as identical. If separate electrode signals must be subtracted, first confirm units, sampling, polarity, and reference meaning; do not subtract again from an already bipolar signal. Compare central and frontal results on the same 99 eligible participants as an additional sensitivity analysis.

Choose the primary channel using anatomy, available metadata, and coverage before seeing accuracy. A “best ISRUC channel” selected after evaluating multiple channels would use target labels for model selection.

For annotations, the original dataset provides epoch/stage workbook descriptions; the [dataset's annotation documentation](https://dataset.isr.uc.pt/ISRUC_Sleep/Content.pdf) is the first reference. A curator-maintained [BIDS conversion](https://www.nemar.org/dataset/nm000111) uses scorer-1 Excel annotations, but that convention is supporting evidence, not proof that every local conflict is resolved.

Recommended initial annotation plan:

- Use scorer 1 as the primary reference and provisionally use its original workbook, subject to provenance and alignment verification.
- Preserve scorer 2 for sensitivity analysis and inter-scorer agreement, rather than optimizing against whichever scorer favors the model.
- Use explicit mappings to W/N1/N2/N3/REM. Preserve unknown labels separately; verify numeric codes against the chosen files and documentation.
- Resolve the known participant-18 scorer-2 TXT length mismatch: 1,006 text labels versus 999 signal epochs. Do not silently truncate or shift them.
- Resolve or document the 91 imperfect TXT-versus-Excel comparisons among 199 equal-length comparisons in the saved audit.
- Select one canonical recording for participant 65. An alternate file has a different hash; determine whether its signal content differs before calling it a duplicate.
- Retain a discrepancy table and exclusion log. Avoid blanket “remove the last 30 epochs” conventions without dataset-specific evidence.

Some local workbooks contain an explicit header; parsers must use epoch/stage content rather than assuming the first row is always data. The inspected scorer-1 workbook for participant 18 contains epochs 1–999, consistent with the signal-duration check.

On unknown labels, break supervised temporal runs for the primary conservative protocol. If a later label-independent inference protocol allows unknown-scored epochs as context, declare it in advance and evaluate only eligible targets; never stitch across missing signal intervals. Use identical masks and target sets for all compared models.

**Exit criterion:** a locked ISRUC manifest, canonical signal adapter, label-authority decision, channel policy, and discrepancy log. Engineering checks can inspect signal metadata and annotation structure; predictive target scores remain reserved for the frozen evaluation.

**6. Implement the compact multi-view architecture. Week 2.**

Use an epoch encoder followed by a small temporal model. This separates morphology learning from inter-epoch sleep context and makes pretraining feasible on 4 GB.

```text
One 30-second EEG epoch                 Ten engineered features
            |                                   |
     compact strided 1-D CNN          log/scale + feature tokenization
            |                                   |
     P waveform tokens, d=64             10 feature tokens, d=64
            |                                   |
            +---- waveform queries; spectral keys/values ----+
                                  |
                        gated residual fusion
                                  |
                         pooled epoch embedding
                                  |
                ten epoch embeddings in temporal order
                                  |
                    small GRU or existing stacked LSTM
                                  |
                       five-stage classification
```

Initial engineering specification, subject to a memory pilot:

| Component | Starting design |
|---|---|
| Raw input | `(B, L, 3000, 1)` with `L=10` |
| Feature input | `(B, L, 10)` |
| CNN | Three or four strided blocks; channels approximately 16/32/64; normalization independent of target-batch updates |
| Waveform token sequence | Approximately 30–60 tokens per epoch; retain some within-epoch timing |
| Shared token width | 64 |
| Spectral tokens | Ten tokens, one per named feature, with feature-identity embeddings |
| Cross-attention | One block, two or four heads |
| Fusion | Learned sigmoid gate plus a raw residual path |
| Temporal module | One 64-unit GRU initially; use the same temporal module across raw/fusion comparisons |
| Output | Five-class softmax for the final epoch |
| Size budget | Start below approximately 0.5 million parameters; measure actual size and runtime |

Keep the current stacked feature LSTM as a reference baseline. Use the same temporal context and targets for all new architectures; do not add a larger temporal transformer at the same time as introducing fusion and SSL.

For feature j, a simple tokenization is `t_j = W_j * standardized_feature_j + e_j`, where `e_j` identifies delta absolute, theta relative, entropy, and so on. The model can then distinguish feature meaning as well as value.

For waveform tokens R and feature tokens S:

```text
C = Attention(Q=R Wq, K=S Wk, V=S Wv)
G = sigmoid(MLP(concat(R, C)))
H = LayerNorm(R + G * C)
```

The gate explicitly controls the injected spectral contribution. The residual retains a waveform path. Neither mechanism guarantees improved accuracy, calibrated reliability, or a physiological explanation.

**Attention implementation trap:** if the entire ten-feature vector is embedded into one key/value token, attention over that axis has a softmax of length one. Its weight is always one, so query-dependent selection among spectral cues cannot occur. Use multiple feature/group tokens and verify the attention axes.

Start with raw queries and spectral keys/values within each epoch. Bidirectional cross-attention and cross-epoch spectral attention can be later ablations if needed. Keep timing and feature-identity information explicit.

Required fusion controls are raw-only, spectral-only, simple concatenation, and gated cross-attention. Compare parameter counts and add a reasonably matched raw/concatenation model if attention has materially greater capacity. If simple concatenation performs as well at lower cost, that is an informative result.

**7. Implement and validate self-supervised pretraining incrementally. Week 3.**

The first version should pretrain the waveform encoder on individual source-training epochs. Then initialize the downstream raw encoder from that checkpoint and train the fusion and temporal modules using source labels. This tests the value of SSL without requiring a complicated joint objective immediately.

Use two independently augmented views of the same epoch as the positive pair. A small projection MLP maps pooled CNN embeddings into a contrastive space. Train a standard normalized contrastive objective, such as NT-Xent, and use the encoder representation before the projector for downstream tasks. This follows the broad [SimCLR](https://proceedings.mlr.press/v119/chen20j.html) design; augmentation suitability must be evaluated for EEG, rather than inherited from images.

A manageable starting configuration is:

| Setting | Local pilot | Cloud pilot |
|---|---|---|
| Encoder | The same small waveform CNN used downstream | Same model |
| Projection width | 64 or 128 | Same width |
| Batch | Try 16–32 original epochs, each with two views | Try 64–128 if memory permits |
| Optimizer | Adam/AdamW; small source-only learning-rate search | Same search budget |
| Temperature | Small declared search, e.g. 0.1 and 0.2 | Same choices |
| Pretraining budget | Start with 10 epochs for diagnostics, then extend if useful | Time a pilot before choosing a full schedule |
| Checkpointing | Save encoder, projector, optimizer, RNG state, and sampler state | Save persistently to survive interruptions |

These batch sizes are starting trials, not guaranteed fits. Profile memory and throughput. Gradient accumulation increases the optimizer's effective batch but does not increase the number of negatives in each contrastive denominator. If small-batch contrastive learning is inadequate, consider a queue-based implementation or a noncontrastive reference as a separate method decision; do not silently call accumulated microbatches a large contrastive batch.

Treat augmentations as physiological assumptions:

| Candidate | Initial treatment |
|---|---|
| Mild amplitude scaling | Test a narrow range on source development; preserve or explicitly account for spectral-power changes |
| Low-amplitude additive noise | Set strength relative to signal statistics; inspect stage-specific degradation |
| Short time masking | Keep masks limited; inspect whether useful transient morphology disappears |
| Small within-epoch shifts | Preserve the epoch; avoid wraparound discontinuities and crossing epoch boundaries |
| Strong frequency masking/filtering | Defer; can remove the very cues used for sleep staging |
| Large time warping, segment permutation, heavy cropping | Defer; can change frequency content, chronology, or stage evidence |

The initial raw-only pretraining avoids a cross-view mismatch. If you later pretrain both views, either compute features from the same augmented waveform or explicitly define a clean-versus-corrupted consistency task. For amplitude scaling by a, absolute power changes by a^2; relative powers ideally remain unchanged under pure scaling. Add tests for those transformations before using them in a multi-view loss. Arbitrarily pairing modified waveforms with incompatible clean features can teach the wrong invariance.

For sampling, avoid treating temporally adjacent or overlapping windows as arbitrary negatives. Mix participants and recordings while retaining some temporally separated within-participant negatives; using only different participants as negatives can make participant identity a shortcut. A fixed temporal exclusion zone is a useful candidate, but its size must be chosen with source data. Same-stage epochs remain possible false negatives without labels. The sleep-specific [ContraWR study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10715804/) is relevant to this limitation.

Do not automatically assume adjacent epochs form positive pairs: the current system struggles with N1, where transitions matter. Monitor embedding variance/effective rank, nearest neighbors, and participant predictability. Use a frozen-encoder linear probe as well as full fine-tuning. Lower SSL loss alone is not evidence of better sleep representations.

**Optional second SSL version, after a successful raw-only experiment:** pretrain raw and spectral encoders with raw-view consistency plus same-epoch cross-view alignment, using separate projection heads. Retain a waveform-specific residual representation so alignment with ten summaries does not force the encoder to discard morphology those summaries cannot express. Treat loss weights and negative-sampling policy as explicit ablations. Cross-view alignment is a proposed hypothesis, with close precedent in TF-C and MC2SleepNet; adding it is not automatically a novel contribution.

**8. Keep the external-evaluation boundary explicit.**

For the primary study:

| Data | SSL fitting | Supervised fitting | Model/threshold selection | Final scoring |
|---|---|---|---|---|
| Sleep-EDF training participants | Yes, labels hidden during SSL | Yes, within the chosen label budget | Training diagnostics only | Training diagnostics |
| Sleep-EDF validation participants | No | No | Yes | Source validation |
| Existing Sleep-EDF test participants | No | No | No new selection | Historical/held-out source evaluation, with previous inspection disclosed |
| ISRUC-I | No | No | No | Yes, after protocol/checkpoint freeze |

Repeat normalization fitting and pretraining inside the training partition whenever source folds change. Pretraining on source validation or test signals would change the claim even without using their labels. The existing source test has already been inspected; reshuffling the same participants does not erase that history.

Using unlabeled ISRUC for SSL, adversarial alignment, pseudo-labeling, batch-normalization adaptation, or test-time training would become a different adaptation experiment. It can be useful later, but should be reported separately from this source-only external test.

At evaluation, freeze preprocessing parameters, feature scalers, normalization layers, checkpoints, class order, thresholds, ensembles, channel rules, and exclusion policies. Deterministic signal processing needed to pass a new record through the fixed model is allowed; fitting target-cohort scalers or updating model statistics is outside the primary policy. Any predefined per-record normalization that estimates statistics from that record must be disclosed, and should not be confused with source-fitted scaling.

Keep target labels in an evaluation-only path. For the final batch of predeclared models, compute and report all results together. If the external results motivate new model development, record that the next iteration has used ISRUC feedback; obtain a new independent cohort for a fresh external confirmation.

**9. Use an experiment matrix that separates the contributions.**

The complete core matrix is:

| ID | Inputs/fusion | Initialization | Purpose |
|---|---|---|---|
| M0 | Current spectral features + LSTM | Random/reproduced baseline | Continuity with the current project |
| M1 | Waveform CNN + temporal module | Random | Value of waveform learning |
| M2 | Waveform + features, concatenation | Random | Value of simple additional guidance |
| M3 | Waveform + feature-token gated attention | Random | Attention versus simple fusion |
| M4 | Waveform CNN + temporal module | Source-only raw SSL | SSL benefit without feature fusion |
| M5 | Waveform + features, concatenation | Same raw SSL | SSL plus simple fusion control |
| M6 | Waveform + feature-token gated attention | Same raw SSL | Full first-version proposal |

M2/M3/M5/M6 form a useful 2-by-2 experiment: fusion type versus initialization. They help distinguish an SSL benefit from an attention benefit. M4 tests whether physiological guidance adds anything beyond a pretrained waveform encoder. Reuse identical pretrained checkpoints across corresponding fine-tuning comparisons, and pair seeds and labeled subsets.

Implement configurations and experiment-runner support for all seven core conditions during the month. Validate every training path with short runs, then screen with one seed and repeat principal comparisons with at least three seeds for the research campaign. Prioritize longer runs for the four factorial conditions and the strongest raw-only control as compute becomes available. Code completion requires working comparison paths; completion of every full-budget repeated run is a separately tracked experimental milestone. Label single-seed analyses exploratory, keep model-selection opportunities comparable, and disclose extra SSL compute.

Implement configurable participant-level label budgets and reproducible subset draws during the month. Validate them first at 25% and 100% of source training participants. On the existing 70-participant training split, that means approximately 18 and 70 participants; report actual counts and labeled hours. The same implementation should support 10% (7 participants) and 50% (35 participants), with full experiment execution prioritized by compute availability. Use whole participants and their recordings, not random overlapping windows. Prespecify several nested subset draws, check that supervised training has the required classes using source training labels, and use identical draws for every comparison.

All source training participants can contribute unlabeled signals to SSL even when only a subset contributes stage labels; that is the intended label-efficiency setup. The same eligible unlabeled pool must be available to compared SSL methods. Avoid constructing stage-balanced “unlabeled” batches from hidden labels.

The validation annotation budget also matters: if all 15 validation participants remain fully labeled, say that training-label efficiency is being evaluated with a fixed fully labeled validation set. Count and report validation labels; do not imply that the stated fraction is the total annotation cost. A later stricter experiment can budget validation labels as well.

**10. Specify evaluation and analysis before viewing external scores.**

For both datasets, save per-epoch records with participant, recording, epoch, time, true stage, prediction, five probabilities, scorer, run ID, and eligibility mask.

Report:

- Participant-mean five-class macro-F1 as the primary external endpoint; define absent-class handling before scoring and show class-presence/support tables. If using fixed five-class scoring, disclose how a missing class contributes.
- Pooled macro-F1, balanced accuracy, accuracy, weighted F1, Cohen's kappa, and per-stage precision/recall/F1.
- Participant distributions and paired model differences, with participant-cluster bootstrap intervals. Retain both nights together for Sleep-EDF.
- Variation across training seeds separately from participant uncertainty. Do not treat participant-seed pairs as independent participants.
- Confusion matrices, N1 and REM performance, and errors near real stage transitions.
- Scorer-1 primary results, scorer-2 sensitivity results where alignment is valid, and inter-scorer agreement on their common eligible epochs.
- Calibration, for example negative log likelihood and Brier score; any temperature fitting uses source validation only.
- Excluded targets, run-start coverage, preprocessing failures, and identical target masks across compared models.
- Parameter count, peak memory, feature-extraction time, inference latency, and total pretraining/fine-tuning compute on declared hardware.

Do not interpret raw source-minus-target score differences as a pure measure of domain invariance: class prevalence, montage, population, reference scorer, recording duration, and valid-target policy all differ. Compare models within ISRUC on the same subjects and masks. Secondary label-informed crops or demographic analyses are analyses, not inputs for model selection.

Attention maps can help describe model behavior, but do not establish explanations by themselves. Pair them with branch removal, feature-group ablations, and small physically coherent signal perturbations followed by feature recomputation. Shuffling features between epochs is a useful mismatch diagnostic, but creates inconsistent views and is not a realistic physiological intervention. Treat artificially corrupting only one branch as an explicit branch-failure test.

**11. Engineer for 4 GB and use cloud compute selectively.**

The local GPU is sufficient to attempt a small architecture and debug the full workflow, provided the framework can actually use it. It should not be assumed to support full-scale training until measured. Your current environment uses modern TensorFlow on Windows; native Windows CUDA GPU support ended after TensorFlow 2.10. For a compatible NVIDIA GPU, TensorFlow GPU training needs a supported Linux/WSL2 setup or a cloud Linux environment. Verify actual device visibility first. [TensorFlow installation guidance](https://www.tensorflow.org/install/pip).

Keep TensorFlow/Keras for the initial extension unless a specific reproduced baseline requires another framework. A simultaneous framework rewrite adds risk to the month-long plan.

Use these memory controls:

- Pretrain single epochs first; fine-tune ten-epoch sequences with a small batch, initially 4–8 sequences if needed.
- Apply a strided CNN before attention. Full attention over all 3,000 samples per epoch is unnecessary for the first model.
- Use mixed precision only after confirming device support and numerical stability; compute contrastive similarity/loss reductions in float32 if needed.
- Stream windows from per-recording float32 arrays or shards. Avoid eagerly duplicating overlapping raw windows.
- Cache epoch embeddings only when the encoder is frozen, such as a linear probe or temporal-head pilot. End-to-end fine-tuning requires current encoder outputs.
- Time I/O and augmentations as well as GPU operations. Avoid a data pipeline that starves the GPU.

At the current 457,577 labeled-epoch count, one 3,000-sample float32 channel occupies approximately 5.49 GB before metadata. Eagerly materializing ten-epoch overlapping windows can multiply waveform storage toward roughly 55 GB. A shared epoch store plus sequence indices avoids most of that duplication. The complete label-independent pretraining grid can be somewhat larger than the labeled cache.

Cloud training means running the same code on a remote GPU computer. Your local GPU memory does not limit that machine. Upload the source-training data shards, code, environment specification, and manifests; save checkpoints/logs persistently and retrieve them afterward. ISRUC does not need to be uploaded for source training and can be evaluated separately after the models are frozen.

| Route | Suitable use | Practical trade-off |
|---|---|---|
| Local 4 GB | Data preparation, debugging, compact pilots, some fine-tuning | Small batches and potentially slow experiments |
| Google Colab | Easiest notebook-based cloud pilot | GPU type, availability, and session duration vary; persist checkpoints |
| Dedicated rented GPU instance | Repeated scripted training with a selected memory capacity | More setup and explicit compute/storage billing |

[Colab's official FAQ](https://research.google.com/colaboratory/faq.html) explains its variable resource limits. Free access is useful for a pilot, but implementation validation and the subsequent experiment campaign should not depend on guaranteed free GPU availability.

For a paid comparison point, [Runpod's displayed pricing](https://www.runpod.io/pricing) on the audit date listed an RTX 3090 with 24 GB at $0.50/hour; other hardware and deployment options differ. At that example rate, 40 GPU-hours would cost $20 for compute, and 100 hours $50, plus applicable storage and other charges. These are arithmetic illustrations, not estimates of how long your study will run or a guaranteed quote. Confirm the selected instance rate before renting. Stopped-instance storage can still incur charges; persist critical outputs separately. [Provider storage/billing explanation](https://www.runpod.io/blog/where-did-my-files-go-a-straight-guide-to-runpod-storage).

Recommended cloud workflow:

1. Finish the local pipeline and a small successful training run.
2. Package source-training shards and verify upload hashes.
3. Run a 30–60 minute pilot on an accessible cloud GPU, aiming for 16–24 GB memory if available.
4. Record examples/second, GPU utilization, memory, time per epoch, and checkpoint size.
5. Estimate the remaining schedule from measured runtime and the actual experiment count. Set a spending limit before full runs.
6. Checkpoint regularly, export results, and stop compute when finished; account for retained storage.

A useful estimate is `total cost = GPU hours × selected hourly rate + storage + other applicable charges`. Derive GPU hours from measured pretraining and fine-tuning runs, including interrupted/repeated jobs. Avoid buying a large subscription or planning hundreds of runs before the pilot.

**12. Organize the implementation so the paper can be regenerated.**

Extend the existing project with a small set of focused modules and configuration files. These are proposed locations, not files already implemented:

```text
configs/
  preprocessing.yaml
  source_split.yaml
  supervised_raw.yaml
  supervised_fusion.yaml
  ssl_raw.yaml
  external_isruc.yaml
manifests/
  recordings.csv
  epochs.csv
  participant_splits.csv
  annotation_discrepancies.csv
src/data/
  adapters/sleepedf.py
  adapters/isruc.py
  epoch_store.py
src/models/
  waveform_encoder.py
  spectral_tokens.py
  fusion.py
  temporal.py
src/pretraining/
  augmentations.py
  sampling.py
  contrastive.py
src/training/
  pretrain.py
  finetune.py
src/evaluation/
  source.py
  external.py
  participant_statistics.py
runs/<run_id>/
  resolved_config.json
  data_and_code_manifest.json
  checkpoints/
  training_history.csv
  predictions.csv
  metrics.json
```

Keep adapters responsible for dataset-specific naming/reference/annotation details. Shared signal processing should implement the common policy. Model code should receive aligned tensors and identifiers, rather than opening EDF files. Notebooks should inspect outputs and produce figures from saved artifacts.

Each run needs the resolved configuration, software versions, split hashes, preprocessing hash, random seeds, initialization checkpoint, labels used, eligible data, and code revision. Record uncommitted changes when present. Keep external labels outside the training data loader's accessible schema.

Meaningful checks include gap alignment, raw/feature correspondence, feature scaling under amplitude changes, deterministic inference bundle loading, no participant overlap, no forbidden pretraining participants, identical target masks, model serialization, and a tiny-batch overfit sanity check. Add a test proving that cross-attention sees ten key tokens and has the expected shape.

**13. A one-month implementation schedule and acceptance criteria.**

| Period | Required work | Deliverable and decision gate |
|---|---|---|
| Week 1 | Repair alignment/indexing/provenance; implement shared raw/feature storage and dataset adapters; specify ISRUC channels and annotations; verify GPU/cloud pilot | Reproducible baseline, passing data checks, participant manifests, and the external specification |
| Week 2 | Implement waveform encoder, feature tokenization, concatenation and gated cross-attention, temporal head, supervised trainer, serialization, and inference | All supervised variants train on a small source subset, reload correctly, and produce aligned predictions; record memory and throughput |
| Week 3 | Implement augmentations, contrastive objective and sampler, source-only pretraining, checkpoint transfer, linear probes, fine-tuning, and label-budget sampling | Demonstrated pretrain-to-fine-tune execution, successful checkpoint resume, and checks excluding forbidden participants from fitting |
| Week 4 | Implement external inference/scoring separation, participant statistics, experiment configurations, result export, and figure-generation scripts; complete integration tests and usage documentation | An executable code release covering all three approaches and core comparison paths, with a validated end-to-end run and a queued/recorded full experiment plan |
| Remaining days, if using a 30-day month | Resolve integration defects, verify a clean environment, and rerun the release acceptance workflow | Reproducible implementation handoff; paper writing begins after this point |

Training should begin during weeks 2 and 3 as stable components become available. Profile a full-data epoch early enough to identify compute constraints before the last week. Target at least one complete source training/pretraining-to-fine-tuning workflow before handoff, with its actual data size and training budget documented. Short smoke tests alone do not validate convergence or the research hypotheses; full-budget run completion and scientific conclusions must remain explicit in the experiment ledger.

The code release is complete when:

- Both dataset adapters and the shared raw/feature data contract pass alignment, unit, gap, and provenance checks.
- The spectral baseline, raw-only model, concatenation fusion, and gated-attention model have working training and inference configurations.
- Source-only contrastive pretraining, checkpoint restoration, initialization transfer, linear probing, and supervised fine-tuning run successfully.
- Participant splits, label-budget subsets, seeds, and preprocessing parameters are reproducible and recorded.
- The experiment runner supports M0–M6, repeated seeds, and configured label budgets without notebook edits.
- ISRUC inference uses the frozen model/scalers and predetermined channel rules; the scoring path is separate from training and model selection.
- Metrics, participant-cluster uncertainty, prediction exports, and paper-figure scripts operate on saved artifacts.
- A fresh environment can execute the documented acceptance workflow, including checkpoint reload and inference agreement, with measured memory/throughput and a record of completed versus pending large runs.

During integration, test external plumbing using synthetic scoring fixtures and unlabeled ISRUC forward passes. Reserve predictive ISRUC scoring for the prespecified frozen model batch; if development is still changing the models, do not inspect target scores just to demonstrate that the evaluator runs. External scoring can occur within the implementation month if the relevant models and protocol are already frozen.

The month covers the complete first-version implementation: supervised raw/spectral fusion, raw-encoder contrastive SSL, and source-only external evaluation. Joint raw/spectral SSL, bidirectional fusion, larger temporal models, expanded source cross-validation, additional cohorts, and target-domain adaptation remain extensions. Decide on them using source evidence and remaining resources. If attention matches concatenation or SSL underperforms, retain the comparison and diagnose using source data; implementation completion does not require a positive result.

**14. After implementation: experiments and the paper package.**

Paper writing starts after the implementation period. Use the saved experiment ledger to identify which repeated runs and analyses are complete and which must finish before results can be claimed. The release should already include the scripts needed to produce the research artifacts below.

The study should produce an architecture/data-flow figure, a dataset and protocol table, source and external model-comparison tables with uncertainty, a label-budget curve, stage-wise results, paired participant improvements, a cost table, and a small set of explanatory diagnostics. All should be generated from saved predictions and manifests.

In the manuscript, distinguish hypotheses, method choices, and observed results. Do not promise a target accuracy before experiments. Strong source results with a large target drop are not evidence of generalization; an external improvement with honest uncertainty and competitive cost may be more valuable than an elaborate model with a higher source score.

A working framing is: **Physiology-informed dual-view EEG representations for label-efficient sleep staging under cross-dataset shift.** Treat this as a research framing, not a novelty claim or final title. The first concrete milestone is a clean raw-plus-feature data contract and an external-evaluation specification that can be trusted before the model grows.
