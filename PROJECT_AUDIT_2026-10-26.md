**Assessment: a credible compact research baseline, but not ready for journal submission.**

The corrected saved model produces the reported results when evaluated independently. The project has useful engineering and experimental work behind it: participant isolation, training-only normalization, temporal gap handling, validation-based configuration selection, stage-specific metrics, and comparisons across training seeds. These are substantive strengths.

The main obstacles are a confirmed preprocessing regeneration bug, conflicting experiment artifacts, incomplete reproducibility, limited evidence about generalization, and an insufficiently differentiated scientific contribution. These issues are repairable. I would prioritize them before adding architectures or polishing publication figures.

Audit date: 26 October 2026. Reviewed the current working tree, including the four notebooks already modified before this audit. No existing source, notebook, model, dataset, or experiment result was changed. Audit evidence is in `reports/audit_2026-10-26/`.

**What was checked and what remains unverified**

The review covered the source modules, nine notebooks and their stored outputs, model/configuration/scaler packages, experiment and result tables, dataset compatibility scripts/reports, repository documentation, dependency declarations, and local data organization. Runtime checks included all 197 Sleep-EDF PSG/hypnogram pairings, all cached epoch labels against annotation intervals, feature spot checks against raw recordings, both corrected model packages on the complete test window set, inference API agreement, and targeted reproductions of implementation defects.

The full training search and three-seed training were not rerun. Feature values were independently recomputed for 16 epochs across three recordings, not every EEG epoch. The ISRUC discussion below relies on inspection of the existing compatibility report and audit implementation; a new complete ISRUC signal audit or model evaluation was not performed. Literature checks are a focused comparison, not a systematic novelty review. No manuscript was present to audit.

**Results that are supported**

The feature table contains 457,577 rows, 197 recordings, and 100 filename-derived participants. It has no missing values or duplicate `(subject_id, epoch_index)` pairs. All stored epoch starts equal `30 * epoch_index`; cached labels matched the corresponding hypnogram intervals for every row. The participant mapping is consistent with the documented Sleep-EDF filename convention. [PhysioNet dataset documentation](https://physionet.org/content/sleep-edfx/1.0.0/sleep-cassette/).

The corrected split has 70 training, 15 validation, and 15 test participants, with no participant overlap. It contains 138, 29, and 30 recordings respectively. Refitting the scaler on the corrected training data reproduced its saved means and scales exactly.

The selected system uses ten features, ten 30-second epochs per window, two 32-unit LSTM layers, and 13,989 model parameters. Each prediction classifies the last epoch in its window; this is current-epoch staging, not forecasting a future stage.

| Metric | Independently reproduced canonical model | Saved three-seed mean ± sample SD |
|---|---:|---:|
| Accuracy | 82.663% | 82.401% ± 0.228 percentage points |
| Balanced accuracy | 65.172% | 65.399% ± 0.242 percentage points |
| Macro-F1 | 65.337% | 65.259% ± 0.072 percentage points |
| Weighted F1 | 82.449% | 82.335% ± 0.121 percentage points |
| Cohen's kappa | 0.68519 | 0.68231 ± 0.00288 |
| Macro one-vs-rest ROC-AUC | 0.94511 | 0.94496 ± 0.00014 |

The independent evaluation used all 69,539 corrected test windows. It exactly reproduced the saved confusion matrix and the canonical model's classification metrics. Both `final_model/stacked_lstm_32_32_10_epochs.keras` and `final_model_corrected/stacked_lstm_final.keras` yielded the same evaluated results. Their archive hashes differ; differing archive contents alone do not imply differing predictions. The inference API agreed with the direct model prediction in the sampled window to approximately 2.5e-9 in probability.

The existing validator passed with:

```powershell
.venv\Scripts\python.exe -X utf8 -m src.evaluation.validate_final_model
```

Without `-X utf8`, the validator failed while printing the intersection symbol under this session's Windows stdout encoding. This is a portability issue, not a failed scientific check.

**1. High priority: regenerating features with the current code can misalign EEG and labels.**

Evidence: `src/preprocessing/annotations.py:84` initializes a compact epoch counter and line 137 increments it only for retained annotations. `src/preprocessing/epoching.py:95` and line 173 subsequently treat this counter as an index into the original continuous recording. These meanings diverge after an excluded movement/unknown interval. The continuity checks later in sequence construction cannot repair features extracted from the wrong signal interval.

A synthetic recording with Wake at 0–30 seconds, Movement at 30–60, and N2 at 60–90 generates retained indices `[0, 1]`. Extraction selects the first and second EEG segments; it should select the first and third. This was reproduced at runtime.

For real recording SC4002, the first discrepancy is a row labeled with start time 28,110 seconds but assigned index 936, which extracts EEG starting at 28,080 seconds. Across all 197 recordings, the current metadata function produces 92,570 index/time mismatches in 80 recordings, before quality filtering.

**This does not establish that the cached training data is corrupted.** The cached table preserves the proper timeline indices, all cached label/onset checks passed, and recomputed features from 16 epochs in SC4001, SC4002, and ST7012 matched the cached values, including epochs after the SC4002 gap. The evidence instead establishes that the current implementation does not reliably regenerate its own existing dataset.

Recommended repair: derive extraction bounds from annotation onsets, or preserve absolute grid indices with explicit grid validation. Separate a row identifier from the physical signal position. Add regression checks for excluded middle intervals, nonzero starts, recording boundaries, and flat-epoch exclusions. Regenerate into a new versioned directory and compare with the preserved cache before replacing anything. Whether retraining is necessary depends on those comparisons.

**2. High priority: incompatible historical and corrected results coexist as “final.”**

The older `data/features/subject_split.csv` assigns recordings from 43 participants to multiple subsets. The corrected `participant_split.csv` has zero overlap. Notebooks 02–04 still contain the older workflow, while notebook 06 implements corrected participant-level development.

| Existing artifact | What it currently represents |
|---|---|
| `results/final_evaluation/final_metrics.csv` | Older 20-epoch model; 87.396% accuracy, 71.457% macro-F1 |
| `results/final_evaluation/final_predictions.csv` | Older evaluation; 67,017 rows |
| `results/final_evaluation/final_classification_report.csv` | Older evaluation; 67,017 supported examples |
| `results/final_evaluation/final_results_summary.json` | Older three-seed summary; 86.988% accuracy, 71.126% macro-F1; calls 197 recordings “subjects” |
| `results/final_evaluation/tables/` | Includes older experiment and final performance tables |
| `results/final_evaluation/final_confusion_matrix.csv` | Corrected evaluation; 69,539 windows; 82.663% accuracy |
| `results/final_evaluation/definitive_experiment_record.json` | Corrected 10-epoch participant-level three-seed summary |
| `results/corrected_model_selection/` | Corrected validation-based configuration selection |

The old prediction CSV cannot generate the current final confusion matrix. Using both in a manuscript would describe inconsistent experiments. Rerunning notebook 04 also writes its older scaler, configuration, and results into paths used by the corrected pipeline.

Recommended repair: archive the old workflow explicitly; give each experiment an immutable run directory and manifest; select one canonical corrected release. Generate every table and figure from its recorded predictions/configuration. Do not use the older numbers to claim performance on unseen participants. Also do not attribute the entire old/new score difference to leakage: the split, window length, and fitted models changed together.

**3. High priority: the inference example loads an incompatible model/configuration combination.**

`notebooks/05_inference_test.ipynb:55` loads `stacked_lstm_32_32.keras`, whose input is `(None, 20, 10)`, alongside the current configuration specifying ten epochs and the corrected scaler. The predictor constructor checks class names but does not verify model input shape against configuration. A ten-epoch prediction with that combination fails at runtime with the expected 20-versus-10 shape error. The notebook also uses the older recording split.

Recommended repair: make the inference example load the canonical ten-epoch model and its matching scaler/config/split. Validate model dimensions, ordered features, classes, and bundle identity at initialization. The corrected bundle's inference path itself passed the spot check.

**4. Medium priority: the sequence API silently confuses DataFrame labels with array positions.**

`src/preprocessing/sequences.py:143` obtains `run.index`, then uses those values as NumPy row positions at line 162. The documented contract only requires row-for-row alignment. It does not require a zero-based RangeIndex.

Runtime reproduction: three chronologically ordered rows with index labels `[2, 0, 1]` and aligned values `[10, 20, 30]` produce windows `[[30, 10], [10, 20]]` instead of `[[10, 20], [20, 30]]`. Other index labels can cause an out-of-bounds error. Notebook 06 resets indices, so the reproduced canonical evaluation is protected from this particular defect.

Recommended repair: create an internal positional column or reset indices inside the helper before grouping/sorting, while preserving explicit metadata for target alignment. Test shuffled labels and nonzero indices.

**5. Medium priority: cache validation and processing failure behavior can silently change the cohort.**

`src/data/process_all.py:67` claims to identify completed feature files but reads only their headers. A CSV containing the expected header and zero rows was accepted in a runtime check. A partially written CSV or a file generated with different preprocessing parameters can also pass. There is no source/configuration fingerprint or atomic completion marker.

Separately, `src/data/process_dataset.py:169` catches individual recording failures and can overwrite the combined feature table with the successful subset. `process_all.py` instead stops on failures. Two entry points therefore have different dataset-integrity behavior.

Recommended repair: use one authoritative processing command; write files atomically; validate rows, unique indices, finite values, and expected coverage; record raw-file hashes and preprocessing parameters. Fail before publishing a combined dataset if expected recordings are missing.

**6. Scientific limitation: performance is uneven, and overall accuracy is dominated by Wake.**

Wake represents 63.33% of cached epochs and 64.07% of corrected test targets. An always-Wake classifier therefore scores 64.07% accuracy on those targets. The learned model clearly improves on that baseline, but accuracy alone overstates its ability to distinguish all five stages.

| Stage | Saved three-seed mean F1 |
|---|---:|
| Wake | 94.72% |
| N1 | 34.19% |
| N2 | 69.57% |
| N3 | 73.09% |
| REM | 54.72% |

Canonical N1 recall is 26.34%: about three quarters of N1 targets are missed. REM is another substantial weakness. A macro ROC-AUC around 0.945 measures ranking performance; it does not resolve the weaker multiclass decisions or establish calibration.

The long Wake proportion is understandable: the Sleep Cassette recordings include roughly 20-hour day/night periods. This is a legitimate evaluation population if explicitly defined, but it should not be equated with a differently cropped Sleep-EDF benchmark. [PhysioNet acquisition description](https://physionet.org/content/sleep-edfx/1.0.0/sleep-cassette/).

Recommended reporting: lead with macro-F1, balanced accuracy, per-stage precision/recall/F1, and kappa. State SC/ST inclusion, wake retention, excluded stages, flat-epoch policy, and window coverage. Specify that the first nine epochs of each contiguous run lack a ten-epoch prediction; the current test set excludes 614 otherwise valid epochs for this reason. Any alternative sleep-window evaluation should be prespecified and reported separately, including whether it uses reference annotations to define the window.

**7. Scientific limitation: three seeds quantify training variability, not population uncertainty.**

All three reported seeds use the same 15 test participants. Their small standard deviation says little about sensitivity to which people were held out. Overlapping windows and paired nights also preclude treating 69,539 targets as independent patients.

For this audit, an exploratory percentile bootstrap resampled the 15 participants with replacement, retaining both nights and all their windows, and recomputed pooled metrics from summed confusion matrices. With 5,000 resamples and RNG seed 42, the fixed canonical model gave:

| Metric | Exploratory participant-cluster bootstrap 95% interval |
|---|---:|
| Accuracy | 79.27%–85.65% |
| Macro-F1 | 60.77%–69.58% |

These intervals are conditional on this fitted model and small cohort. They do not include uncertainty from retraining or model selection and are not a substitute for independent validation. They demonstrate why a 0.072-percentage-point seed SD should not be presented as generalization precision.

Recommended design: use participant-grouped development folds, paired participant-level comparisons, and a frozen external evaluation. Keep all preprocessing fitting and hyperparameter selection inside development folds. Prespecify metrics and label handling for participants missing a stage. The existing test set has already been inspected; future development should not treat it as untouched.

**8. Experimental limitation: the current ablations do not establish that each chosen component is necessary.**

Notebook 06 correctly selects configurations on validation metrics, with early stopping on validation loss. That distinction is valid and should be documented precisely. The search is sequential: temporal context, weighting, capacity, then architecture. It is not a factorial experiment and does not establish globally optimal combinations.

The five-minute and ten-minute contexts differ by only about 0.165 percentage points in validation macro-F1, from single runs. Such a margin is not strong evidence of superiority. Longer windows also exclude more run-start targets, so context comparisons should evaluate an identical target set where possible.

Missing controls include a no-context model using the same features, conventional feature classifiers, feature-group ablations, and a contemporary compact EEG model under a matched protocol. Without those controls, the project does not yet establish how much benefit comes from recurrence, the ten-feature choice, or stacking.

There is also an initialization-order issue in the selection loops: candidate models are built before `train_model()` calls `set_experiment_seed()` (`06_participant_split_model.ipynb:2148`, `:2826`, and analogous loss/capacity/architecture cells). Seeding after construction does not reset initialized weights. The final three-seed helper seeds before construction, so this criticism specifically concerns the search procedure. Move seed setup before every build and record execution/environment settings. [Keras reproducibility guidance](https://keras.io/examples/keras_recipes/reproducibility_recipes/).

**9. Scientific limitation: ISRUC is present, but external validation has not been implemented.**

The local v3 compatibility report identifies 100 canonical ISRUC-I recordings at 200 Hz. It does not provide model predictions or external performance. Its channel inventory lacks Fpz-Cz, so a frontal/mastoid derivation is not the same input montage as the Sleep-EDF channel.

The report also identifies a scorer-2 text-label length mismatch for participant 18, 91 imperfect TXT-versus-Excel comparisons among 199 equal-length comparisons, differing unknown-stage handling, and an extra recording for participant 65 whose file hash differs from the canonical recording. The top-level v3 script docstring calls that extra file a duplicate; byte identity was not established by its own report, and sample-level equivalence would need separate verification. These are issues to resolve, not reasons to discard ISRUC.

Recommended protocol: choose and document the annotation authority and primary scorer; resolve discrepancy rules before testing; explicitly map available channels; standardize sampling/units and Welch segment duration; preserve gaps; use the frozen source-trained scaler for zero-shot evaluation. Report adaptation or target-domain training separately. Treat montage, population, scoring-rule, and sampling differences as separate sources of domain shift. Do not tune on the external test labels.

**10. Reproducibility limitation: the paper cannot yet be regenerated cleanly from a fresh checkout.**

`requirements.txt` lists unpinned packages and omits a Python version. The README ends in an unclosed methodology code fence and lacks a complete data-to-results execution guide. The authoritative training procedure is interleaved with exploratory notebook code. Two notebooks retain stored `NameError` outputs. No automated regression suite or CI workflow was found.

The corrected split is bundled in `final_model_corrected`, but notebook 06 reads another copy under ignored `data/features`. The seed-123 and seed-456 corrected checkpoints, their probability predictions, and their histories are not present as independently reloadable artifacts. The existing old prediction CSV also lacks recording/participant/timestamp identifiers.

The publication notebooks manually embed many metric arrays. The embedded canonical confusion matrix matches the independently reproduced matrix, so hardcoding is not evidence of invented results. It does create avoidable provenance and staleness risks when results change.

Recommended release: pinned environment; one configuration-driven preprocessing/training/evaluation workflow; tracked split manifests; model/scaler/config/input hashes; all reported seed artifacts; prediction rows with participant, recording, epoch, label, probabilities, and run ID; figures generated from those files. Preserve the current notebook work as exploration. Extend the validator to recompute evaluation metrics and verify feature extraction alignment, not only counts/shapes/scaling statistics.

**11. Scope limitation: “compact,” “interpretable,” and “real-time” need different evidence.**

The 13,989-parameter count supports a compact-network description. It does not measure total deployment cost: continuous filtering, Welch features, normalization, buffering, inference, memory, and runtime must be included. Record timing on declared hardware and compare competing pipelines under the same conditions.

Named spectral features are understandable, but they do not automatically explain the LSTM's decisions. Use feature-group ablations or carefully designed perturbations if interpretability is a central claim; correlated absolute/relative powers complicate naive importance rankings.

The current filter is a zero-phase FIR applied to a complete recording (`src/preprocessing/filtering.py`). Although the window ends at the target epoch, that preprocessing uses samples on both sides of each time point. A real-time or strictly causal claim would require a different filtering/buffering protocol and latency measurements. Offline staging is an appropriate description of the current pipeline.

**Publication prospects and the research question**

My judgment is that “spectral features plus stacked LSTM for five-stage sleep scoring” is currently an implementation and evaluation study, not yet a sufficiently distinct method contribution. Recurrence for sleep staging has substantial precedent in [DeepSleepNet](https://pubmed.ncbi.nlm.nih.gov/28678710/), and efficient single-channel models were already central to [TinySleepNet](https://ieeexplore.ieee.org/document/9176741/).

The current literature also makes model size alone a difficult novelty argument. [PicoSleepNet](https://doi.org/10.1109/JBHI.2025.3602502) reports models with approximately 14.0–25.8 thousand parameters, and [DistillSleep](https://academic.oup.com/sleep/article/48/12/zsaf240/8239690) studies device-level inference, interpretability, and independent-cohort evaluation. These are relevant comparators, not directly comparable scores: dataset populations, montages, splits, wake policies, and scoring protocols must first be matched. Broader cohort evaluation is illustrated by [U-Sleep](https://www.nature.com/articles/s41746-021-00440-5).

A plausible direction is: **How much stage-specific accuracy and cross-cohort robustness can a very small spectral-feature sequence model retain, at a measured computational cost?** That question can support a useful empirical paper if answered rigorously, even without claiming a new neural architecture. Another possibility is a controlled study of how temporal context and feature groups behave under participant and dataset shift. A strong result could be a clear limitation or failure analysis, not necessarily a leaderboard win.

An appropriate journal target cannot be judged reliably without the completed contribution and manuscript. I would not promise acceptance, and I would not submit the current artifacts as they stand.

**Recommended order of work**

1. Preserve the verified cached data and corrected model. Repair annotation/extraction alignment and sequence indexing. Add focused regression checks and compare regenerated features before deciding whether to retrain.
2. Freeze one corrected release and archive the leaked historical protocol. Fix the inference example and make final metrics, predictions, confusion matrices, configurations, and figures agree automatically.
3. Define one research question and protocol before further test inspection. Add no-context and conventional-feature baselines, a compact published comparator, and targeted feature/context ablations. Use the same evaluation targets for fair comparisons.
4. Measure uncertainty across participants and development folds. Resolve ISRUC compatibility discrepancies, freeze external evaluation rules, and run a genuine external test.
5. Measure complete pipeline cost. Write the paper around supported trade-offs and limitations; release the exact artifacts needed to reproduce its tables.

The verified corrected model is worth building on. The most valuable next step is to make its evidence reproducible and scientifically discriminating.
