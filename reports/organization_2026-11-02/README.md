# Workspace organization - 2 November 2026

The paper scope now prioritizes Sleep-EDF Expanded, SSL, cross-attention and a
planned BiLSTM head. Cross-dataset work is preserved for later. This change
organizes paths and documentation; it does not implement the new architecture.

`relocations.json` maps original paths to organized locations. Root-level audits
moved to `scripts/audits/cross_dataset/`, configurations are grouped by source,
compute and deferred external work, and historical notebooks are grouped by
baseline and plotting use. Documentation is under `docs/`; pinned environment
specifications are under `environments/`. Notebook root discovery and current
commands were adjusted. Notebook outputs/execution counts were retained.

`training_stop.json` records the explicit user-requested stop of the prior CPU
campaign. Three complete SSL epochs and every file in the latest checkpoint
seal were preserved and verified. No supervised job started. The scheduler
released its lock normally after worker interruption. The old status/ledger
records a failed interrupted attempt; `user_stop.json` beside that campaign
records the intentional stop. No old attempt was deleted or resealed as complete.

Raw/prepared data, installed environments, model artifacts, participant splits,
old run directories and older report contents remain at their original paths.
The portable capsule allowlist and tests are updated for the organized paths
while accepting legacy root-level archive members. Source imports remain stable;
only packaging path metadata changes under `src/`. A new study is required after
that source change. Do not restart the previous GRU matrix automatically.

Validation is recorded in `validation.json` and `capsule_tests.txt`:

- Six capsule tests passed, including nested configuration/dependency paths and legacy root-document archive restoration.
- All nine notebooks retained their outputs, execution counts and cell metadata; 453 code cells compiled without execution. Root discovery was checked from both the checkout root and the new notebook directories.
- All 35 checked local Markdown links resolved. Three moved audit CLIs and the capsule CLI passed `--help`; dependency consistency passed.
- All ten preserved baseline artifact hashes matched. The 67 previously tracked report files were unchanged after Git line-ending normalization.
- All configuration and moved environment-specification bytes were preserved. Installed package versions match the stopped campaign.
- All seven files in the latest three-epoch SSL checkpoint were verified again after organization.

No research training, notebook cells or real ISRUC prediction was executed for
this validation. Model/training/preprocessing/inference modules retain their
prior code; the only changed source module is the capsule packaging allowlist.
