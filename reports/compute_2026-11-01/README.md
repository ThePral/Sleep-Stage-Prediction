# Compute workflow evidence — 1 November 2026

All **103 tests passed**. The full-source bounded study completed all five jobs
on native Windows CPU, with the default model sizes and all 70/15 source
training/validation participants. This is a capped execution check, not a
full-budget scientific experiment.

`source_validation_results.csv` and `paired_source_validation.csv` retain both
outcomes. M3 macro-F1 is 0.4733 and M6 is 0.3615 on the same capped 256 targets,
with the same normalizer. These short runs do not establish an SSL benefit or
harm. Checkpoint selection used source validation; no new source-test or real
ISRUC prediction was performed. Histories are finite and all job seals verified.

`cpu_gated_b4_50steps.json` records a separate fresh-process source-training
profile: 50 measured steps after five warmups, approximately 110 examples/second
with cached loading/normalization, sampled host RSS 534.7 MiB. The selected source
recording is SC4001. This excludes initial data verification and original signal
preparation, and does not establish GPU memory fit or campaign duration. Saved
artifacts remained unchanged.

`capsule_creation.json` and `capsule_roundtrip.json` record a real 4.89 GB archive
and complete extraction/verification of all 573 files, covering 138 training and
29 validation recordings. Only source training/validation epoch payloads are
included. The split CSV retains all participant assignments. The archive is in
ignored `runs/` and was not uploaded; its code/config snapshot is preserved by
the archive inventory. Final milestone documentation/evidence was added afterward.

`windows_cpu_kernels.json` passes both default synthetic training workloads.
`windows_gpu_required.json` correctly fails because native TensorFlow exposes
only CPU. WSL GPU dependencies are still downloading at this snapshot. The live
continuation status is
`runs/compute_2026-10-31/gpu_setup_status_2026-11-01.json`; later GPU reports and
the installed Linux package lock remain unreviewed run artifacts until collected.
Do not interpret the hardware inventory or pinned requirements as a passed GPU
training check. No paid instance, upload, system package or driver change occurred.

`evidence.json` records source/package/job hashes, actual participant coverage,
capsule verification and unchanged hashes for the ten historical artifacts.
Full convergence, repeated seeds and frozen independent-dataset evaluation remain
outstanding.
