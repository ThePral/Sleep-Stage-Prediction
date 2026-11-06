# Prepared Sleep-EDF access repair — 6 November 2026

The user's terminal repeatedly failed the frozen participant-scope check while
the assistant's native Windows checks passed, including a traced execution CLI
that was deliberately stopped immediately after validation and before any
scheduler lock, job or optimizer operation.

Inspection found a concrete account/permission difference. The assistant runs
as `POURYA\CodexSandboxOffline`; the checkout owner is `POURYA\ELECOMP`. Prepared
recording directories and their files had protected access rules granting the
owner/sandbox, SYSTEM and Administrators access, without an access grant for
ELECOMP. Root-level listing was accessible, but glob traversal into recording
stores could silently omit manifests for an unprivileged terminal account.
This explains why repeating the same command did not address the failure.

`scripts/repair_sleepedf_access.ps1` grants explicit read/traverse access to
ELECOMP on the 197 recording directories and read access on their 591 files,
plus read/traverse on the Sleep-EDF parent: 789 objects. It retains ownership
and existing access grants, and changes no file bytes or training settings.
The script checks the final explicit grants, preserved owners, file sizes and
content-write timestamps. No read access was broadened to other user accounts.

`access_before.json` preserves access descriptors before the first attempt.
PowerShell's Set-Acl encountered a security-privilege error after updating the
parent, so the repair uses native icacls to update only access grants. One
recording directory was tested directly before the full repair.
`native_repair/access_before.json` preserves the pre-native-pass descriptors;
`native_repair/repair.json` records its verification outcome. Data hashes and
frozen implementation/package versions are checked again in `status_after.json`.
The scope check verifies manifest hashes, not complete waveform payload hashes.

Training code, packages, datasets, study manifests, historical checkpoints and
the empty job ledger remain unchanged. No training was launched. The user must
run the read-only status command in their own terminal to confirm access with
their actual process token, then launch their existing staged training command.
