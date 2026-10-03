# Code-release validation — 3 October 2026

This release validation concerns source packaging and executable interfaces, not a rerun of the research experiments.

| Check | Result |
|---|---|
| Confirmation wrapper tests | 13 passed |
| Development/base plus vendor tests | 61 passed, 4 VMAS-dependent tests skipped |
| Standalone corrected-study tests | 23 passed, 2 VMAS-dependent tests skipped |
| Portable source verification and plan | Passed; 1,640 navigation evaluations, 52,480 episode records, 1,920 game replays, 30 primary tests |
| Python syntax | All 83 newly published Python files parsed successfully |
| Portable Bash syntax | Passed |
| Mechanism smoke | One seed per family; 7 telemetry rows, 2 scheduler rows, 9 shield-profile/depth rows, 2 screening rows |
| Historical-audit CLI | Help/import check passed; no original raw CSV rerun claimed |

The vendor tests overlap the standalone corrected-study tests; do not add the pass counts as if they were unique independent tests. Analysis/export tests include explicitly synthetic fixtures. The mechanism smoke checks execution, not the publication sample sizes or effectiveness claims.

Local environment: Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0. CUDA and VMAS are unavailable locally. The portable installer deliberately targets Linux Python 3.11/3.12 and the frozen GPU pins; a fresh dependency installation and full GPU run were not performed here. Earlier package validation files retain their original dates and scope. The previously returned IBM experiments are not represented as having been re-executed during this code release.

## Preservation

All 50 development-package files are unchanged. The confirmation's numerical Python, tests, protocol and base match the executed package; 59 of its 61 files are byte-identical. Only the private-host transfer guide and its checksum ledger entry changed. The 19 imported corrected-study source/protocol files are unchanged; a checksum ledger was added. The mechanism and historical-audit scripts are unchanged from the final author archive. New portable launchers are separate from these snapshots.

All historical root code, configurations, requirements, license and scheduler scripts remain unchanged. The original README is retained as `docs/LEGACY_README.md`, and its diagram HTML is preserved verbatim in the new README. No manuscript, reviewer response, author document, original result dataset, trained model, or physical trace is added to the code branch.
