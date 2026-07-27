# Journal of Information and Intelligence major revision

**Manuscript:** InteCom-D-26-00100  
**Title:** Adaptive Accountability in Networked MAS: Tracing and Mitigating Emergent Norms at Scale

This protected revision workspace preserves the repository's existing `main` branch and records the final second-pass source revision, the point-by-point reviewer response, the editor-only title-page source, the empirical audit, and the reviewer-driven adversarial stress tests.

## Restore the exact committed source set

```bash
cd paper/jii_major_revision_final
./restore_source_bundle.sh
```

The script concatenates the tracked Base64 fragments, reconstructs `aaf_jii_github_min_bundle.tar.xz`, verifies its SHA-256 checksum, and extracts it into `restored_source/`.

## Bundle contents

- clean revised manuscript source;
- revised point-by-point response source;
- editor-only title-page source;
- empirical audit script and reports;
- targeted robustness script and raw summaries;
- experiment-status statement;
- preservation manifest.

The separately delivered `AAF_JII_major_revision_FINAL.zip` contains the compiled clean manuscript, red-highlighted reviewer copy, full redline, response PDF, title-page PDF, original submitted source, revised figures, build support, full patch, PDF preflight, and package-wide checksum ledger.

## Experiment status

The essential reviewer-driven mechanism tests have been completed and integrated. The 87,480-row archive was re-audited rather than incorrectly treated as 87,480 independent runs: it contains 29,160 distinct configuration-seed records and 19,440 effective seeded main-grid records across 216 operational regimes. Full PPO-loop adaptive-evasion testing, a physical attestation stack, and an embodied hard-real-time benchmark remain useful future extensions but are explicitly outside the revised claims.

## Preservation

No existing repository file is deleted or replaced by this commit. The original source archive is retained in the final delivery package, and the clean, highlighted, and redline variants remain separate artifacts.
