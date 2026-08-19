# JII submission-verified source record

**Manuscript:** InteCom-D-26-00100  
**Title:** Adaptive Accountability in Networked MAS: Tracing and Mitigating Emergent Norms at Scale

This directory records the final verification pass on the protected branch `revision/jii-major-revision-redline`. The repository `main` branch remains unchanged.

The verification was performed against the submitted manuscript, the original figure assets, the released experiment code, the corrected empirical audit, and the reviewer comments. It does not blindly restore every older graphic. Instead, it retains valid run-derived geometry and replaces only figures or interpretations that depended on pseudo-replication, an uncorrected alarm clock convention, an unavailable metric, or a deterministic model presented as a measurement.

The verified manuscript has eight figures, matching the eight distinct figure concepts in the submitted paper. In particular it restores:

- the original allocation-Gini heat-map organization and cell means;
- the professional resource-sharing empirical-CDF format, using corrected conditional delay and coverage accounting;
- the submitted 5,000-step temporal learning curves as a separately scoped descriptive diagnostic; and
- the original canonical resource-sharing scaling compromise and no-attack runtime trajectories.

The third scaling panel is retained only with its correct interpretation as a deterministic modeled byte counter, not measured network bandwidth. All quantitative graphics use one semantic palette.

The complete compiled delivery is supplied separately as `AAF_JII_submission_VERIFIED.zip`. The tracked Base64 archive in this directory reconstructs the exact editable text source, audit reports, stress-test summaries, figure-verification record, and original-to-final patches used for that delivery.

## Restore

```bash
cd paper/jii_submission_verified
./restore_source_bundle.sh
```

The script reconstructs the archive, verifies its SHA-256 checksum, and extracts it to `restored_source/`.
