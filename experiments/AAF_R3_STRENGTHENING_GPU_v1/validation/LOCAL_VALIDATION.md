# Local validation — engineering, not new scientific results

- **61 tests passed; 4 actual-VMAS-dependent tests skipped.**
- Inherited R3 code is byte-for-byte unchanged and verified from the actual returned archive.
- Original four controller arms match the inherited numerical rules on 500-step multi-world test sequences, including missing evidence.
- New tests cover authority/cooldown/rolling starts, next-step versus immediate control, missing-evidence handling, split delay channels, predictor constraints, immutable source, checksums and incompatible resumes.
- Nine parameterized wiring tests use an explicitly named test double. They are NOT validation of the actual VMAS engine or physical safety.
- A real corrected-PPO/game smoke run completed 32 fixed-proposal replay comparisons, generated its audit, and was reused without rerunning on an identical second invocation.
- Scripted sparse/diffuse/rotating streams test mathematical selection opportunity; they are not new MARL publication evidence.
- Bash syntax and Python compilation checks passed.

## Not locally exercised

This authoring environment has Torch 2.10.0+cpu, no CUDA device and no installed VMAS. Dependency installation was attempted but network resolution was unavailable. The pinned GPU environment and real-engine integration therefore have NOT been run here. The IBM launcher requires actual CUDA arithmetic, actual VMAS 1.5.2 free-motion verification, native-engine tests and tiny training/evaluation before longer development work. It stops rather than substituting another simulator or claiming the missing tests passed.

Local dependency versions differ from the supplied isolated GPU installation; exact local versions are in LOCAL_VALIDATION.json. Numerical repeatability is scoped to a fixed software/device stack, not CPU-versus-GPU identity. Deterministic Torch kernels are requested; unsupported operations fail rather than silently relaxing that request.

No final experimental superiority, speedup, certified safety, or readiness for submission is asserted by these checks. The run plan has 164 development navigation evaluations nested in two new policy seeds, and 192 separately labeled game replay comparisons.

Return-export integrity was also exercised: the CPU game smoke return ZIP passed CRC and all 15 result-entry hashes, its reconstructed source passed the release checksum check, its CLI plan ran, and policy binaries were present.
