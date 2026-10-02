# Provenance and primary technical sources

## User-provided research artifacts

The executed R3 source is copied from `AAF_R3_NEXT_STAGE_RESULTS.zip/source/`.
The complete R3 manuscript and audits motivate the new diagnostic questions; this
package does not rewrite their results. See BASELINE_PROVENANCE.json for hashes.
No original figure or manuscript file is overwritten.

## Official dependency/control sources checked when preparing this package

- PyTorch official previous versions, v2.10.0 CUDA 12.8 wheel command:
  https://pytorch.org/get-started/previous-versions/
- VMAS release pinned for consistency with the completed study:
  https://pypi.org/project/vmas/1.5.2/
- Official VMAS navigation scenario and semi-implicit world integration:
  https://github.com/proroklab/VectorizedMultiAgentSimulator/blob/main/vmas/scenarios/navigation.py
  https://github.com/proroklab/VectorizedMultiAgentSimulator/blob/main/vmas/simulator/core.py
  Current source inspection is supplementary: the actual installed 1.5.2 engine is
  checked at runtime, rather than assuming a current main branch equals that release.
- Wabersich and Zeilinger, A predictive safety filter for learning-based control of
  constrained nonlinear dynamical systems, arXiv:1812.05506:
  https://arxiv.org/abs/1812.05506
  Background only. Our finite-candidate diagnostic does NOT implement their full
  certified framework, terminal conditions, robust constraints or guarantee.

The preparation environment has CPU Torch but lacks VMAS and CUDA. See LOCAL_VALIDATION.md.
watsonx is not called and no model service credentials are used.
