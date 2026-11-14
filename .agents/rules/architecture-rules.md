# Architecture & Engineering Rules

## Do:
- **Preserve Physical Geometry:** CT scans must retain native resolution ($512 \times 512 \times Z$), physical voxel spacing ($dx, dy, dz$), affine transformation matrices, and RAS+ coordinates.
- **End-to-End Machine Learning:** Learn bone, calcification, and hemorrhage boundaries through neural network representations (e.g. Bone Window channel, spatial attention) rather than heuristic post-filters.
- **Contract Enforcement:** All model predictions exported as `findings.json` must strictly validate against `findings.schema.json` via `validate_output.py`.
- **Reproducibility:** Fix seeds across Python, NumPy, PyTorch, and CUDA (`SEED = 2026`).

## Don't:
- **No Silent Downsampling:** Never downsample $512 \times 512$ inputs to lower resolutions (e.g. 256) without explicit user consent. Downsampling destroys thin sulcal SAH bleeds.
- **No Hardcoded Heuristic Patches:** Never use ad-hoc post-processing rules (e.g., `(hu >= 15) & (hu <= 105)` or `size < 15px`). These heuristics fail on clinical edge cases (DAI microbleeds, anemia, clot retraction).
- **No Cross-Block Boundary Violations:** Block 3 outputs classifications and masks. It does NOT compute volume in mL (owned by Block 4) and does NOT generate textual medical reports (owned by Block 5).
- **No Split Leaks:** Never re-split or modify the frozen patient-level partition (`block3/data/splits/`). Zero patient overlap across train, validation, and test.
