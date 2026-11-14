# Code Style Guidelines

- Python 3.10+ PEP 8 standards with strict type hinting (`from typing import Optional, Tuple, List`).
- Naming conventions:
  - Functions & Variables: `snake_case` (e.g., `compute_dice_score`, `case_id`).
  - Classes & Models: `PascalCase` (e.g., `FocalTverskyLoss`, `RSNAWinningClassifier`).
  - Constants & Configs: `UPPER_SNAKE_CASE` (e.g., `SEED`, `BHSD_TO_CANONICAL`).
- Keep modules deep and concise. Functions should have a single responsibility.
- Use explicit numpy/torch array dimensions in docstrings or comments (e.g., `# (B, C, H, W)`).

```python
def compute_dice(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    """Compute Dice coefficient for binary masks with shape (H, W) or (D, H, W)."""
    intersection = float(np.sum(pred * target))
    total = float(np.sum(pred) + np.sum(target))
    return (2.0 * intersection) / (total + eps) if total > 0 else 1.0
```
