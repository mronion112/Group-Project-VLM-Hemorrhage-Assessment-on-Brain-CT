"""CT preprocessing for Block 3 (Phase 2): HU verification, slice ordering, multi-windowing (E1), and orig-grid metrics.

Rules enforced:
1. Convert DICOM stored pixels to HU ONLY when RescaleSlope and RescaleIntercept are present in header.
2. Never apply an extra -1024 offset to pre-converted NIfTI HU volumes (BHSD, PhysioNet CT-ICH).
3. Preserve slice ordering strictly by physical axial coordinate z_mm and preserve spacing/orientation.
4. Provide 3 window configurations for controlled ablation E1:
   - brain_only:          Brain window (L=40, W=80 -> [0, 80] HU)
   - brain_subdural:      Brain (40, 80) + Subdural-sensitive (L=80, W=200 -> [-20, 180] HU)
   - brain_subdural_bone: Brain (40, 80) + Subdural (80, 200) + Bone (L=600, W=2800 -> [-800, 2000] HU)
5. Restore predicted masks from working resolution (256x256) back to original grid (orig_shape, orig_spacing)
   before computing Dice and HD95 (in mm).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

from prepare_data import dicom_bytes_to_hu_and_meta, restore_mask_to_orig_grid, verify_preconverted_hu_array

CT_WINDOWS: dict[str, tuple[float, float]] = {
    "brain": (40.0, 80.0),       # [0, 80] HU
    "subdural": (80.0, 200.0),   # [-20, 180] HU (SDH-sensitive window)
    "bone": (600.0, 2800.0),     # [-800, 2000] HU
}

WINDOW_MODES: dict[str, tuple[str, ...]] = {
    "brain_only": ("brain",),
    "brain_subdural": ("brain", "subdural"),
    "brain_subdural_bone": ("brain", "subdural", "bone"),
}


def apply_window(hu: np.ndarray, level: float, width: float) -> np.ndarray:
    """Apply a single CT window (level, width) and normalize linearly to [0.0, 1.0] float32."""
    if width <= 0:
        raise ValueError(f"Window width must be > 0, got {width}")
    if not np.all(np.isfinite(hu)):
        raise ValueError("Input HU array contains NaN or Inf values.")
    low = float(level) - float(width) / 2.0
    high = float(level) + float(width) / 2.0
    windowed = (hu.astype(np.float32) - low) / (high - low)
    return np.clip(windowed, 0.0, 1.0, out=windowed)


def apply_window_mode(
    hu_array: np.ndarray,
    mode: str = "brain_subdural_bone",
    expand_to_3ch: bool = True,
    verify_hu: bool = True,
) -> np.ndarray:
    """Convert 2D (H, W) or 3D (Z, H, W) HU array into multi-channel windowed float32 in [0, 1].

    Returns:
      - (C, H, W) if input is 2D (H, W)
      - (Z, C, H, W) if input is 3D (Z, H, W)
    """
    if mode not in WINDOW_MODES:
        raise ValueError(f"Unknown window mode '{mode}'. Valid modes: {list(WINDOW_MODES)}")
    if hu_array.ndim not in (2, 3):
        raise ValueError(f"Expected 2D (H,W) or 3D (Z,H,W) array, got shape={hu_array.shape}")
    if verify_hu:
        verify_preconverted_hu_array(hu_array, source_name=f"apply_window_mode({mode})")

    win_names = WINDOW_MODES[mode]
    channels = [apply_window(hu_array, *CT_WINDOWS[w]) for w in win_names]
    if expand_to_3ch:
        if len(channels) == 1:
            channels = [channels[0], channels[0], channels[0]]
        elif len(channels) == 2:
            channels = [channels[0], channels[1], channels[0]]

    channel_axis = 0 if hu_array.ndim == 2 else 1
    return np.stack(channels, axis=channel_axis).astype(np.float32)


def order_slices_by_z(
    slices: list[tuple[float, int, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Sort slices strictly by physical axial coordinate (z_mm, slice_index) and stack into (Z, H, W)."""
    if not slices:
        raise ValueError("Cannot order an empty slice list.")
    sorted_slices = sorted(slices, key=lambda item: (float(item[0]), int(item[1])))
    vol_zyx = np.stack([item[2] for item in sorted_slices], axis=0)
    z_mm = np.array([float(item[0]) for item in sorted_slices], dtype=np.float32)
    return vol_zyx, z_mm


def compute_binary_dice(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    """Compute binary Sorensen-Dice coefficient on boolean/0-1 arrays of identical shape."""
    if pred_bin.shape != gt_bin.shape:
        raise ValueError(f"Shape mismatch in Dice: pred={pred_bin.shape} vs gt={gt_bin.shape}")
    p = pred_bin > 0
    g = gt_bin > 0
    p_sum = int(np.count_nonzero(p))
    g_sum = int(np.count_nonzero(g))
    if p_sum == 0 and g_sum == 0:
        return 1.0
    inter = int(np.count_nonzero(p & g))
    return float(2.0 * inter / (p_sum + g_sum))


def compute_hd95_mm(
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
    spacing_xyz_mm: tuple[float, float, float] | list[float] | np.ndarray,
) -> float:
    """Compute 95th-percentile Hausdorff Distance (HD95) in mm on (Z, Y, X) binary masks."""
    if pred_bin.shape != gt_bin.shape:
        raise ValueError(f"Shape mismatch in HD95: pred={pred_bin.shape} vs gt={gt_bin.shape}")
    p = pred_bin > 0
    g = gt_bin > 0
    p_any = bool(np.any(p))
    g_any = bool(np.any(g))
    if not p_any and not g_any:
        return 0.0
    if p_any != g_any:
        return float("nan")

    dx, dy, dz = float(spacing_xyz_mm[0]), float(spacing_xyz_mm[1]), float(spacing_xyz_mm[2])
    sampling_zyx = (dz, dy, dx)

    p_border = p ^ binary_erosion(p, border_value=0)
    g_border = g ^ binary_erosion(g, border_value=0)
    if not np.any(p_border):
        p_border = p
    if not np.any(g_border):
        g_border = g

    # Crop to convex joint bounding box of (p_border U g_border): mathematically identical Euclidean distances
    # while avoiding allocating 70 MB float64 EDT arrays over empty background on 512x512 volumes.
    union_coords = np.argwhere(p_border | g_border)
    mins = np.maximum(union_coords.min(axis=0) - 1, 0)
    maxs = np.minimum(union_coords.max(axis=0) + 2, p.shape)
    sl = tuple(slice(int(mins[i]), int(maxs[i])) for i in range(3))
    p_sub = p_border[sl]
    g_sub = g_border[sl]

    dt_g = distance_transform_edt(~g_sub, sampling=sampling_zyx)
    dt_p = distance_transform_edt(~p_sub, sampling=sampling_zyx)
    dists = np.concatenate([dt_g[p_sub], dt_p[g_sub]])
    return float(np.percentile(dists, 95))


def evaluate_mask_on_orig_grid(
    pred_mask_sub_zyx: np.ndarray,
    gt_mask_orig_zyx: np.ndarray,
    orig_spacing_xyz_mm: tuple[float, float, float] | list[float] | np.ndarray,
) -> dict[str, float | list[int]]:
    """Restore predicted mask (e.g. 256x256) to original grid (e.g. 512x512) and compute Dice & HD95 (mm)."""
    orig_shape_zyx = gt_mask_orig_zyx.shape
    pred_orig_zyx = restore_mask_to_orig_grid(pred_mask_sub_zyx, orig_shape_zyx)
    dice = compute_binary_dice(pred_orig_zyx > 0, gt_mask_orig_zyx > 0)
    hd95 = compute_hd95_mm(pred_orig_zyx > 0, gt_mask_orig_zyx > 0, orig_spacing_xyz_mm)
    return {
        "dice_on_orig_grid": round(dice, 6),
        "hd95_mm_on_orig_grid": round(hd95, 4) if np.isfinite(hd95) else float("nan"),
        "evaluated_shape_zyx": list(orig_shape_zyx),
    }


__all__ = [
    "CT_WINDOWS",
    "WINDOW_MODES",
    "apply_window",
    "apply_window_mode",
    "compute_binary_dice",
    "compute_hd95_mm",
    "dicom_bytes_to_hu_and_meta",
    "evaluate_mask_on_orig_grid",
    "order_slices_by_z",
    "restore_mask_to_orig_grid",
]
