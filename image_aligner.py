#!/usr/bin/env python3
"""
Image Alignment Tool

Aligns images from align_from/ to match reference images in align_to/.
Handles:
1. Pixel size matching
2. Aspect ratio conforming
3. Geometric alignment via homography (robust to ~25% foreground changes)
4. Color/tonality matching using optimal transport (Monge-Kantorovich)
"""

import argparse
import os
import re
import warnings
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from scipy.linalg import sqrtm, inv
from skimage import exposure


def find_matching_pairs(align_to_dir: Path, align_from_dir: Path) -> list[tuple[Path, Path]]:
    """Find image pairs based on filename prefix matching."""
    pairs = []

    # Get reference images
    ref_images = {}
    for f in align_to_dir.iterdir():
        if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']:
            # Extract base name (e.g., "hasta_100" from "hasta_100.png")
            base = f.stem
            ref_images[base] = f

    # Match source images to references
    for f in align_from_dir.iterdir():
        if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']:
            # Find which reference this matches (e.g., "hasta_100_clown" -> "hasta_100")
            for ref_base, ref_path in ref_images.items():
                if f.stem.startswith(ref_base + '_') or f.stem == ref_base:
                    pairs.append((ref_path, f))
                    break

    return pairs


def load_image(path: Path) -> np.ndarray:
    """Load image as BGR numpy array."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load image: {path}")

    # Convert RGBA to RGB if needed
    if len(img.shape) == 3 and img.shape[2] == 4:
        # Use white background for alpha compositing
        alpha = img[:, :, 3:4] / 255.0
        rgb = img[:, :, :3]
        white_bg = np.ones_like(rgb) * 255
        img = (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)

    return img


def resize_to_target(img: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Resize image to target size (width, height) using high-quality interpolation."""
    return cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)


def extract_features(img: np.ndarray) -> tuple:
    """Extract SIFT features from grayscale image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use SIFT for robust feature detection
    sift = cv2.SIFT_create(nfeatures=10000, contrastThreshold=0.02, edgeThreshold=15)
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    return keypoints, descriptors


def match_features(desc1: np.ndarray, desc2: np.ndarray, ratio_thresh: float = 0.85) -> list:
    """Match features using FLANN with Lowe's ratio test.

    Using a more lenient ratio threshold (0.85 vs typical 0.75) to allow
    more matches through, relying on RANSAC to filter outliers later.
    This helps when there are significant foreground changes.
    """
    if desc1 is None or desc2 is None:
        return []

    # FLANN parameters for SIFT
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=150)  # More checks for better matching

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        matches = flann.knnMatch(desc1, desc2, k=2)
    except cv2.error:
        return []

    # Apply Lowe's ratio test (lenient to allow more candidates)
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

    return good_matches


def compute_homography(kp1: list, kp2: list, matches: list,
                       ransac_reproj_thresh: float = 8.0,
                       confidence: float = 0.9999) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute homography using RANSAC for robustness to outliers.
    Returns homography matrix and inlier mask.

    Uses higher reprojection threshold (8px vs typical 3-5px) and very high
    confidence to handle up to ~50% outliers from foreground changes.
    """
    if len(matches) < 4:
        return None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Use USAC_MAGSAC for maximum robustness to outliers
    # Higher confidence and more iterations to handle large outlier ratios
    H, mask = cv2.findHomography(
        src_pts, dst_pts,
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=ransac_reproj_thresh,
        maxIters=10000,
        confidence=confidence
    )

    return H, mask


def compute_affine(kp1: list, kp2: list, matches: list,
                   ransac_reproj_thresh: float = 8.0,
                   confidence: float = 0.9999) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute affine transformation (subset of homography) for more stable alignment.
    Uses robust estimation to handle large foreground changes.
    """
    if len(matches) < 3:
        return None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Estimate affine transformation with robust parameters
    M, mask = cv2.estimateAffine2D(
        src_pts, dst_pts,
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=ransac_reproj_thresh,
        maxIters=10000,
        confidence=confidence
    )

    if M is not None:
        # Convert to 3x3 homography format
        H = np.vstack([M, [0, 0, 1]])
        return H, mask

    return None, None


def apply_transform(img: np.ndarray, H: np.ndarray,
                    output_size: tuple[int, int]) -> np.ndarray:
    """Apply homography/affine transform to image."""
    return cv2.warpPerspective(img, H, output_size,
                               flags=cv2.INTER_LANCZOS4,
                               borderMode=cv2.BORDER_REPLICATE)


def create_inlier_mask(keypoints: list, matches: list, inlier_mask: np.ndarray,
                       image_shape: tuple, radius: int = 50,
                       use_target_keypoints: bool = True) -> np.ndarray:
    """
    Create a boolean mask from inlier keypoints.

    The mask marks regions around inlier keypoints as "background" (True),
    which are used for computing color statistics.

    Args:
        keypoints: List of keypoints (source or target)
        matches: List of matches
        inlier_mask: Boolean mask from RANSAC indicating which matches are inliers
        image_shape: (height, width) of the output mask
        radius: Radius around each keypoint to mark as inlier region
        use_target_keypoints: If True, use trainIdx (target); else queryIdx (source)

    Returns:
        Boolean mask where True indicates inlier/background regions
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=bool)

    # Get inlier keypoint positions
    for i, m in enumerate(matches):
        if inlier_mask[i]:
            if use_target_keypoints:
                pt = keypoints[m.trainIdx].pt
            else:
                pt = keypoints[m.queryIdx].pt
            x, y = int(pt[0]), int(pt[1])

            # Mark circular region around keypoint
            y_min = max(0, y - radius)
            y_max = min(h, y + radius + 1)
            x_min = max(0, x - radius)
            x_max = min(w, x + radius + 1)

            # Create circular mask for this keypoint
            yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
            circle = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
            mask[y_min:y_max, x_min:x_max] |= circle

    # Dilate the mask to expand coverage
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2).astype(bool)

    return mask


def optimal_transport_color_transfer(source: np.ndarray, target: np.ndarray,
                                     n_samples: int = 50000,
                                     mask: np.ndarray = None) -> np.ndarray:
    """
    Apply color transfer using optimal transport (Monge-Kantorovich).

    This uses a linear approximation via covariance matching, which is the
    closed-form solution for Gaussian distributions (Monge-Kantorovich with
    quadratic cost).

    Args:
        source: Source image (BGR)
        target: Target image (BGR)
        n_samples: Number of samples for statistics computation
        mask: Optional boolean mask - if provided, statistics are computed
              only from masked (True) regions, but transform is applied globally
    """
    # Work in float64 for precision
    source = source.astype(np.float64)
    target = target.astype(np.float64)

    # Flatten to pixel arrays
    h, w, c = source.shape
    src_pixels = source.reshape(-1, c)
    tgt_pixels = target.reshape(-1, c)

    # If mask provided, use only masked pixels for computing statistics
    if mask is not None:
        mask_flat = mask.reshape(-1)
        src_masked = src_pixels[mask_flat]
        tgt_masked = tgt_pixels[mask_flat]
    else:
        src_masked = src_pixels
        tgt_masked = tgt_pixels

    # Subsample for efficiency if needed
    if len(src_masked) > n_samples:
        idx_src = np.random.choice(len(src_masked), n_samples, replace=False)
        idx_tgt = np.random.choice(len(tgt_masked), min(n_samples, len(tgt_masked)), replace=False)
        src_sample = src_masked[idx_src]
        tgt_sample = tgt_masked[idx_tgt]
    else:
        src_sample = src_masked
        tgt_sample = tgt_masked

    # Compute means
    src_mean = np.mean(src_sample, axis=0)
    tgt_mean = np.mean(tgt_sample, axis=0)

    # Center the data
    src_centered = src_sample - src_mean
    tgt_centered = tgt_sample - tgt_mean

    # Compute covariances
    src_cov = np.cov(src_centered.T)
    tgt_cov = np.cov(tgt_centered.T)

    # Regularize covariances for numerical stability
    eps = 1e-5
    src_cov += eps * np.eye(c)
    tgt_cov += eps * np.eye(c)

    use_fallback = False

    # Suppress warnings during matrix operations
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # T = Σ_s^(-1/2) @ (Σ_s^(1/2) @ Σ_t @ Σ_s^(1/2))^(1/2) @ Σ_s^(-1/2)
            src_sqrt = sqrtm(src_cov).real
            src_sqrt_inv = inv(src_sqrt)

            # Compute middle term
            middle = src_sqrt @ tgt_cov @ src_sqrt
            middle_sqrt = sqrtm(middle).real

            # Final transport matrix
            T = src_sqrt_inv @ middle_sqrt @ src_sqrt_inv

            # Check for numerical issues
            if not np.isfinite(T).all():
                use_fallback = True
            else:
                # Apply transformation to all pixels
                result_pixels = (src_pixels - src_mean) @ T.T + tgt_mean

                # Check result validity
                if not np.isfinite(result_pixels).all():
                    use_fallback = True

        except Exception:
            use_fallback = True

    if use_fallback:
        # Fallback: simple mean/std matching per channel (Reinhard method)
        result_pixels = src_pixels.copy()
        for i in range(c):
            src_std = np.std(src_sample[:, i]) + 1e-6
            tgt_std = np.std(tgt_sample[:, i]) + 1e-6
            result_pixels[:, i] = (src_pixels[:, i] - src_mean[i]) * (tgt_std / src_std) + tgt_mean[i]

    # Reshape and clip
    result = result_pixels.reshape(h, w, c)
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def histogram_matching_rgb(source: np.ndarray, target: np.ndarray,
                           mask: np.ndarray = None) -> np.ndarray:
    """
    Simple per-channel histogram matching in RGB space.

    Args:
        source: Source image (BGR)
        target: Target image (BGR)
        mask: Optional boolean mask - statistics from masked regions only
    """
    if mask is None:
        matched = np.zeros_like(source)
        for i in range(3):
            matched[:, :, i] = exposure.match_histograms(
                source[:, :, i], target[:, :, i]
            )
        return matched

    # With mask: compute lookup table from masked regions, apply globally
    matched = np.zeros_like(source)
    for i in range(3):
        src_masked = source[:, :, i][mask]
        tgt_masked = target[:, :, i][mask]
        # Build lookup table from masked pixels
        lookup = _build_histogram_lookup(src_masked, tgt_masked)
        matched[:, :, i] = lookup[source[:, :, i]]
    return matched


def _build_histogram_lookup(src_channel: np.ndarray, tgt_channel: np.ndarray,
                            n_bins: int = 256) -> np.ndarray:
    """Build a histogram matching lookup table from sample pixels."""
    src_hist, _ = np.histogram(src_channel.flatten(), bins=n_bins, range=(0, 256))
    tgt_hist, _ = np.histogram(tgt_channel.flatten(), bins=n_bins, range=(0, 256))

    src_cdf = np.cumsum(src_hist).astype(np.float64)
    src_cdf = src_cdf / src_cdf[-1]

    tgt_cdf = np.cumsum(tgt_hist).astype(np.float64)
    tgt_cdf = tgt_cdf / tgt_cdf[-1]

    # Build lookup
    lookup = np.zeros(n_bins, dtype=np.uint8)
    tgt_idx = 0
    for src_idx in range(n_bins):
        while tgt_idx < n_bins - 1 and tgt_cdf[tgt_idx] < src_cdf[src_idx]:
            tgt_idx += 1
        lookup[src_idx] = tgt_idx

    return lookup


def histogram_matching_lab(source: np.ndarray, target: np.ndarray,
                           mask: np.ndarray = None) -> np.ndarray:
    """
    Histogram matching in LAB color space for better perceptual results.
    LAB separates luminance from chrominance, giving more natural color transfer.

    Args:
        source: Source image (BGR)
        target: Target image (BGR)
        mask: Optional boolean mask - statistics from masked regions only
    """
    # Convert BGR to LAB
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    if mask is None:
        # Match histograms in LAB space
        matched_lab = np.zeros_like(source_lab)
        for i in range(3):
            matched_lab[:, :, i] = exposure.match_histograms(
                source_lab[:, :, i], target_lab[:, :, i]
            )
    else:
        # With mask: compute from masked regions, apply globally
        matched_lab = np.zeros_like(source_lab)
        for i in range(3):
            src_masked = source_lab[:, :, i][mask]
            tgt_masked = target_lab[:, :, i][mask]
            lookup = _build_histogram_lookup_float(src_masked, tgt_masked)
            # Apply lookup with interpolation
            src_channel = source_lab[:, :, i]
            src_floor = np.floor(src_channel).astype(np.int32)
            src_ceil = np.minimum(src_floor + 1, 255)
            src_frac = src_channel - src_floor
            src_floor = np.clip(src_floor, 0, 255)
            matched_lab[:, :, i] = (1 - src_frac) * lookup[src_floor] + src_frac * lookup[src_ceil]

    # Convert back to BGR
    matched_lab = np.clip(matched_lab, 0, 255).astype(np.uint8)
    matched = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)

    return matched


def _build_histogram_lookup_float(src_channel: np.ndarray, tgt_channel: np.ndarray,
                                   n_bins: int = 256) -> np.ndarray:
    """Build a histogram matching lookup table for float data."""
    src_hist, _ = np.histogram(src_channel.flatten(), bins=n_bins, range=(0, 256))
    tgt_hist, _ = np.histogram(tgt_channel.flatten(), bins=n_bins, range=(0, 256))

    src_cdf = np.cumsum(src_hist).astype(np.float64)
    src_cdf = src_cdf / (src_cdf[-1] + 1e-10)

    tgt_cdf = np.cumsum(tgt_hist).astype(np.float64)
    tgt_cdf = tgt_cdf / (tgt_cdf[-1] + 1e-10)

    lookup = np.zeros(n_bins, dtype=np.float32)
    tgt_idx = 0
    for src_idx in range(n_bins):
        while tgt_idx < n_bins - 1 and tgt_cdf[tgt_idx] < src_cdf[src_idx]:
            tgt_idx += 1
        lookup[src_idx] = tgt_idx

    return lookup


def histogram_matching_multichannel(source: np.ndarray, target: np.ndarray,
                                    mask: np.ndarray = None) -> np.ndarray:
    """
    Full multichannel histogram matching using skimage.
    Matches the joint distribution, not just marginals.

    Args:
        source: Source image (BGR)
        target: Target image (BGR)
        mask: Optional boolean mask - if provided, uses RGB matching with mask
    """
    if mask is None:
        # skimage's match_histograms can handle multichannel
        matched = exposure.match_histograms(source, target, channel_axis=2)
        return matched.astype(np.uint8)
    else:
        # Fall back to per-channel with mask
        return histogram_matching_rgb(source, target, mask)


def piecewise_linear_histogram_transfer(source: np.ndarray, target: np.ndarray,
                                         n_bins: int = 256,
                                         mask: np.ndarray = None) -> np.ndarray:
    """
    Piecewise linear histogram transfer using CDF matching.
    More precise than simple histogram matching, preserves more detail.

    Args:
        source: Source image (BGR)
        target: Target image (BGR)
        n_bins: Number of histogram bins
        mask: Optional boolean mask - statistics from masked regions only
    """
    result = np.zeros_like(source, dtype=np.float32)

    for c in range(3):
        # Use masked pixels for statistics if mask provided
        if mask is not None:
            src_channel = source[:, :, c][mask].astype(np.float32)
            tgt_channel = target[:, :, c][mask].astype(np.float32)
        else:
            src_channel = source[:, :, c].flatten().astype(np.float32)
            tgt_channel = target[:, :, c].flatten().astype(np.float32)

        # Compute histograms and CDFs
        src_hist, src_bins = np.histogram(src_channel, bins=n_bins, range=(0, 256))
        tgt_hist, tgt_bins = np.histogram(tgt_channel, bins=n_bins, range=(0, 256))

        # Compute CDFs
        src_cdf = np.cumsum(src_hist).astype(np.float64)
        src_cdf = src_cdf / (src_cdf[-1] + 1e-10)  # Normalize

        tgt_cdf = np.cumsum(tgt_hist).astype(np.float64)
        tgt_cdf = tgt_cdf / (tgt_cdf[-1] + 1e-10)  # Normalize

        # Build lookup table using CDF matching
        lookup = np.zeros(n_bins, dtype=np.float32)
        tgt_idx = 0
        for src_idx in range(n_bins):
            while tgt_idx < n_bins - 1 and tgt_cdf[tgt_idx] < src_cdf[src_idx]:
                tgt_idx += 1
            lookup[src_idx] = tgt_idx

        # Apply lookup with interpolation for smoother results (globally)
        src_img = source[:, :, c].astype(np.float32)
        src_floor = np.floor(src_img).astype(np.int32)
        src_ceil = np.minimum(src_floor + 1, n_bins - 1)
        src_frac = src_img - src_floor

        src_floor = np.clip(src_floor, 0, n_bins - 1)

        result[:, :, c] = (1 - src_frac) * lookup[src_floor] + src_frac * lookup[src_ceil]

    return np.clip(result, 0, 255).astype(np.uint8)


def full_histogram_matching(source: np.ndarray, target: np.ndarray,
                            mask: np.ndarray = None) -> np.ndarray:
    """
    Comprehensive histogram matching combining multiple techniques:
    1. LAB space matching for perceptually uniform results
    2. Fine-tuning with piecewise linear transfer
    3. Blend for natural appearance

    Args:
        source: Source image (BGR)
        target: Target image (BGR)
        mask: Optional boolean mask - statistics from masked regions only,
              but transform is applied globally
    """
    # Method 1: LAB histogram matching (good for overall color)
    lab_matched = histogram_matching_lab(source, target, mask)

    # Method 2: Piecewise linear CDF matching (good for detail preservation)
    cdf_matched = piecewise_linear_histogram_transfer(source, target, mask=mask)

    # Method 3: Direct multichannel matching
    multi_matched = histogram_matching_multichannel(source, target, mask)

    # Blend the results (weighted average favoring LAB for color accuracy)
    result = (0.5 * lab_matched.astype(np.float32) +
              0.3 * cdf_matched.astype(np.float32) +
              0.2 * multi_matched.astype(np.float32))

    return np.clip(result, 0, 255).astype(np.uint8)


def compute_diff_image(img1: np.ndarray, img2: np.ndarray, amplify: float = 3.0) -> np.ndarray:
    """
    Compute a colored difference image between two images.
    Red channel shows where img1 > img2, blue shows where img2 > img1.
    """
    # Convert to float and grayscale for diff computation
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)

    diff = gray1 - gray2

    # Create RGB diff visualization
    diff_img = np.zeros((*gray1.shape, 3), dtype=np.uint8)

    # Red where img1 is brighter (positive diff)
    pos_diff = np.clip(diff * amplify, 0, 255).astype(np.uint8)
    # Blue where img2 is brighter (negative diff)
    neg_diff = np.clip(-diff * amplify, 0, 255).astype(np.uint8)

    diff_img[:, :, 2] = pos_diff  # Red channel (BGR format)
    diff_img[:, :, 0] = neg_diff  # Blue channel

    # Add some green where they're similar for better visibility
    similar = 255 - np.clip(np.abs(diff) * amplify, 0, 255).astype(np.uint8)
    diff_img[:, :, 1] = similar // 4  # Subtle green

    return diff_img


def create_visualization_panel(naive_resized: np.ndarray,
                                aligned: np.ndarray,
                                target: np.ndarray,
                                title: str = "") -> np.ndarray:
    """
    Create a visualization panel with 5 images:
    Row 1: [Naive Resized] [Aligned] [Target/Reference]
    Row 2: [Diff: Naive vs Target] [Diff: Aligned vs Target]

    Args:
        naive_resized: Source image naively resized to target dimensions
        aligned: Geometrically aligned (and color matched) image
        target: Reference/target image

    Returns:
        Visualization panel as numpy array
    """
    h, w = target.shape[:2]

    # Compute difference images
    diff_naive = compute_diff_image(naive_resized, target)
    diff_aligned = compute_diff_image(aligned, target)

    # Create labels
    label_height = 40
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    font_color = (255, 255, 255)
    bg_color = (40, 40, 40)

    def add_label(img: np.ndarray, text: str) -> np.ndarray:
        """Add a label bar above an image."""
        label_bar = np.full((label_height, img.shape[1], 3), bg_color, dtype=np.uint8)
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = (img.shape[1] - text_size[0]) // 2
        text_y = (label_height + text_size[1]) // 2
        cv2.putText(label_bar, text, (text_x, text_y), font, font_scale, font_color, font_thickness)
        return np.vstack([label_bar, img])

    # Add labels to all images
    naive_labeled = add_label(naive_resized, "Naive Resize")
    aligned_labeled = add_label(aligned, "After Alignment")
    target_labeled = add_label(target, "Target Reference")
    diff_naive_labeled = add_label(diff_naive, "Diff: Naive vs Target")
    diff_aligned_labeled = add_label(diff_aligned, "Diff: Aligned vs Target")

    # Create empty placeholder for bottom row (to balance 3 vs 2)
    empty = np.full((h + label_height, w, 3), bg_color, dtype=np.uint8)

    # Build rows
    row1 = np.hstack([naive_labeled, aligned_labeled, target_labeled])
    row2 = np.hstack([diff_naive_labeled, diff_aligned_labeled, empty])

    # Add title bar if provided
    if title:
        title_height = 50
        title_bar = np.full((title_height, row1.shape[1], 3), (60, 60, 60), dtype=np.uint8)
        title_size = cv2.getTextSize(title, font, 0.9, 2)[0]
        title_x = (row1.shape[1] - title_size[0]) // 2
        title_y = (title_height + title_size[1]) // 2
        cv2.putText(title_bar, title, (title_x, title_y), font, 0.9, (255, 255, 255), 2)
        panel = np.vstack([title_bar, row1, row2])
    else:
        panel = np.vstack([row1, row2])

    return panel


def align_image(source_img: np.ndarray, target_img: np.ndarray,
                use_affine: bool = False,
                color_method: str = 'optimal_transport') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align source image to target image.

    Args:
        source_img: Image to be aligned (BGR)
        target_img: Reference image (BGR)
        use_affine: If True, use affine transform; else use full homography
        color_method: 'optimal_transport' or 'histogram'

    Returns:
        Tuple of (final_result, naive_resized, aligned_before_color)
    """
    target_h, target_w = target_img.shape[:2]
    target_size = (target_w, target_h)

    # Step 1: Initial resize to target dimensions
    source_resized = resize_to_target(source_img, target_size)
    naive_resized = source_resized.copy()  # Keep for visualization

    # Step 2: Extract features
    print("    Extracting features...")
    kp_src, desc_src = extract_features(source_resized)
    kp_tgt, desc_tgt = extract_features(target_img)

    print(f"    Found {len(kp_src)} source and {len(kp_tgt)} target keypoints")

    # Step 3: Match features
    print("    Matching features...")
    matches = match_features(desc_src, desc_tgt)
    print(f"    Found {len(matches)} good matches")

    # Initialize variables for inlier mask
    inlier_mask = None
    inlier_matches = None
    inlier_kp_tgt = None

    # Step 4: Compute geometric transform
    if len(matches) >= 4:
        print("    Computing geometric transform...")
        if use_affine:
            H, mask = compute_affine(kp_src, kp_tgt, matches)
        else:
            H, mask = compute_homography(kp_src, kp_tgt, matches)

        if H is not None:
            inliers = np.sum(mask) if mask is not None else 0
            print(f"    Inliers: {inliers}/{len(matches)}")

            # Store inlier information for color masking
            if mask is not None:
                inlier_mask = mask.ravel()
                inlier_matches = matches
                inlier_kp_tgt = kp_tgt

            # Apply geometric transformation
            aligned = apply_transform(source_resized, H, target_size)
        else:
            print("    Warning: Could not compute transform, using resized image")
            aligned = source_resized
    else:
        print("    Warning: Not enough matches, using resized image")
        aligned = source_resized

    # Step 5: Create background mask from inliers for color matching
    color_mask = None
    if inlier_mask is not None and inlier_matches is not None:
        print("    Creating background mask from inliers...")
        color_mask = create_inlier_mask(
            inlier_kp_tgt, inlier_matches, inlier_mask,
            target_img.shape, radius=50, use_target_keypoints=True
        )
        mask_coverage = np.sum(color_mask) / color_mask.size * 100
        print(f"    Background mask coverage: {mask_coverage:.1f}%")

    # Step 6: Color matching (using background mask)
    print(f"    Applying color matching ({color_method}) on background regions...")
    if color_method == 'optimal_transport':
        try:
            result = optimal_transport_color_transfer(aligned, target_img, mask=color_mask)
        except Exception as e:
            print(f"    Warning: Optimal transport failed ({e}), using full histogram")
            result = full_histogram_matching(aligned, target_img, mask=color_mask)
    elif color_method == 'full_histogram':
        result = full_histogram_matching(aligned, target_img, mask=color_mask)
    elif color_method == 'lab':
        result = histogram_matching_lab(aligned, target_img, mask=color_mask)
    elif color_method == 'cdf':
        result = piecewise_linear_histogram_transfer(aligned, target_img, mask=color_mask)
    else:  # 'histogram' or fallback
        result = histogram_matching_rgb(aligned, target_img, mask=color_mask)

    return result, naive_resized, aligned


def process_pairs(align_to_dir: Path, align_from_dir: Path, output_dir: Path,
                  use_affine: bool = False, color_method: str = 'optimal_transport',
                  suffix: str = '_aligned', create_visualization: bool = True):
    """Process all image pairs."""
    pairs = find_matching_pairs(align_to_dir, align_from_dir)

    if not pairs:
        print("No matching image pairs found!")
        return

    print(f"Found {len(pairs)} image pairs to process\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create visualization subdirectory
    if create_visualization:
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

    for ref_path, src_path in pairs:
        print(f"Processing: {src_path.name} -> {ref_path.name}")

        # Load images
        target_img = load_image(ref_path)
        source_img = load_image(src_path)

        print(f"  Source: {source_img.shape[1]}x{source_img.shape[0]}")
        print(f"  Target: {target_img.shape[1]}x{target_img.shape[0]}")

        # Align
        aligned, naive_resized, aligned_geom = align_image(
            source_img, target_img,
            use_affine=use_affine,
            color_method=color_method
        )

        # Save aligned image with suffix
        out_name = f"{src_path.stem}{suffix}{src_path.suffix}"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), aligned)
        print(f"  Saved: {out_path}")

        # Create and save visualization panel
        if create_visualization:
            print("    Creating visualization panel...")
            title = f"Alignment: {src_path.name} -> {ref_path.name}"
            panel = create_visualization_panel(
                naive_resized=naive_resized,
                aligned=aligned,
                target=target_img,
                title=title
            )
            viz_name = f"{src_path.stem}_viz.png"
            viz_path = viz_dir / viz_name
            cv2.imwrite(str(viz_path), panel)
            print(f"  Visualization: {viz_path}")

        print()


def process_single_pair(source_path: Path, target_path: Path, output_path: Path,
                        use_affine: bool = False, color_method: str = 'full_histogram',
                        create_visualization: bool = True):
    """Process a single image pair."""
    print(f"Processing: {source_path.name} -> {target_path.name}")

    # Load images
    target_img = load_image(target_path)
    source_img = load_image(source_path)

    print(f"  Source: {source_img.shape[1]}x{source_img.shape[0]}")
    print(f"  Target: {target_img.shape[1]}x{target_img.shape[0]}")

    # Align
    aligned, naive_resized, aligned_geom = align_image(
        source_img, target_img,
        use_affine=use_affine,
        color_method=color_method
    )

    # Save aligned image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), aligned)
    print(f"  Saved: {output_path}")

    # Create and save visualization panel
    if create_visualization:
        print("    Creating visualization panel...")
        title = f"Alignment: {source_path.name} -> {target_path.name}"
        panel = create_visualization_panel(
            naive_resized=naive_resized,
            aligned=aligned,
            target=target_img,
            title=title
        )
        viz_path = output_path.parent / f"{output_path.stem}_viz.png"
        cv2.imwrite(str(viz_path), panel)
        print(f"  Visualization: {viz_path}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description='Align images with geometric and color matching',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single pair mode (default):
  %(prog)s source.jpg target.jpg                    # Align source to target
  %(prog)s source.jpg target.jpg -o result.png      # Specify output path
  %(prog)s photo.png reference.png --color lab      # Use LAB color matching

  # Directory mode (batch processing):
  %(prog)s --batch                                  # Process align_from/ -> align_to/
  %(prog)s --batch -t refs/ -f inputs/ -o outputs/  # Custom directories
        """
    )

    # Positional arguments for single-pair mode
    parser.add_argument('source', type=Path, nargs='?',
                        help='Source image to align')
    parser.add_argument('target', type=Path, nargs='?',
                        help='Target/reference image to align to')

    # Directory mode
    parser.add_argument('--batch', '-b', action='store_true',
                        help='Batch mode: process directories instead of single pair')
    parser.add_argument('--align-to', '-t', type=Path, default=Path('align_to'),
                        help='Directory with reference images (batch mode, default: align_to)')
    parser.add_argument('--align-from', '-f', type=Path, default=Path('align_from'),
                        help='Directory with images to align (batch mode, default: align_from)')

    # Output
    parser.add_argument('--output', '-o', type=Path,
                        help='Output path (file for single mode, directory for batch mode)')

    # Alignment options
    parser.add_argument('--affine', '-a', action='store_true',
                        help='Use affine transform instead of homography')
    parser.add_argument('--color', '-c',
                        choices=['full_histogram', 'optimal_transport', 'lab', 'cdf', 'histogram'],
                        default='full_histogram',
                        help='Color matching method (default: full_histogram)')
    parser.add_argument('--suffix', '-s', default='_aligned',
                        help='Suffix for output filenames in batch mode (default: _aligned)')
    parser.add_argument('--no-viz', action='store_true',
                        help='Disable visualization panel generation')

    args = parser.parse_args()

    # Determine mode
    if args.batch:
        # Batch/directory mode
        if not args.align_to.exists():
            print(f"Error: Reference directory not found: {args.align_to}")
            return 1

        if not args.align_from.exists():
            print(f"Error: Source directory not found: {args.align_from}")
            return 1

        output_dir = args.output or Path('aligned_output')

        process_pairs(
            args.align_to,
            args.align_from,
            output_dir,
            use_affine=args.affine,
            color_method=args.color,
            suffix=args.suffix,
            create_visualization=not args.no_viz
        )

    elif args.source and args.target:
        # Single pair mode
        if not args.source.exists():
            print(f"Error: Source image not found: {args.source}")
            return 1

        if not args.target.exists():
            print(f"Error: Target image not found: {args.target}")
            return 1

        # Determine output path
        if args.output:
            output_path = args.output
        else:
            # Default: source_aligned.ext in current directory
            output_path = Path(f"{args.source.stem}_aligned{args.source.suffix}")

        process_single_pair(
            args.source,
            args.target,
            output_path,
            use_affine=args.affine,
            color_method=args.color,
            create_visualization=not args.no_viz
        )

    else:
        # No arguments - show help
        parser.print_help()
        print("\nError: Provide source and target images, or use --batch for directory mode.")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
