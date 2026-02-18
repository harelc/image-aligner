#!/usr/bin/env python3
"""
Image Aligner - FastAPI Web Interface with API
Dedicated with love and devotion to Alon Y., Daniel B., Denis Z., Tal S., Adi B.
and the rest of the Animation Taskforce 2026
"""

import io
import os
import base64
import subprocess
import tempfile
import warnings
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.linalg import sqrtm, inv
from skimage import exposure
import uvicorn


# ============== Image Alignment Core ==============

def extract_features(img: np.ndarray) -> tuple:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    n_pixels = img.shape[0] * img.shape[1]
    nfeatures = min(10000, max(2000, n_pixels // 200))
    sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=0.02, edgeThreshold=15)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_features(desc1: np.ndarray, desc2: np.ndarray, ratio_thresh: float = 0.85) -> list:
    if desc1 is None or desc2 is None:
        return []
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=150)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    try:
        matches = flann.knnMatch(desc1, desc2, k=2)
    except cv2.error:
        return []
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
    return good_matches


def compute_homography(kp1, kp2, matches, ransac_reproj_thresh=8.0, confidence=0.9999):
    if len(matches) < 4:
        return None, None
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(
        src_pts, dst_pts,
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=ransac_reproj_thresh,
        maxIters=10000,
        confidence=confidence
    )
    return H, mask


def homography_deviation(H, width, height):
    H_norm = H / H[2, 2]
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, H_norm)
    displacement = np.sqrt(np.sum((warped.reshape(-1, 2) - corners.reshape(-1, 2)) ** 2, axis=1))
    return float(np.mean(displacement))


def create_inlier_mask(keypoints, matches, inlier_mask, image_shape, radius=50):
    h, w = image_shape[:2]
    mask_img = np.zeros((h, w), dtype=np.uint8)
    for i, m in enumerate(matches):
        if inlier_mask[i]:
            pt = keypoints[m.trainIdx].pt
            cv2.circle(mask_img, (int(pt[0]), int(pt[1])), radius, 1, -1)
    mask = mask_img.astype(bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2).astype(bool)
    return mask


def _build_histogram_lookup(src_channel, tgt_channel, n_bins=256):
    src_hist, _ = np.histogram(src_channel.flatten(), bins=n_bins, range=(0, 256))
    tgt_hist, _ = np.histogram(tgt_channel.flatten(), bins=n_bins, range=(0, 256))
    src_cdf = np.cumsum(src_hist).astype(np.float64)
    src_cdf = src_cdf / (src_cdf[-1] + 1e-10)
    tgt_cdf = np.cumsum(tgt_hist).astype(np.float64)
    tgt_cdf = tgt_cdf / (tgt_cdf[-1] + 1e-10)
    lookup = np.searchsorted(tgt_cdf, src_cdf).astype(np.uint8)
    return lookup


def _build_histogram_lookup_float(src_channel, tgt_channel, n_bins=256):
    src_hist, _ = np.histogram(src_channel.flatten(), bins=n_bins, range=(0, 256))
    tgt_hist, _ = np.histogram(tgt_channel.flatten(), bins=n_bins, range=(0, 256))
    src_cdf = np.cumsum(src_hist).astype(np.float64)
    src_cdf = src_cdf / (src_cdf[-1] + 1e-10)
    tgt_cdf = np.cumsum(tgt_hist).astype(np.float64)
    tgt_cdf = tgt_cdf / (tgt_cdf[-1] + 1e-10)
    lookup = np.searchsorted(tgt_cdf, src_cdf).astype(np.float32)
    return lookup


def histogram_matching_lab(source, target, mask=None):
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    if mask is None:
        matched_lab = np.zeros_like(source_lab)
        for i in range(3):
            matched_lab[:, :, i] = exposure.match_histograms(source_lab[:, :, i], target_lab[:, :, i])
    else:
        matched_lab = np.zeros_like(source_lab)
        for i in range(3):
            src_masked = source_lab[:, :, i][mask]
            tgt_masked = target_lab[:, :, i][mask]
            lookup = _build_histogram_lookup_float(src_masked, tgt_masked)
            src_channel = source_lab[:, :, i]
            src_floor = np.floor(src_channel).astype(np.int32)
            src_ceil = np.minimum(src_floor + 1, 255)
            src_frac = src_channel - src_floor
            src_floor = np.clip(src_floor, 0, 255)
            matched_lab[:, :, i] = (1 - src_frac) * lookup[src_floor] + src_frac * lookup[src_ceil]
    matched_lab = np.clip(matched_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)


def histogram_matching_rgb(source, target, mask=None):
    if mask is None:
        matched = np.zeros_like(source)
        for i in range(3):
            matched[:, :, i] = exposure.match_histograms(source[:, :, i], target[:, :, i])
        return matched
    matched = np.zeros_like(source)
    for i in range(3):
        src_masked = source[:, :, i][mask]
        tgt_masked = target[:, :, i][mask]
        lookup = _build_histogram_lookup(src_masked, tgt_masked)
        matched[:, :, i] = lookup[source[:, :, i]]
    return matched


def piecewise_linear_histogram_transfer(source, target, n_bins=256, mask=None):
    result = np.zeros_like(source, dtype=np.float32)
    for c in range(3):
        if mask is not None:
            src_channel = source[:, :, c][mask].astype(np.float32)
            tgt_channel = target[:, :, c][mask].astype(np.float32)
        else:
            src_channel = source[:, :, c].flatten().astype(np.float32)
            tgt_channel = target[:, :, c].flatten().astype(np.float32)
        src_hist, _ = np.histogram(src_channel, bins=n_bins, range=(0, 256))
        tgt_hist, _ = np.histogram(tgt_channel, bins=n_bins, range=(0, 256))
        src_cdf = np.cumsum(src_hist).astype(np.float64)
        src_cdf = src_cdf / (src_cdf[-1] + 1e-10)
        tgt_cdf = np.cumsum(tgt_hist).astype(np.float64)
        tgt_cdf = tgt_cdf / (tgt_cdf[-1] + 1e-10)
        lookup = np.searchsorted(tgt_cdf, src_cdf).astype(np.float32)
        src_img = source[:, :, c].astype(np.float32)
        src_floor = np.floor(src_img).astype(np.int32)
        src_ceil = np.minimum(src_floor + 1, n_bins - 1)
        src_frac = src_img - src_floor
        src_floor = np.clip(src_floor, 0, n_bins - 1)
        result[:, :, c] = (1 - src_frac) * lookup[src_floor] + src_frac * lookup[src_ceil]
    return np.clip(result, 0, 255).astype(np.uint8)


def fast_color_transfer(source, target, mask=None):
    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    if mask is not None:
        src_stats = src_lab[mask]
        tgt_stats = tgt_lab[mask]
    else:
        src_stats = src_lab.reshape(-1, 3)
        tgt_stats = tgt_lab.reshape(-1, 3)
    for i in range(3):
        s_mean, s_std = src_stats[:, i].mean(), src_stats[:, i].std() + 1e-6
        t_mean, t_std = tgt_stats[:, i].mean(), tgt_stats[:, i].std() + 1e-6
        src_lab[:, :, i] = (src_lab[:, :, i] - s_mean) * (t_std / s_std) + t_mean
    return cv2.cvtColor(np.clip(src_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def full_histogram_matching(source, target, mask=None):
    lab_matched = histogram_matching_lab(source, target, mask)
    cdf_matched = piecewise_linear_histogram_transfer(source, target, mask=mask)
    multi_matched = histogram_matching_rgb(source, target, mask)
    result = (0.5 * lab_matched.astype(np.float32) +
              0.3 * cdf_matched.astype(np.float32) +
              0.2 * multi_matched.astype(np.float32))
    return np.clip(result, 0, 255).astype(np.uint8)


# ============== Post-Processing ==============

# Level configs: (blur_sigma_mult, blur_sigma_min, motion_min, crf_boost)
PP_LEVELS = {
    0: None,  # disabled
    1: {'sigma_mult': 0.8, 'sigma_min': 0.0, 'motion_min': 1, 'crf_boost': 0},
    2: {'sigma_mult': 1.1, 'sigma_min': 0.0, 'motion_min': 1, 'crf_boost': 2},
    3: {'sigma_mult': 1.5, 'sigma_min': 0.3, 'motion_min': 3, 'crf_boost': 5},
}


def detect_foreground_mask(aligned, target, threshold=25, min_area=500):
    diff = cv2.absdiff(aligned, target)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cv2.GaussianBlur(cleaned.astype(np.float32) / 255.0, (31, 31), 0)


def estimate_blur_level(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var > 500:
        return 0.0
    elif laplacian_var < 10:
        return 3.0
    return max(0.0, 2.5 - np.log10(laplacian_var) * 0.9)


def estimate_motion_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 4
    angles_deg = np.arange(0, 180, 5)
    angles_rad = np.deg2rad(angles_deg)
    rs = np.arange(5, radius)
    dx = np.cos(angles_rad)
    dy = np.sin(angles_rad)
    X_pos = (cx + np.outer(rs, dx)).astype(int)
    Y_pos = (cy + np.outer(rs, dy)).astype(int)
    X_neg = (cx - np.outer(rs, dx)).astype(int)
    Y_neg = (cy - np.outer(rs, dy)).astype(int)
    valid_pos = (X_pos >= 0) & (X_pos < w) & (Y_pos >= 0) & (Y_pos < h)
    valid_neg = (X_neg >= 0) & (X_neg < w) & (Y_neg >= 0) & (Y_neg < h)
    energy_pos = np.where(valid_pos, magnitude[np.clip(Y_pos, 0, h-1), np.clip(X_pos, 0, w-1)], 0.0)
    energy_neg = np.where(valid_neg, magnitude[np.clip(Y_neg, 0, h-1), np.clip(X_neg, 0, w-1)], 0.0)
    total_energy = energy_pos.sum(axis=0) + energy_neg.sum(axis=0)
    total_count = valid_pos.sum(axis=0) + valid_neg.sum(axis=0)
    valid_angles = total_count > 0
    avg_energies = np.where(valid_angles, total_energy / (total_count + 1e-10), 0.0)
    if valid_angles.any():
        min_idx = np.argmin(np.where(valid_angles, avg_energies, np.inf))
        max_idx = np.argmax(np.where(valid_angles, avg_energies, -np.inf))
        best_angle = angles_deg[min_idx]
        min_energy = avg_energies[min_idx]
        max_energy = avg_energies[max_idx]
    else:
        best_angle, min_energy, max_energy = 0.0, 0.0, 0.0
    blur_angle = (best_angle + 90) % 180
    anisotropy = (max_energy - min_energy) / (max_energy + 1e-6)
    kernel_size = 1 if anisotropy < 0.05 else max(1, int(anisotropy * 25))
    return kernel_size, blur_angle


def estimate_crf(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    h, w = gray.shape
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    hf_energy = np.mean(np.abs(laplacian))
    cols_4 = np.arange(4, w - 1, 4)
    rows_4 = np.arange(4, h - 1, 4)
    block_diffs_x = np.mean(np.abs(gray[:, cols_4] - gray[:, cols_4 - 1]), axis=0) if len(cols_4) else np.array([])
    block_diffs_y = np.mean(np.abs(gray[rows_4, :] - gray[rows_4 - 1, :]), axis=1) if len(rows_4) else np.array([])
    block_diffs = np.concatenate([block_diffs_x, block_diffs_y])
    cols_interior = np.arange(3, w - 1, 4)
    cols_interior = cols_interior[cols_interior % 4 != 0]
    interior_diffs = np.mean(np.abs(gray[:, cols_interior] - gray[:, cols_interior - 1]), axis=0) if len(cols_interior) else np.array([])
    avg_block = np.median(block_diffs) if len(block_diffs) else 0
    avg_interior = np.median(interior_diffs) if len(interior_diffs) else 1
    blockiness = avg_block / (avg_interior + 1e-6)
    if hf_energy > 30:
        crf_from_hf = 15
    elif hf_energy > 15:
        crf_from_hf = 23
    elif hf_energy > 8:
        crf_from_hf = 30
    else:
        crf_from_hf = 38
    crf_from_blockiness = 18 + int((blockiness - 1.0) * 20)
    crf = int(0.6 * crf_from_hf + 0.4 * crf_from_blockiness)
    return max(0, min(51, crf))


def apply_h264_compression(image, crf=23):
    h, w = image.shape[:2]
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, 'in.png')
        out_path = os.path.join(tmpdir, 'out.mp4')
        dec_path = os.path.join(tmpdir, 'dec.png')
        cv2.imwrite(in_path, image)
        subprocess.run([
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-i', in_path,
            '-c:v', 'libx264', '-crf', str(crf),
            '-pix_fmt', 'yuv420p', '-frames:v', '1',
            out_path
        ], check=True, capture_output=True)
        subprocess.run([
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-i', out_path, '-frames:v', '1', dec_path
        ], check=True, capture_output=True)
        result = cv2.imread(dec_path)
    if result.shape[:2] != (h, w):
        result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return result


def apply_motion_blur(image, kernel_size=11, angle=0.0):
    if kernel_size <= 1:
        return image.copy()
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    angle_rad = np.deg2rad(angle)
    dx, dy = np.cos(angle_rad), np.sin(angle_rad)
    for i in range(kernel_size):
        t = i - center
        x, y = int(round(center + t * dx)), int(round(center + t * dy))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0
    kernel /= kernel.sum() + 1e-8
    return cv2.filter2D(image, -1, kernel)


def apply_gaussian_blur(image, sigma):
    if sigma <= 0:
        return image.copy()
    ksize = int(np.ceil(sigma * 6)) | 1
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def postprocess_foreground(aligned, target, level=2):
    if level <= 0 or level not in PP_LEVELS:
        return aligned

    cfg = PP_LEVELS[level]

    # Estimate target degradation
    blur_sigma = estimate_blur_level(target)
    motion_kernel, motion_angle = estimate_motion_blur(target)
    crf = estimate_crf(target)

    # Detect foreground
    fg_mask = detect_foreground_mask(aligned, target)
    if np.mean(fg_mask) < 0.001:
        return aligned

    degraded = aligned.copy()

    # 1. Gaussian blur
    applied_sigma = max(blur_sigma * cfg['sigma_mult'], cfg['sigma_min'])
    if applied_sigma > 0:
        degraded = apply_gaussian_blur(degraded, applied_sigma)

    # 2. Motion blur
    applied_motion = max(motion_kernel, cfg['motion_min'])
    if applied_motion > 1:
        degraded = apply_motion_blur(degraded, applied_motion, motion_angle)

    # 3. H.264 CRF compression
    applied_crf = min(crf + cfg['crf_boost'], 51)
    try:
        degraded = apply_h264_compression(degraded, applied_crf)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to JPEG
        jpeg_q = max(5, 95 - applied_crf * 2)
        _, encoded = cv2.imencode('.jpg', degraded, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
        degraded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    # Blend into foreground only
    mask_3ch = fg_mask[:, :, np.newaxis]
    result = (degraded.astype(np.float32) * mask_3ch +
              aligned.astype(np.float32) * (1.0 - mask_3ch))
    return np.clip(result, 0, 255).astype(np.uint8)


# ============== Paste-back unedited regions ==============

def detect_unedited_mask(aligned, target, threshold=45, min_edit_area=2000,
                         safety_radius=40, blur_size=31):
    diff = cv2.absdiff(aligned, target)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, edited_binary = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)

    grow_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    edited_binary = cv2.dilate(edited_binary, grow_kernel, iterations=1)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    edited_binary = cv2.morphologyEx(edited_binary, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edited_binary, connectivity=8)
    cleaned = np.zeros_like(edited_binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_edit_area:
            cleaned[labels == i] = 255
    edited_binary = cleaned

    if safety_radius > 0:
        safety_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                  (safety_radius * 2 + 1, safety_radius * 2 + 1))
        edited_binary = cv2.dilate(edited_binary, safety_kernel, iterations=1)

    unedited_binary = 255 - edited_binary

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    unedited_binary = cv2.morphologyEx(unedited_binary, cv2.MORPH_OPEN, open_kernel, iterations=2)

    blur_size = blur_size | 1
    soft_mask = cv2.GaussianBlur(unedited_binary.astype(np.float32) / 255.0,
                                 (blur_size, blur_size), 0)
    return soft_mask


def paste_unedited_regions(aligned, target, mask):
    mask_3ch = mask[:, :, np.newaxis]
    result = target.astype(np.float32) * mask_3ch + aligned.astype(np.float32) * (1.0 - mask_3ch)
    return np.clip(result, 0, 255).astype(np.uint8)


# ============== Alignment Pipeline ==============

def align_image(source_img, target_img, pp_level=2, paste_back=True):
    target_h, target_w = target_img.shape[:2]
    target_size = (target_w, target_h)
    source_resized = cv2.resize(source_img, target_size, interpolation=cv2.INTER_LANCZOS4)
    naive_resized = source_resized.copy()

    kp_src, desc_src = extract_features(source_resized)
    kp_tgt, desc_tgt = extract_features(target_img)
    matches = match_features(desc_src, desc_tgt)

    color_mask = None
    skip_warp = False
    max_deviation_px = 100
    if len(matches) >= 4:
        H, mask = compute_homography(kp_src, kp_tgt, matches)
        if H is not None and mask is not None:
            deviation = homography_deviation(H, target_w, target_h)
            if deviation > max_deviation_px:
                skip_warp = True
                aligned = source_resized
            else:
                inlier_mask = mask.ravel()
                aligned = cv2.warpPerspective(source_resized, H, target_size,
                                              flags=cv2.INTER_LANCZOS4,
                                              borderMode=cv2.BORDER_REPLICATE)
                color_mask = create_inlier_mask(kp_tgt, matches, inlier_mask,
                                                target_img.shape, radius=50)
        else:
            skip_warp = True
            aligned = source_resized
    else:
        skip_warp = True
        aligned = source_resized

    result = fast_color_transfer(aligned, target_img, mask=color_mask)

    # Paste back unedited regions from target (skip if warp was skipped)
    pre_paste = result.copy()
    unedited_mask = None
    if paste_back and not skip_warp:
        unedited_mask = detect_unedited_mask(result, target_img)
        result = paste_unedited_regions(result, target_img, unedited_mask)

    # Post-processing (only affects edited regions, then re-paste)
    pp_result = None
    if pp_level > 0:
        pp_result = postprocess_foreground(result, target_img, level=pp_level)
        if paste_back and not skip_warp and unedited_mask is not None:
            pp_result = paste_unedited_regions(pp_result, target_img, unedited_mask)

    final = pp_result if pp_result is not None else result
    return final, naive_resized, result, pre_paste, unedited_mask, pp_result


def compute_diff_image(img1, img2, amplify=3.0):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    abs_diff = np.abs(gray1 - gray2)
    diff_vis = np.clip(abs_diff * amplify, 0, 255).astype(np.uint8)
    return cv2.cvtColor(diff_vis, cv2.COLOR_GRAY2BGR)


def create_visualization_panel(naive_resized, aligned, target, pre_paste=None,
                                unedited_mask=None, postprocessed=None):
    h, w = target.shape[:2]
    label_height = 40
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    font_color = (255, 255, 255)
    bg_color = (40, 40, 40)

    def add_label(img, text):
        label_bar = np.full((label_height, img.shape[1], 3), bg_color, dtype=np.uint8)
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = (img.shape[1] - text_size[0]) // 2
        text_y = (label_height + text_size[1]) // 2
        cv2.putText(label_bar, text, (text_x, text_y), font, font_scale, font_color, font_thickness)
        return np.vstack([label_bar, img])

    empty = np.full((h + label_height, w, 3), bg_color, dtype=np.uint8)

    mask_vis_bgr = None
    if unedited_mask is not None:
        mask_vis = (unedited_mask * 255).astype(np.uint8)
        mask_vis_bgr = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)

    diff_naive = compute_diff_image(naive_resized, target)
    diff_aligned = compute_diff_image(aligned, target)
    diff_pre_paste = compute_diff_image(pre_paste, target) if pre_paste is not None else None

    if postprocessed is not None:
        diff_pp = compute_diff_image(postprocessed, target)
        diff_aligned_vs_pp = compute_diff_image(aligned, postprocessed)
        row1_items = [
            add_label(naive_resized, "Naive Resize"),
            add_label(aligned, "Aligned+Pasted"),
            add_label(postprocessed, "Post-processed"),
            add_label(target, "Target Reference"),
        ]
        row2_items = [
            add_label(diff_naive, "Diff: Naive vs Target"),
            add_label(diff_pre_paste, "Diff: Pre-paste vs Target") if diff_pre_paste is not None else empty,
            add_label(diff_aligned, "Diff: Pasted vs Target"),
            add_label(diff_pp, "Diff: Post-proc vs Target"),
        ]
        if mask_vis_bgr is not None:
            row1_items.append(add_label(mask_vis_bgr, "Unedited Mask"))
            row2_items.append(empty)
    else:
        row1_items = [
            add_label(naive_resized, "Naive Resize"),
            add_label(aligned, "Aligned+Pasted"),
            add_label(target, "Target Reference"),
        ]
        row2_items = [
            add_label(diff_naive, "Diff: Naive vs Target"),
            add_label(diff_pre_paste, "Diff: Pre-paste vs Target") if diff_pre_paste is not None else empty,
            add_label(diff_aligned, "Diff: Pasted vs Target"),
        ]
        if mask_vis_bgr is not None:
            row1_items.append(add_label(mask_vis_bgr, "Unedited Mask"))
            row2_items.append(empty)

    row1 = np.hstack(row1_items)
    row2 = np.hstack(row2_items)
    return np.vstack([row1, row2])


# ============== FastAPI App ==============

app = FastAPI(title="Image Aligner API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def decode_image(data: bytes) -> np.ndarray:
    img_array = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


def encode_image_png(img: np.ndarray) -> bytes:
    _, buffer = cv2.imencode('.png', img)
    return buffer.tobytes()


@app.post("/api/align")
async def align_api(
    source: UploadFile = File(..., description="Source image to align"),
    target: UploadFile = File(..., description="Target reference image"),
    pp: int = Form(2, description="Post-processing level 0-3 (0=none, default=2)"),
    paste_back: bool = Form(True, description="Paste back unedited regions from target (default=true)")
):
    """
    Align source image to target image.
    Returns the aligned image as PNG.
    """
    try:
        pp_level = max(0, min(3, pp))
        source_data = await source.read()
        target_data = await target.read()

        source_img = decode_image(source_data)
        target_img = decode_image(target_data)

        if source_img is None or target_img is None:
            raise HTTPException(status_code=400, detail="Failed to decode images")

        final, *_ = align_image(source_img, target_img, pp_level=pp_level, paste_back=paste_back)
        png_bytes = encode_image_png(final)

        return Response(content=png_bytes, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/align/base64")
async def align_base64_api(
    source: UploadFile = File(...),
    target: UploadFile = File(...),
    pp: int = Form(2, description="Post-processing level 0-3 (0=none, default=2)"),
    paste_back: bool = Form(True, description="Paste back unedited regions from target (default=true)")
):
    """
    Align source image to target image.
    Returns the aligned image as base64-encoded PNG.
    """
    try:
        pp_level = max(0, min(3, pp))
        source_data = await source.read()
        target_data = await target.read()

        source_img = decode_image(source_data)
        target_img = decode_image(target_data)

        if source_img is None or target_img is None:
            raise HTTPException(status_code=400, detail="Failed to decode images")

        final, *_ = align_image(source_img, target_img, pp_level=pp_level, paste_back=paste_back)
        png_bytes = encode_image_png(final)
        b64 = base64.b64encode(png_bytes).decode('utf-8')

        return {"image": f"data:image/png;base64,{b64}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/align/viz")
async def align_viz_api(
    source: UploadFile = File(...),
    target: UploadFile = File(...),
    pp: int = Form(2, description="Post-processing level 0-3 (0=none, default=2)"),
    paste_back: bool = Form(True, description="Paste back unedited regions from target (default=true)")
):
    """
    Align source image to target and return visualization panel + final result.
    """
    try:
        pp_level = max(0, min(3, pp))
        source_data = await source.read()
        target_data = await target.read()

        source_img = decode_image(source_data)
        target_img = decode_image(target_data)

        if source_img is None or target_img is None:
            raise HTTPException(status_code=400, detail="Failed to decode images")

        final, naive_resized, pasted, pre_paste, unedited_mask, pp_result = \
            align_image(source_img, target_img, pp_level=pp_level, paste_back=paste_back)

        panel = create_visualization_panel(
            naive_resized, pasted, target_img,
            pre_paste=pre_paste,
            unedited_mask=unedited_mask,
            postprocessed=pp_result
        )

        panel_bytes = encode_image_png(panel)
        final_bytes = encode_image_png(final)
        panel_b64 = base64.b64encode(panel_bytes).decode('utf-8')
        final_b64 = base64.b64encode(final_bytes).decode('utf-8')

        return {
            "panel": f"data:image/png;base64,{panel_b64}",
            "image": f"data:image/png;base64,{final_b64}",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Aligner</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            color: #e8e8e8;
            padding: 2rem;
        }
        .dedication {
            text-align: center;
            padding: 2rem;
            background: linear-gradient(135deg, rgba(255, 121, 198, 0.15), rgba(139, 233, 253, 0.15));
            border-radius: 12px;
            margin-bottom: 2rem;
        }
        .dedication h2 { font-size: 1.2rem; font-weight: 300; margin-bottom: 0.5rem; }
        .dedication .names {
            font-size: 1.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, #ff79c6, #ffb86c, #8be9fd, #50fa7b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .dedication .team { font-size: 1.1rem; color: #8be9fd; margin-top: 0.5rem; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { text-align: center; margin-bottom: 0.5rem; font-weight: 300; font-size: 2.5rem; }
        .subtitle { text-align: center; color: #888; margin-bottom: 2rem; }
        .upload-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-bottom: 2rem; }
        .upload-box {
            background: rgba(255,255,255,0.03);
            border: 2px dashed rgba(255,255,255,0.2);
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            min-height: 250px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .upload-box:hover { border-color: rgba(255,255,255,0.4); background: rgba(255,255,255,0.05); }
        .upload-box.has-image { padding: 1rem; }
        .upload-box img { max-width: 100%; max-height: 200px; border-radius: 8px; }
        .upload-box input { display: none; }
        .upload-box h3 { margin-bottom: 0.5rem; }
        .upload-box.source h3 { color: #8be9fd; }
        .upload-box.target h3 { color: #ffb86c; }
        .options-row {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 1.5rem;
            margin-bottom: 2rem;
            flex-wrap: wrap;
        }
        .options-row label {
            font-size: 0.95rem;
            color: #aaa;
        }
        .pp-select {
            background: rgba(255,255,255,0.08);
            color: #e8e8e8;
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 6px;
            padding: 0.5rem 1rem;
            font-size: 0.95rem;
            cursor: pointer;
        }
        .pp-select option { background: #1a1a2e; color: #e8e8e8; }
        .btn {
            display: block;
            width: 100%;
            max-width: 300px;
            margin: 0 auto 2rem;
            padding: 1rem 2rem;
            font-size: 1.1rem;
            font-weight: 600;
            border: none;
            border-radius: 8px;
            background: linear-gradient(135deg, #50fa7b, #00d9ff);
            color: #1a1a2e;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 10px 30px rgba(80,250,123,0.3); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .result { text-align: center; display: none; }
        .result.show { display: block; }
        .result img { max-width: 100%; border-radius: 8px; margin: 1rem 0; }
        .result a {
            display: inline-block;
            padding: 0.8rem 2rem;
            background: rgba(255,255,255,0.1);
            color: #fff;
            text-decoration: none;
            border-radius: 8px;
            margin-top: 1rem;
        }
        .loading { display: none; text-align: center; padding: 2rem; }
        .loading.show { display: block; }
        .spinner {
            width: 50px; height: 50px;
            border: 3px solid rgba(255,255,255,0.1);
            border-top-color: #50fa7b;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .api-docs {
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            padding: 2rem;
            margin-top: 3rem;
        }
        .api-docs h2 { margin-bottom: 1rem; color: #50fa7b; }
        .api-docs pre {
            background: rgba(0,0,0,0.3);
            padding: 1rem;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 0.9rem;
        }
        .api-docs code { color: #8be9fd; }
        @media (max-width: 768px) { .upload-grid { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="container">
        <div class="dedication">
            <h2>Dedicated with &#9829; love and devotion to</h2>
            <div class="names">Alon Y., Daniel B., Denis Z., Tal S., Adi B.</div>
            <div class="team">and the rest of the Animation Taskforce 2026</div>
        </div>

        <h1>&#127919; Image Aligner</h1>
        <p class="subtitle">Geometric alignment with background-aware color matching</p>

        <div class="upload-grid">
            <div class="upload-box source" onclick="document.getElementById('sourceInput').click()">
                <input type="file" id="sourceInput" accept="image/*">
                <h3>&#128247; Source Image</h3>
                <p>Click to upload</p>
            </div>
            <div class="upload-box target" onclick="document.getElementById('targetInput').click()">
                <input type="file" id="targetInput" accept="image/*">
                <h3>&#127919; Target Reference</h3>
                <p>Click to upload</p>
            </div>
        </div>

        <div class="options-row">
            <label for="ppLevel">Post-processing:</label>
            <select id="ppLevel" class="pp-select">
                <option value="0">0 - None</option>
                <option value="1">1 - Weak</option>
                <option value="2" selected>2 - Medium (default)</option>
                <option value="3">3 - Strong</option>
            </select>
            <label style="display:flex;align-items:center;gap:0.4rem;cursor:pointer;">
                <input type="checkbox" id="pasteBack" checked style="width:18px;height:18px;cursor:pointer;">
                Paste back unedited regions
            </label>
        </div>

        <button class="btn" id="alignBtn" disabled onclick="alignImages()">&#10024; Align Images</button>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Aligning images...</p>
        </div>

        <div class="result" id="result">
            <h2>&#10024; Visualization</h2>
            <img id="panelImg" src="" style="max-width:100%">
            <br>
            <a id="downloadLink" download="aligned.png">Download Aligned Image</a>
        </div>

        <div class="api-docs">
            <h2>&#128225; API Usage</h2>
            <p>POST to <code>/api/align</code> with multipart form data:</p>
            <pre><code>// JavaScript (fetch)
const formData = new FormData();
formData.append('source', sourceFile);
formData.append('target', targetFile);
formData.append('pp', '2');  // 0=none, 1=weak, 2=medium, 3=strong

const response = await fetch('/api/align', {
    method: 'POST',
    body: formData
});
const blob = await response.blob();
const url = URL.createObjectURL(blob);

// Or use /api/align/base64 for base64 response:
const response = await fetch('/api/align/base64', {
    method: 'POST',
    body: formData
});
const data = await response.json();
console.log(data.image); // data:image/png;base64,...</code></pre>
        </div>
    </div>

    <script>
        let sourceFile = null;
        let targetFile = null;

        document.getElementById('sourceInput').onchange = (e) => {
            sourceFile = e.target.files[0];
            showPreview('source', sourceFile);
            updateButton();
        };

        document.getElementById('targetInput').onchange = (e) => {
            targetFile = e.target.files[0];
            showPreview('target', targetFile);
            updateButton();
        };

        function showPreview(type, file) {
            const box = document.querySelector(`.upload-box.${type}`);
            const reader = new FileReader();
            reader.onload = (e) => {
                box.innerHTML = `<img src="${e.target.result}">`;
                box.classList.add('has-image');
            };
            reader.readAsDataURL(file);
        }

        function updateButton() {
            document.getElementById('alignBtn').disabled = !(sourceFile && targetFile);
        }

        async function alignImages() {
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');

            loading.classList.add('show');
            result.classList.remove('show');

            try {
                const formData = new FormData();
                formData.append('source', sourceFile);
                formData.append('target', targetFile);
                formData.append('pp', document.getElementById('ppLevel').value);
                formData.append('paste_back', document.getElementById('pasteBack').checked ? 'true' : 'false');

                const response = await fetch('/api/align/viz', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) throw new Error('Alignment failed');

                const data = await response.json();

                document.getElementById('panelImg').src = data.panel;
                document.getElementById('downloadLink').href = data.image;
                result.classList.add('show');
            } catch (err) {
                alert('Error: ' + err.message);
            } finally {
                loading.classList.remove('show');
            }
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_CONTENT


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
