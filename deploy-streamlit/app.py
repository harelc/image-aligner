#!/usr/bin/env python3
"""
Image Aligner - Streamlit Web Interface
Dedicated with love and devotion to Alon Y., Daniel B., Denis Z., Tal S., Adi B.
and the rest of the Animation Taskforce 2026
"""

import warnings
import cv2
import numpy as np
import streamlit as st
from scipy.linalg import sqrtm, inv
from skimage import exposure
from PIL import Image
import io


# ============== Image Alignment Core ==============

def extract_features(img: np.ndarray) -> tuple:
    """Extract SIFT features from grayscale image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=10000, contrastThreshold=0.02, edgeThreshold=15)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_features(desc1: np.ndarray, desc2: np.ndarray, ratio_thresh: float = 0.85) -> list:
    """Match features using FLANN with Lowe's ratio test."""
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
    """Compute homography using RANSAC."""
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


def create_inlier_mask(keypoints, matches, inlier_mask, image_shape, radius=50):
    """Create a boolean mask from inlier keypoints."""
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=bool)

    for i, m in enumerate(matches):
        if inlier_mask[i]:
            pt = keypoints[m.trainIdx].pt
            x, y = int(pt[0]), int(pt[1])

            y_min = max(0, y - radius)
            y_max = min(h, y + radius + 1)
            x_min = max(0, x - radius)
            x_max = min(w, x + radius + 1)

            yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
            circle = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
            mask[y_min:y_max, x_min:x_max] |= circle

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2).astype(bool)

    return mask


def _build_histogram_lookup(src_channel, tgt_channel, n_bins=256):
    """Build a histogram matching lookup table."""
    src_hist, _ = np.histogram(src_channel.flatten(), bins=n_bins, range=(0, 256))
    tgt_hist, _ = np.histogram(tgt_channel.flatten(), bins=n_bins, range=(0, 256))

    src_cdf = np.cumsum(src_hist).astype(np.float64)
    src_cdf = src_cdf / (src_cdf[-1] + 1e-10)

    tgt_cdf = np.cumsum(tgt_hist).astype(np.float64)
    tgt_cdf = tgt_cdf / (tgt_cdf[-1] + 1e-10)

    lookup = np.zeros(n_bins, dtype=np.uint8)
    tgt_idx = 0
    for src_idx in range(n_bins):
        while tgt_idx < n_bins - 1 and tgt_cdf[tgt_idx] < src_cdf[src_idx]:
            tgt_idx += 1
        lookup[src_idx] = tgt_idx

    return lookup


def _build_histogram_lookup_float(src_channel, tgt_channel, n_bins=256):
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


def histogram_matching_lab(source, target, mask=None):
    """Histogram matching in LAB color space."""
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    if mask is None:
        matched_lab = np.zeros_like(source_lab)
        for i in range(3):
            matched_lab[:, :, i] = exposure.match_histograms(
                source_lab[:, :, i], target_lab[:, :, i]
            )
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
    matched = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)

    return matched


def histogram_matching_rgb(source, target, mask=None):
    """Simple per-channel histogram matching."""
    if mask is None:
        matched = np.zeros_like(source)
        for i in range(3):
            matched[:, :, i] = exposure.match_histograms(
                source[:, :, i], target[:, :, i]
            )
        return matched

    matched = np.zeros_like(source)
    for i in range(3):
        src_masked = source[:, :, i][mask]
        tgt_masked = target[:, :, i][mask]
        lookup = _build_histogram_lookup(src_masked, tgt_masked)
        matched[:, :, i] = lookup[source[:, :, i]]
    return matched


def piecewise_linear_histogram_transfer(source, target, n_bins=256, mask=None):
    """Piecewise linear histogram transfer using CDF matching."""
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

        lookup = np.zeros(n_bins, dtype=np.float32)
        tgt_idx = 0
        for src_idx in range(n_bins):
            while tgt_idx < n_bins - 1 and tgt_cdf[tgt_idx] < src_cdf[src_idx]:
                tgt_idx += 1
            lookup[src_idx] = tgt_idx

        src_img = source[:, :, c].astype(np.float32)
        src_floor = np.floor(src_img).astype(np.int32)
        src_ceil = np.minimum(src_floor + 1, n_bins - 1)
        src_frac = src_img - src_floor
        src_floor = np.clip(src_floor, 0, n_bins - 1)

        result[:, :, c] = (1 - src_frac) * lookup[src_floor] + src_frac * lookup[src_ceil]

    return np.clip(result, 0, 255).astype(np.uint8)


def full_histogram_matching(source, target, mask=None):
    """Comprehensive histogram matching combining multiple techniques."""
    lab_matched = histogram_matching_lab(source, target, mask)
    cdf_matched = piecewise_linear_histogram_transfer(source, target, mask=mask)
    multi_matched = histogram_matching_rgb(source, target, mask)

    result = (0.5 * lab_matched.astype(np.float32) +
              0.3 * cdf_matched.astype(np.float32) +
              0.2 * multi_matched.astype(np.float32))

    return np.clip(result, 0, 255).astype(np.uint8)


def compute_diff_image(img1, img2, amplify=3.0):
    """Compute a colored difference image."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)

    diff = gray1 - gray2

    diff_img = np.zeros((*gray1.shape, 3), dtype=np.uint8)
    pos_diff = np.clip(diff * amplify, 0, 255).astype(np.uint8)
    neg_diff = np.clip(-diff * amplify, 0, 255).astype(np.uint8)

    diff_img[:, :, 2] = pos_diff
    diff_img[:, :, 0] = neg_diff
    similar = 255 - np.clip(np.abs(diff) * amplify, 0, 255).astype(np.uint8)
    diff_img[:, :, 1] = similar // 4

    return diff_img


def create_visualization_panel(naive_resized, aligned, target, title=""):
    """Create a visualization panel with comparison images."""
    h, w = target.shape[:2]

    diff_naive = compute_diff_image(naive_resized, target)
    diff_aligned = compute_diff_image(aligned, target)

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

    naive_labeled = add_label(naive_resized, "Naive Resize")
    aligned_labeled = add_label(aligned, "After Alignment")
    target_labeled = add_label(target, "Target Reference")
    diff_naive_labeled = add_label(diff_naive, "Diff: Naive vs Target")
    diff_aligned_labeled = add_label(diff_aligned, "Diff: Aligned vs Target")

    empty = np.full((h + label_height, w, 3), bg_color, dtype=np.uint8)

    row1 = np.hstack([naive_labeled, aligned_labeled, target_labeled])
    row2 = np.hstack([diff_naive_labeled, diff_aligned_labeled, empty])

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


def detect_unedited_mask(aligned, target, threshold=45, min_edit_area=2000,
                         safety_radius=8, blur_size=31):
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


def align_image(source_img, target_img):
    """Main alignment function."""
    target_h, target_w = target_img.shape[:2]
    target_size = (target_w, target_h)

    # Resize source to target dimensions
    source_resized = cv2.resize(source_img, target_size, interpolation=cv2.INTER_LANCZOS4)
    naive_resized = source_resized.copy()

    # Extract and match features
    kp_src, desc_src = extract_features(source_resized)
    kp_tgt, desc_tgt = extract_features(target_img)
    matches = match_features(desc_src, desc_tgt)

    # Compute transform
    inlier_mask = None
    color_mask = None

    if len(matches) >= 4:
        H, mask = compute_homography(kp_src, kp_tgt, matches)

        if H is not None and mask is not None:
            inlier_mask = mask.ravel()
            aligned = cv2.warpPerspective(source_resized, H, target_size,
                                          flags=cv2.INTER_LANCZOS4,
                                          borderMode=cv2.BORDER_REPLICATE)

            # Create background mask for color matching
            color_mask = create_inlier_mask(kp_tgt, matches, inlier_mask,
                                            target_img.shape, radius=50)
        else:
            aligned = source_resized
    else:
        aligned = source_resized

    # Color matching
    result = full_histogram_matching(aligned, target_img, mask=color_mask)

    # Paste back unedited regions from target
    unedited_mask = detect_unedited_mask(result, target_img)
    result = paste_unedited_regions(result, target_img, unedited_mask)

    return result, naive_resized, aligned


def pil_to_cv2(pil_image):
    """Convert PIL Image to OpenCV BGR format."""
    rgb = np.array(pil_image)
    if len(rgb.shape) == 2:  # Grayscale
        return cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    elif rgb.shape[2] == 4:  # RGBA
        return cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
    else:  # RGB
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image):
    """Convert OpenCV BGR to PIL Image."""
    rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# ============== Streamlit Interface ==============

st.set_page_config(
    page_title="Image Aligner",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .dedication {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, rgba(255, 121, 198, 0.15), rgba(139, 233, 253, 0.15));
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .dedication h2 {
        font-size: 1.2rem;
        font-weight: 300;
        color: #fff;
        margin: 0 0 0.5rem 0;
    }
    .dedication .names {
        font-size: 1.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #ff79c6, #ffb86c, #8be9fd, #50fa7b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .dedication .team {
        font-size: 1.1rem;
        color: #8be9fd;
        margin-top: 0.5rem;
    }
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
</style>
""", unsafe_allow_html=True)

# Dedication
st.markdown("""
<div class="dedication">
    <h2>Dedicated with ♥ love and devotion to</h2>
    <div class="names">Alon Y., Daniel B., Denis Z., Tal S., Adi B.</div>
    <div class="team">and the rest of the Animation Taskforce 2026</div>
</div>
""", unsafe_allow_html=True)

# Title
st.title("🎯 Image Aligner")
st.markdown("Geometric alignment with background-aware color matching")

# File uploaders
col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Source Image")
    source_file = st.file_uploader("Upload source image (to align)", type=['png', 'jpg', 'jpeg', 'webp'], key="source")

with col2:
    st.subheader("🎯 Target Reference")
    target_file = st.file_uploader("Upload target image (reference)", type=['png', 'jpg', 'jpeg', 'webp'], key="target")

# Show uploaded images
if source_file and target_file:
    col1, col2 = st.columns(2)

    source_pil = Image.open(source_file)
    target_pil = Image.open(target_file)

    with col1:
        st.image(source_pil, caption=f"Source: {source_pil.size[0]}x{source_pil.size[1]}", use_container_width=True)

    with col2:
        st.image(target_pil, caption=f"Target: {target_pil.size[0]}x{target_pil.size[1]}", use_container_width=True)

    # Align button
    if st.button("✨ Align Images", type="primary", use_container_width=True):
        with st.spinner("Aligning images... This may take a moment."):
            # Convert to OpenCV format
            source_cv2 = pil_to_cv2(source_pil)
            target_cv2 = pil_to_cv2(target_pil)

            # Align
            aligned, naive_resized, _ = align_image(source_cv2, target_cv2)

            # Create visualization panel
            panel = create_visualization_panel(naive_resized, aligned, target_cv2, "Alignment Result")

        st.success("Alignment complete!")

        # Results
        st.subheader("📊 Visualization Panel")
        panel_pil = cv2_to_pil(panel)
        st.image(panel_pil, use_container_width=True)

        # Download buttons
        st.subheader("💾 Download Results")

        col1, col2 = st.columns(2)

        # Aligned image download
        aligned_pil = cv2_to_pil(aligned)
        buf_aligned = io.BytesIO()
        aligned_pil.save(buf_aligned, format='PNG')

        with col1:
            st.download_button(
                label="Download Aligned Image",
                data=buf_aligned.getvalue(),
                file_name="aligned.png",
                mime="image/png",
                use_container_width=True
            )

        # Panel download
        buf_panel = io.BytesIO()
        panel_pil.save(buf_panel, format='PNG')

        with col2:
            st.download_button(
                label="Download Visualization Panel",
                data=buf_panel.getvalue(),
                file_name="visualization.png",
                mime="image/png",
                use_container_width=True
            )

        # Show aligned result
        st.subheader("✨ Aligned Result")
        st.image(aligned_pil, use_container_width=True)

else:
    st.info("👆 Upload both a source image and a target reference image to get started.")

# Footer
st.markdown("---")
st.markdown("""
**How it works:**
1. Upload a source image and a target reference image
2. Click "Align Images"
3. The tool will geometrically align and color-match the source to the target
4. Download the aligned result
""")
