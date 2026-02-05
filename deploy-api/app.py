#!/usr/bin/env python3
"""
Image Aligner - FastAPI Web Interface with API
Dedicated with love and devotion to Alon Y., Daniel B., Denis Z., Tal S.
and the rest of the Animation Taskforce 2026
"""

import io
import base64
import warnings
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.linalg import sqrtm, inv
from skimage import exposure
import uvicorn


# ============== Image Alignment Core ==============

def extract_features(img: np.ndarray) -> tuple:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=10000, contrastThreshold=0.02, edgeThreshold=15)
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


def create_inlier_mask(keypoints, matches, inlier_mask, image_shape, radius=50):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    for i, m in enumerate(matches):
        if inlier_mask[i]:
            pt = keypoints[m.trainIdx].pt
            x, y = int(pt[0]), int(pt[1])
            y_min, y_max = max(0, y - radius), min(h, y + radius + 1)
            x_min, x_max = max(0, x - radius), min(w, x + radius + 1)
            yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
            circle = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
            mask[y_min:y_max, x_min:x_max] |= circle
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
    lookup = np.zeros(n_bins, dtype=np.uint8)
    tgt_idx = 0
    for src_idx in range(n_bins):
        while tgt_idx < n_bins - 1 and tgt_cdf[tgt_idx] < src_cdf[src_idx]:
            tgt_idx += 1
        lookup[src_idx] = tgt_idx
    return lookup


def _build_histogram_lookup_float(src_channel, tgt_channel, n_bins=256):
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
    lab_matched = histogram_matching_lab(source, target, mask)
    cdf_matched = piecewise_linear_histogram_transfer(source, target, mask=mask)
    multi_matched = histogram_matching_rgb(source, target, mask)
    result = (0.5 * lab_matched.astype(np.float32) +
              0.3 * cdf_matched.astype(np.float32) +
              0.2 * multi_matched.astype(np.float32))
    return np.clip(result, 0, 255).astype(np.uint8)


def align_image(source_img, target_img):
    target_h, target_w = target_img.shape[:2]
    target_size = (target_w, target_h)
    source_resized = cv2.resize(source_img, target_size, interpolation=cv2.INTER_LANCZOS4)

    kp_src, desc_src = extract_features(source_resized)
    kp_tgt, desc_tgt = extract_features(target_img)
    matches = match_features(desc_src, desc_tgt)

    color_mask = None
    if len(matches) >= 4:
        H, mask = compute_homography(kp_src, kp_tgt, matches)
        if H is not None and mask is not None:
            inlier_mask = mask.ravel()
            aligned = cv2.warpPerspective(source_resized, H, target_size,
                                          flags=cv2.INTER_LANCZOS4,
                                          borderMode=cv2.BORDER_REPLICATE)
            color_mask = create_inlier_mask(kp_tgt, matches, inlier_mask,
                                            target_img.shape, radius=50)
        else:
            aligned = source_resized
    else:
        aligned = source_resized

    result = full_histogram_matching(aligned, target_img, mask=color_mask)
    return result


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
    target: UploadFile = File(..., description="Target reference image")
):
    """
    Align source image to target image.
    Returns the aligned image as PNG.
    """
    try:
        source_data = await source.read()
        target_data = await target.read()

        source_img = decode_image(source_data)
        target_img = decode_image(target_data)

        if source_img is None or target_img is None:
            raise HTTPException(status_code=400, detail="Failed to decode images")

        aligned = align_image(source_img, target_img)
        png_bytes = encode_image_png(aligned)

        return Response(content=png_bytes, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/align/base64")
async def align_base64_api(
    source: UploadFile = File(...),
    target: UploadFile = File(...)
):
    """
    Align source image to target image.
    Returns the aligned image as base64-encoded PNG.
    """
    try:
        source_data = await source.read()
        target_data = await target.read()

        source_img = decode_image(source_data)
        target_img = decode_image(target_data)

        if source_img is None or target_img is None:
            raise HTTPException(status_code=400, detail="Failed to decode images")

        aligned = align_image(source_img, target_img)
        png_bytes = encode_image_png(aligned)
        b64 = base64.b64encode(png_bytes).decode('utf-8')

        return {"image": f"data:image/png;base64,{b64}"}

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
            <h2>Dedicated with ♥ love and devotion to</h2>
            <div class="names">Alon Y., Daniel B., Denis Z., Tal S.</div>
            <div class="team">and the rest of the Animation Taskforce 2026</div>
        </div>

        <h1>🎯 Image Aligner</h1>
        <p class="subtitle">Geometric alignment with background-aware color matching</p>

        <div class="upload-grid">
            <div class="upload-box source" onclick="document.getElementById('sourceInput').click()">
                <input type="file" id="sourceInput" accept="image/*">
                <h3>📷 Source Image</h3>
                <p>Click to upload</p>
            </div>
            <div class="upload-box target" onclick="document.getElementById('targetInput').click()">
                <input type="file" id="targetInput" accept="image/*">
                <h3>🎯 Target Reference</h3>
                <p>Click to upload</p>
            </div>
        </div>

        <button class="btn" id="alignBtn" disabled onclick="alignImages()">✨ Align Images</button>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Aligning images...</p>
        </div>

        <div class="result" id="result">
            <h2>✨ Aligned Result</h2>
            <img id="resultImg" src="">
            <br>
            <a id="downloadLink" download="aligned.png">Download Aligned Image</a>
        </div>

        <div class="api-docs">
            <h2>📡 API Usage</h2>
            <p>POST to <code>/api/align</code> with multipart form data:</p>
            <pre><code>// JavaScript (fetch)
const formData = new FormData();
formData.append('source', sourceFile);
formData.append('target', targetFile);

const response = await fetch('/api/align', {
    method: 'POST',
    body: formData
});
const blob = await response.blob();
const url = URL.createObjectURL(blob);

// Or use /api/align/base64 to get base64 response:
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

                const response = await fetch('/api/align', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) throw new Error('Alignment failed');

                const blob = await response.blob();
                const url = URL.createObjectURL(blob);

                document.getElementById('resultImg').src = url;
                document.getElementById('downloadLink').href = url;
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
