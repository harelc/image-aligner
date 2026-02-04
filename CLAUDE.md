# CLAUDE.md

This file provides guidance for Claude Code when working on this repository.

## Project Overview

Image Aligner is a Python tool for aligning images geometrically and matching colors between a source and target image. It's designed for cases where source images have different viewpoints, sizes, aspect ratios, or color profiles compared to reference images.

## Key Files

- `image_aligner.py` - Main script containing all alignment logic
- `pyproject.toml` - Project dependencies (managed by uv)
- `align_to/` - Directory for reference/target images
- `align_from/` - Directory for source images to be aligned
- `aligned_output/` - Output directory (created automatically)

## Architecture

The code is organized as functions in a single file:

1. **Image I/O**: `load_image()`, `resize_to_target()`
2. **Feature Detection**: `extract_features()` - SIFT-based
3. **Feature Matching**: `match_features()` - FLANN with ratio test
4. **Geometric Transform**: `compute_homography()`, `compute_affine()`, `apply_transform()`
5. **Masking**: `create_inlier_mask()` - Creates background mask from RANSAC inliers
6. **Color Transfer**: Multiple methods, all accept optional `mask` parameter:
   - `optimal_transport_color_transfer()` - Monge-Kantorovich
   - `histogram_matching_lab()` - LAB color space
   - `full_histogram_matching()` - Blended approach
   - `piecewise_linear_histogram_transfer()` - CDF-based

## Build & Run

```bash
# Install dependencies
uv sync

# Run batch processing
uv run python image_aligner.py --batch

# Run single pair
uv run python image_aligner.py source.jpg target.jpg
```

## Key Design Decisions

- **Background-based color matching**: Color statistics are computed only from RANSAC inlier regions (background), not the whole image. This prevents foreground objects from affecting the color transfer.
- **Robust estimation**: Uses USAC_MAGSAC with high confidence (0.9999) to handle up to 50% outliers from foreground changes.
- **Multiple color methods**: Different methods suit different scenarios; `full_histogram` is the default and blends multiple approaches.

## Testing

No formal test suite. Test by running batch mode on sample images in `align_from/` and `align_to/` directories and inspecting the visualization panels in `aligned_output/visualizations/`.
