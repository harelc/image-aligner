# Image Aligner

A Python tool for aligning images with geometric transformation and color matching. Designed to handle scenarios where source images have different viewpoints, sizes, or color characteristics compared to reference images.

## Features

- **Geometric Alignment**: Uses SIFT feature detection with RANSAC-based homography estimation to align images robustly, even with up to ~50% foreground changes
- **Color Matching**: Multiple algorithms including optimal transport (Monge-Kantorovich) and histogram matching in LAB color space
- **Background-Aware Color Transfer**: Color statistics are computed from inlier (background) regions only, preventing foreground objects from skewing the color matching
- **Visualization Panels**: Generates comparison panels showing naive resize vs aligned vs target, with difference visualizations

## Installation

Requires Python 3.10+

```bash
# Clone the repository
git clone https://github.com/Harelc/image-aligner.git
cd image-aligner

# Create virtual environment and install dependencies
uv sync
# or
pip install -r requirements.txt
```

### Dependencies

- opencv-python
- numpy
- scipy
- scikit-image

## Usage

### Single Pair Mode

```bash
# Basic usage
python image_aligner.py source.jpg target.jpg

# Specify output path
python image_aligner.py source.jpg target.jpg -o result.png

# Use LAB color matching
python image_aligner.py photo.png reference.png --color lab

# Disable visualization generation
python image_aligner.py source.jpg target.jpg --no-viz
```

### Batch Mode

Process multiple image pairs from directories:

```bash
# Default directories (align_from/ -> align_to/)
python image_aligner.py --batch

# Custom directories
python image_aligner.py --batch -t refs/ -f inputs/ -o outputs/
```

In batch mode, images are matched by filename prefix. For example:
- `align_to/pool.png` matches `align_from/pool_fish.png`, `align_from/pool_edited.jpg`
- `align_to/photo.jpg` matches `align_from/photo_v2.png`

### Options

| Option | Description |
|--------|-------------|
| `--batch`, `-b` | Batch mode: process directories |
| `--align-to`, `-t` | Reference images directory (default: `align_to`) |
| `--align-from`, `-f` | Source images directory (default: `align_from`) |
| `--output`, `-o` | Output path (file or directory) |
| `--affine`, `-a` | Use affine transform instead of homography |
| `--color`, `-c` | Color method: `full_histogram`, `optimal_transport`, `lab`, `cdf`, `histogram` |
| `--suffix`, `-s` | Output filename suffix (default: `_aligned`) |
| `--no-viz` | Disable visualization panel generation |

## How It Works

1. **Feature Extraction**: SIFT keypoints are detected in both source and target images
2. **Feature Matching**: FLANN-based matching with Lowe's ratio test
3. **Geometric Transform**: USAC_MAGSAC robust estimation for homography/affine transform
4. **Inlier Masking**: RANSAC inliers identify background regions for color statistics
5. **Color Transfer**: Statistics computed from background only, applied globally

## Output

- Aligned images saved to output directory
- Visualization panels in `visualizations/` subdirectory showing:
  - Row 1: Naive Resize | After Alignment | Target Reference
  - Row 2: Diff (Naive vs Target) | Diff (Aligned vs Target)

## License

MIT
