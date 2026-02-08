#!/usr/bin/env python3
"""Analyze homographies found by image aligner."""

import numpy as np
from pathlib import Path
import cv2
from image_aligner import (
    find_matching_pairs, load_image, resize_to_target,
    extract_features, match_features, compute_homography
)


def decompose_homography(H, K=None):
    """
    Analyze what transformation a homography represents.
    Returns scale, rotation, translation, and perspective components.
    """
    if H is None:
        return None

    # Normalize so H[2,2] = 1
    H = H / H[2, 2]

    # Extract components
    # For a homography H = [[h00, h01, h02], [h10, h11, h12], [h20, h21, 1]]

    # Affine part (upper-left 2x2)
    A = H[:2, :2]

    # Translation
    t = H[:2, 2]

    # Perspective (bottom row)
    p = H[2, :2]

    # Decompose affine part using SVD
    U, S, Vt = np.linalg.svd(A)

    # Scale factors
    scale_x, scale_y = S

    # Rotation angle (from U and Vt)
    R = U @ Vt
    rotation_angle = np.arctan2(R[1, 0], R[0, 0]) * 180 / np.pi

    # Check for reflection
    det = np.linalg.det(A)
    has_reflection = det < 0

    # Shear
    shear = A[0, 1] / A[0, 0] if abs(A[0, 0]) > 1e-6 else 0

    return {
        'scale_x': scale_x,
        'scale_y': scale_y,
        'rotation_deg': rotation_angle,
        'translation': t,
        'perspective': p,
        'determinant': det,
        'has_reflection': has_reflection,
        'shear': shear,
        'is_near_identity': np.allclose(H, np.eye(3), atol=0.1),
        'is_pure_scale': np.allclose(p, 0, atol=1e-4) and abs(rotation_angle) < 1,
        'is_affine': np.allclose(p, 0, atol=1e-4),
        'has_perspective': not np.allclose(p, 0, atol=1e-4)
    }


def main():
    align_to = Path('align_to')
    align_from = Path('align_from')

    pairs = find_matching_pairs(align_to, align_from)
    print(f"Found {len(pairs)} image pairs\n")
    print("=" * 80)

    all_homographies = []

    for ref_path, src_path in pairs:
        print(f"\n{src_path.name} -> {ref_path.name}")
        print("-" * 60)

        # Load images
        target_img = load_image(ref_path)
        source_img = load_image(src_path)

        target_h, target_w = target_img.shape[:2]
        source_resized = resize_to_target(source_img, (target_w, target_h))

        # Extract and match features
        kp_src, desc_src = extract_features(source_resized)
        kp_tgt, desc_tgt = extract_features(target_img)
        matches = match_features(desc_src, desc_tgt)

        print(f"  Matches: {len(matches)}")

        if len(matches) >= 4:
            H, mask = compute_homography(kp_src, kp_tgt, matches)

            if H is not None:
                inliers = np.sum(mask) if mask is not None else 0
                print(f"  Inliers: {inliers}/{len(matches)} ({100*inliers/len(matches):.1f}%)")

                print(f"\n  Homography matrix:")
                print(f"    [{H[0,0]:8.4f}  {H[0,1]:8.4f}  {H[0,2]:8.2f}]")
                print(f"    [{H[1,0]:8.4f}  {H[1,1]:8.4f}  {H[1,2]:8.2f}]")
                print(f"    [{H[2,0]:8.6f}  {H[2,1]:8.6f}  {H[2,2]:8.4f}]")

                # Analyze
                analysis = decompose_homography(H)

                print(f"\n  Analysis:")
                print(f"    Scale X: {analysis['scale_x']:.4f}")
                print(f"    Scale Y: {analysis['scale_y']:.4f}")
                print(f"    Rotation: {analysis['rotation_deg']:.2f}°")
                print(f"    Translation: ({analysis['translation'][0]:.1f}, {analysis['translation'][1]:.1f}) px")
                print(f"    Perspective: ({analysis['perspective'][0]:.6f}, {analysis['perspective'][1]:.6f})")
                print(f"    Determinant: {analysis['determinant']:.4f}")

                if analysis['is_near_identity']:
                    print(f"    Type: NEAR IDENTITY (images already aligned)")
                elif analysis['is_affine']:
                    if analysis['is_pure_scale']:
                        print(f"    Type: PURE SCALE + TRANSLATION")
                    else:
                        print(f"    Type: AFFINE (scale + rotation + translation)")
                else:
                    print(f"    Type: FULL PERSPECTIVE")

                all_homographies.append({
                    'pair': f"{src_path.name} -> {ref_path.name}",
                    'H': H,
                    'analysis': analysis
                })
            else:
                print("  Could not compute homography")
        else:
            print("  Not enough matches")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if all_homographies:
        scales_x = [h['analysis']['scale_x'] for h in all_homographies]
        scales_y = [h['analysis']['scale_y'] for h in all_homographies]
        rotations = [h['analysis']['rotation_deg'] for h in all_homographies]

        print(f"\nScale X: min={min(scales_x):.4f}, max={max(scales_x):.4f}, mean={np.mean(scales_x):.4f}")
        print(f"Scale Y: min={min(scales_y):.4f}, max={max(scales_y):.4f}, mean={np.mean(scales_y):.4f}")
        print(f"Rotation: min={min(rotations):.2f}°, max={max(rotations):.2f}°, mean={np.mean(rotations):.2f}°")

        affine_count = sum(1 for h in all_homographies if h['analysis']['is_affine'])
        perspective_count = sum(1 for h in all_homographies if h['analysis']['has_perspective'])

        print(f"\nAffine transforms: {affine_count}/{len(all_homographies)}")
        print(f"Perspective transforms: {perspective_count}/{len(all_homographies)}")


if __name__ == '__main__':
    main()
