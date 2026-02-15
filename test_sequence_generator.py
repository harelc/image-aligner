#!/usr/bin/env python3
"""
Test Sequence Generator for Detecting AI-Generated Interpolation.

Generates a video where each frame contains:
- A prominent frame index number in the center
- A grid of colored dots cycling at coprime (prime) frequencies
- Binary indicator dots encoding the frame number
- A progress bar at the bottom
- Timestamp text

The coprime cycling frequencies make it nearly impossible for AI video
generation models to correctly reproduce interpolated frames, since the
dot patterns cannot be linearly blended between keyframes.
"""

import argparse
import math
import os
import subprocess
import sys
import tempfile
import wave

import cv2
import numpy as np


# Prime cycle lengths for coprime dot patterns
PRIMES = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

# Colormap: a set of visually distinct, saturated colors (BGR format for OpenCV)
COLORMAP_COLORS = [
    (0, 0, 255),       # Red
    (0, 127, 255),     # Orange
    (0, 255, 255),     # Yellow
    (0, 255, 0),       # Green
    (255, 255, 0),     # Cyan
    (255, 0, 0),       # Blue
    (255, 0, 127),     # Purple
    (200, 0, 255),     # Magenta
    (128, 0, 255),     # Pink-red
    (0, 200, 128),     # Lime
    (255, 128, 0),     # Teal
    (128, 128, 255),   # Salmon
]


def sample_colormap(phase: float) -> tuple:
    """
    Sample a color from the colormap given a phase in [0, 1).
    Interpolates smoothly between the defined colormap colors.
    """
    n = len(COLORMAP_COLORS)
    scaled = phase * n
    idx = int(scaled) % n
    frac = scaled - int(scaled)
    next_idx = (idx + 1) % n

    c0 = np.array(COLORMAP_COLORS[idx], dtype=np.float64)
    c1 = np.array(COLORMAP_COLORS[next_idx], dtype=np.float64)
    color = c0 * (1.0 - frac) + c1 * frac
    return tuple(int(c) for c in color)


def build_dots(width: int, height: int, num_dots: int = 10):
    """
    Place a fixed number of coprime-cycling dots spread across the frame.

    Each dot gets a unique prime cycle length and phase offset.

    Returns:
        List of (x, y, prime_period, phase_offset) tuples.
    """
    dots = []
    margin_top = 80      # Leave room for binary indicators at the top
    margin_bottom = 80   # Leave room for progress bar at the bottom
    margin_sides = 100

    y_start = margin_top
    y_end = height - margin_bottom
    x_start = margin_sides
    x_end = width - margin_sides

    usable_w = x_end - x_start
    usable_h = y_end - y_start

    # Distribute dots in a scattered but deterministic pattern
    # Use 2 rows of 5 dots
    cols = 5
    rows = (num_dots + cols - 1) // cols

    for i in range(num_dots):
        row = i // cols
        col = i % cols
        x = x_start + int((col + 0.5) * usable_w / cols)
        y = y_start + int((row + 0.5) * usable_h / rows)

        prime_period = PRIMES[i % len(PRIMES)]
        phase_offset = (i * 17 % prime_period) / prime_period

        dots.append((x, y, prime_period, phase_offset))

    return dots


def get_binary_bits(frame_num: int, num_bits: int) -> list:
    """
    Return the binary representation of frame_num as a list of bools,
    MSB first, zero-padded to num_bits.
    """
    bits = []
    for i in range(num_bits - 1, -1, -1):
        bits.append(bool((frame_num >> i) & 1))
    return bits


def render_frame(
    frame_num: int,
    total_frames: int,
    width: int,
    height: int,
    fps: float,
    dots: list,
    dot_radius: int = 8,
) -> np.ndarray:
    """
    Render a single frame of the test sequence.
    """
    # Dark background
    img = np.full((height, width, 3), 24, dtype=np.uint8)

    # --- 1. Draw the coprime-cycling dot grid ---
    for (x, y, prime_period, phase_offset) in dots:
        phase = ((frame_num % prime_period) / prime_period + phase_offset) % 1.0
        color = sample_colormap(phase)
        cv2.circle(img, (x, y), dot_radius, color, -1, lineType=cv2.LINE_AA)

    # --- 2. Binary indicator dots at the top ---
    # Encode up to 20 bits (supports up to 1,048,575 frames)
    num_bits = max(20, math.ceil(math.log2(max(total_frames, 2))))
    bits = get_binary_bits(frame_num, num_bits)

    bit_dot_radius = 12
    bit_spacing = 40
    total_bit_width = num_bits * bit_spacing
    bit_x_start = (width - total_bit_width) // 2 + bit_spacing // 2
    bit_y = 40

    for i, bit_on in enumerate(bits):
        bx = bit_x_start + i * bit_spacing
        if bit_on:
            # Bright white for ON
            cv2.circle(img, (bx, bit_y), bit_dot_radius, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        else:
            # Dark gray outline for OFF
            cv2.circle(img, (bx, bit_y), bit_dot_radius, (80, 80, 80), 2, lineType=cv2.LINE_AA)

    # Small labels above binary dots: "MSB" on the left, "LSB" on the right
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "MSB", (bit_x_start - 20, bit_y - 20),
                font, 0.35, (120, 120, 120), 1, cv2.LINE_AA)
    cv2.putText(img, "LSB", (bit_x_start + (num_bits - 1) * bit_spacing - 12, bit_y - 20),
                font, 0.35, (120, 120, 120), 1, cv2.LINE_AA)

    # --- 3. Frame number displayed large and centered ---
    frame_text = f"{frame_num}"
    # Use a large font scale
    font_scale = 4.0
    thickness = 8

    (text_w, text_h), baseline = cv2.getTextSize(frame_text, font, font_scale, thickness)
    text_x = (width - text_w) // 2
    text_y = (height + text_h) // 2

    # Draw shadow for readability
    cv2.putText(img, frame_text, (text_x + 3, text_y + 3),
                font, font_scale, (0, 0, 0), thickness + 4, cv2.LINE_AA)
    # Draw main text
    cv2.putText(img, frame_text, (text_x, text_y),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # --- 4. Timestamp text ---
    timestamp = frame_num / fps
    time_text = f"Frame {frame_num}/{total_frames - 1}  |  {timestamp:.3f}s  |  {fps:.0f} fps"
    cv2.putText(img, time_text, (20, height - 50),
                font, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

    # --- 5. Progress bar at the bottom ---
    bar_height = 12
    bar_y = height - 30
    bar_margin = 20
    bar_width = width - 2 * bar_margin

    # Background bar (dark gray)
    cv2.rectangle(img, (bar_margin, bar_y), (bar_margin + bar_width, bar_y + bar_height),
                  (60, 60, 60), -1)

    # Filled portion
    if total_frames > 1:
        progress = frame_num / (total_frames - 1)
    else:
        progress = 1.0
    fill_width = int(bar_width * progress)
    if fill_width > 0:
        # Gradient-like color from green to red
        r = int(255 * progress)
        g = int(255 * (1.0 - progress))
        bar_color = (0, g, r)
        cv2.rectangle(img, (bar_margin, bar_y), (bar_margin + fill_width, bar_y + bar_height),
                      bar_color, -1)

    # Bar border
    cv2.rectangle(img, (bar_margin, bar_y), (bar_margin + bar_width, bar_y + bar_height),
                  (140, 140, 140), 1)

    return img


def generate_audio(total_frames: int, fps: float, sample_rate: int = 44100) -> np.ndarray:
    """
    Generate a pleasing audio pattern with coprime structure.

    Uses a pentatonic melody where each note's timing is driven by
    coprime beat divisions, layered with soft chime tones.
    The result sounds musical but has a mathematically precise pattern
    that AI interpolation cannot reproduce correctly.
    """
    duration = total_frames / fps
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    audio = np.zeros(num_samples, dtype=np.float64)

    # Pentatonic scale frequencies (C major pentatonic, octave 4-5)
    pentatonic = np.array([261.63, 293.66, 329.63, 392.00, 440.00,
                           523.25, 587.33, 659.25, 783.99, 880.00])

    # Each voice plays notes from the pentatonic scale, switching at
    # a coprime beat rate. This creates interlocking rhythmic patterns.
    beat_primes = [3, 5, 7, 11]  # beats per second for each voice

    for voice_idx, beats_per_sec in enumerate(beat_primes):
        # Which note in the scale this voice cycles through
        scale_offset = voice_idx * 3

        # Beat index at each sample
        beat_idx = (t * beats_per_sec).astype(np.int64)
        # Note selection per sample
        note_idx = (beat_idx + scale_offset) % len(pentatonic)
        freq = pentatonic[note_idx]

        # Time within current beat
        beat_time = t - beat_idx / beats_per_sec

        # Envelope: quick attack, gentle decay (chime-like)
        attack = np.minimum(beat_time / 0.01, 1.0)
        decay = np.exp(-beat_time * 6.0)
        envelope = attack * decay

        # Cumulative phase to avoid discontinuities at note boundaries
        # phase[i] = 2*pi * integral of freq from 0 to t[i]
        dt = duration / num_samples
        phase = np.cumsum(freq) * dt * 2.0 * np.pi

        # Tone with harmonics for warmth
        voice = (np.sin(phase)
                 + 0.3 * np.sin(2.0 * phase)
                 + 0.1 * np.sin(3.0 * phase))
        voice *= envelope

        amplitude = 0.2 / (voice_idx + 1)
        audio += voice * amplitude

    # Soft continuous pad — low hum modulated at coprime rate (13 Hz)
    pad_freq = 130.81  # C3
    mod_freq = 13.0
    pad = np.sin(2.0 * np.pi * pad_freq * t) * 0.08
    pad *= 0.5 + 0.5 * np.sin(2.0 * np.pi * mod_freq * t)
    audio += pad

    # Normalize to 16-bit range
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.85
    audio_int16 = (audio * 32767).astype(np.int16)

    return audio_int16


def write_wav(path: str, audio: np.ndarray, sample_rate: int = 44100):
    """Write a mono 16-bit WAV file."""
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())


def mux_audio_video(video_path: str, audio_path: str, output_path: str):
    """Combine video and audio using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Generate video test sequences for detecting AI-generated interpolation."
    )
    parser.add_argument("--width", "-W", type=int, default=1920, help="Frame width (default: 1920)")
    parser.add_argument("--height", "-H", type=int, default=1080, help="Frame height (default: 1080)")
    parser.add_argument("--fps", "-F", type=float, default=30.0, help="Frames per second (default: 30)")
    parser.add_argument("--frames", "-N", type=int, default=601, help="Total number of frames (default: 601)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file path (default: auto-generated from params)")
    args = parser.parse_args()

    width = args.width
    height = args.height
    fps = args.fps
    total_frames = args.frames
    output_path = args.output or f"test_sequence_{width}x{height}_f{fps:g}_n{total_frames}.mp4"

    print(f"Generating test sequence: {width}x{height} @ {fps} fps, {total_frames} frames")
    print(f"Output: {output_path}")

    dot_radius = 24

    # Build the 10 coprime-cycling dots
    dots = build_dots(width, height, num_dots=10)
    print(f"Total coprime dots: {len(dots)}, radius: {dot_radius}px")

    # Set up VideoWriter with H.264 codec
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        # Fall back to mp4v if avc1 is not available
        print("Warning: avc1 codec not available, trying mp4v...")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        print("Error: Could not open VideoWriter. Check your OpenCV installation.", file=sys.stderr)
        sys.exit(1)

    # Render frames
    report_interval = max(1, total_frames // 20)
    for f in range(total_frames):
        frame = render_frame(f, total_frames, width, height, fps, dots, dot_radius)
        writer.write(frame)

        if f % report_interval == 0 or f == total_frames - 1:
            pct = (f / (total_frames - 1)) * 100 if total_frames > 1 else 100
            print(f"  Frame {f}/{total_frames - 1} ({pct:.1f}%)")

    writer.release()
    print("Video frames written.")

    # --- Generate audio and mux ---
    print("Generating audio...")
    audio = generate_audio(total_frames, fps)
    print(f"Audio: {len(audio)} samples, {len(audio)/44100:.2f}s")

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "audio.wav")
        write_wav(wav_path, audio)

        # If output already exists as video-only, mux into a temp then replace
        tmp_out = os.path.join(tmpdir, "output_with_audio.mp4")
        print("Muxing audio + video with ffmpeg...")
        mux_audio_video(output_path, wav_path, tmp_out)

        # Replace original with muxed version
        os.replace(tmp_out, output_path)

    print(f"Done. Video saved to: {output_path}")


if __name__ == "__main__":
    main()
