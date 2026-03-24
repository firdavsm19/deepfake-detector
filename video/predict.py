"""
predict.py — Run inference on a single video, a folder, or a list of paths.

Usage:
    # Single video
    python predict.py --input path/to/video.mp4 --checkpoint checkpoints/best_model.pth

    # Folder of videos
    python predict.py --input path/to/folder/ --checkpoint checkpoints/best_model.pth

    # Save annotated video with per-frame scores
    python predict.py --input video.mp4 --save-video

    # Sliding-window for long videos
    python predict.py --input video.mp4 --sliding-window
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from config import cfg
from dataset import FaceExtractor, _build_eval_transform, _build_tta_transforms, sample_frame_indices
from model import DeepfakeDetector
from train import resolve_device


# ──────────────────────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────────────────────

def load_model(checkpoint_path: Path, device: torch.device) -> DeepfakeDetector:
    model = DeepfakeDetector(cfg.model).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"✓ Model loaded from {checkpoint_path}")
    return model


# ──────────────────────────────────────────────────────────────
# Frame Extraction
# ──────────────────────────────────────────────────────────────

def extract_frames_from_video(
    video_path: Path,
    num_frames: int,
    face_extractor: Optional[FaceExtractor],
    mode: str = "uniform",
    window_start: Optional[int] = None,
    stride: int = 1,
) -> Tuple[List[np.ndarray], int]:
    """
    Returns (list of RGB face/frame arrays, total_frame_count).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_frame_indices(
        total_frames=total,
        num_frames=num_frames,
        mode=mode,
        window_start=window_start,
        stride=stride,
    )

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frames.append(frames[-1] if frames else np.zeros(
                (cfg.data.face_size, cfg.data.face_size, 3), dtype=np.uint8
            ))
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if face_extractor is not None:
            frame_rgb = face_extractor.extract(frame_rgb)
        else:
            frame_rgb = cv2.resize(frame_rgb, (cfg.data.face_size, cfg.data.face_size))
        frames.append(frame_rgb)

    cap.release()
    return frames, total


def frames_to_tensor(
    frames: List[np.ndarray],
    transform,
    num_frames: int,
) -> torch.Tensor:
    """Stack augmented frames into (1, T, C, H, W)."""
    while len(frames) < num_frames:
        frames.append(frames[-1])
    frames = frames[:num_frames]

    tensors = [transform(image=f)["image"] for f in frames]
    return torch.stack(tensors).unsqueeze(0)   # (1, T, C, H, W)


# ──────────────────────────────────────────────────────────────
# Single-video Prediction
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_single(
    video_path: Path,
    model: nn.Module,
    face_extractor: Optional[FaceExtractor],
    device: torch.device,
    use_tta: bool = True,
    use_sliding_window: bool = False,
    threshold: float = 0.5,
) -> dict:
    """
    Predict whether a single video is fake or real.

    Returns a dict with:
        fake_prob, prediction, confidence, window_probs (if sliding window)
    """
    t0 = time.time()
    transforms = _build_tta_transforms(cfg.data) if use_tta else [_build_eval_transform(cfg.data)]
    num_frames = cfg.data.num_frames

    if use_sliding_window:
        # Sliding window — aggregate clip probabilities
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        starts = range(0, max(1, total - num_frames), num_frames // 2)
        window_probs = []

        for start in starts:
            frames, _ = extract_frames_from_video(
                video_path, num_frames, face_extractor,
                mode="window", window_start=start
            )
            clip_probs = []
            for transform in transforms:
                x = frames_to_tensor(frames, transform, num_frames).to(device)
                with autocast(enabled=cfg.train.amp and device.type == "cuda"):
                    logits, _ = model(x)
                clip_probs.append(torch.sigmoid(logits).item())
            window_probs.append(float(np.mean(clip_probs)))

        fake_prob = float(np.mean(window_probs))
    else:
        frames, _ = extract_frames_from_video(video_path, num_frames, face_extractor)
        clip_probs = []
        for transform in transforms:
            x = frames_to_tensor(frames, transform, num_frames).to(device)
            with autocast(enabled=cfg.train.amp and device.type == "cuda"):
                logits, _ = model(x)
            clip_probs.append(torch.sigmoid(logits).item())
        fake_prob = float(np.mean(clip_probs))
        window_probs = [fake_prob]

    pred = "FAKE" if fake_prob >= threshold else "REAL"
    confidence = fake_prob if pred == "FAKE" else (1.0 - fake_prob)
    elapsed = time.time() - t0

    return {
        "path": str(video_path),
        "fake_prob": round(fake_prob, 4),
        "prediction": pred,
        "confidence": round(confidence, 4),
        "threshold": threshold,
        "window_probs": [round(p, 4) for p in window_probs],
        "inference_time_s": round(elapsed, 2),
    }


# ──────────────────────────────────────────────────────────────
# Annotated Video Writer
# ──────────────────────────────────────────────────────────────

def save_annotated_video(
    video_path: Path,
    result: dict,
    output_path: Path,
    face_extractor: Optional[FaceExtractor],
    fps: int = 10,
):
    """
    Write a copy of the video with a per-frame overlay showing
    the fake probability bar and final verdict.
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))

    fake_prob   = result["fake_prob"]
    verdict     = result["prediction"]
    color       = (0, 0, 220) if verdict == "FAKE" else (0, 180, 0)  # BGR

    for frame_idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        # Overlay banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (W, 55), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Probability bar
        bar_w = int(W * fake_prob)
        cv2.rectangle(frame, (0, 0), (bar_w, 10), color, -1)

        # Text
        label = f"{verdict}  |  fake_prob: {fake_prob:.3f}"
        cv2.putText(frame, label, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Annotated video saved → {output_path}")


# ──────────────────────────────────────────────────────────────
# Batch Prediction
# ──────────────────────────────────────────────────────────────

def predict_folder(
    folder: Path,
    model: nn.Module,
    face_extractor: Optional[FaceExtractor],
    device: torch.device,
    use_tta: bool = True,
    use_sliding_window: bool = False,
    threshold: float = 0.5,
    save_csv: Optional[Path] = None,
) -> List[dict]:
    exts = cfg.data.video_extensions
    videos = [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in exts]

    if not videos:
        print(f"No videos found in {folder}")
        return []

    print(f"\nFound {len(videos)} videos in {folder}")
    results = []

    for i, vid in enumerate(videos, 1):
        try:
            result = predict_single(
                vid, model, face_extractor, device,
                use_tta=use_tta,
                use_sliding_window=use_sliding_window,
                threshold=threshold,
            )
        except Exception as e:
            print(f"  [{i}/{len(videos)}] ERROR {vid.name}: {e}")
            result = {
                "path": str(vid), "fake_prob": -1.0,
                "prediction": "ERROR", "confidence": 0.0,
                "error": str(e),
            }

        results.append(result)
        icon = "🔴" if result["prediction"] == "FAKE" else "🟢"
        print(
            f"  [{i}/{len(videos)}] {icon} {result['prediction']}  "
            f"p={result.get('fake_prob', '?'):.3f}  "
            f"{vid.name}  ({result.get('inference_time_s', '?')}s)"
        )

    # Summary
    valid = [r for r in results if r["prediction"] in ("FAKE", "REAL")]
    n_fake = sum(1 for r in valid if r["prediction"] == "FAKE")
    print(f"\n  Summary: {n_fake}/{len(valid)} classified as FAKE")

    # Save CSV
    if save_csv:
        with open(save_csv, "w", newline="") as f:
            fieldnames = ["path", "fake_prob", "prediction", "confidence",
                          "threshold", "inference_time_s"]
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved → {save_csv}")

    return results


# ──────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Deepfake Detector — Inference")
    parser.add_argument("--input",      type=str, required=True, help="Video file or folder")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--no-tta",     action="store_true")
    parser.add_argument("--sliding-window", action="store_true")
    parser.add_argument("--save-video", action="store_true", help="Save annotated output video")
    parser.add_argument("--output-dir", type=str, default="results/predictions")
    args = parser.parse_args()

    device = resolve_device(cfg.train.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint) if args.checkpoint else cfg.paths.checkpoint_dir / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = load_model(ckpt_path, device)

    face_extractor = FaceExtractor(
        face_size=cfg.data.face_size,
        margin=cfg.data.face_margin,
        device=str(device),
    ) if cfg.data.use_face_detection else None

    # ── Single video or folder ────────────────────────────────
    input_path = Path(args.input)

    if input_path.is_dir():
        results = predict_folder(
            folder=input_path,
            model=model,
            face_extractor=face_extractor,
            device=device,
            use_tta=not args.no_tta,
            use_sliding_window=args.sliding_window,
            threshold=args.threshold,
            save_csv=output_dir / "predictions.csv",
        )
        with open(output_dir / "predictions.json", "w") as f:
            json.dump(results, f, indent=2)

    elif input_path.is_file():
        result = predict_single(
            video_path=input_path,
            model=model,
            face_extractor=face_extractor,
            device=device,
            use_tta=not args.no_tta,
            use_sliding_window=args.sliding_window,
            threshold=args.threshold,
        )

        # Pretty print
        print("\n" + "=" * 50)
        print("  PREDICTION")
        print("=" * 50)
        for k, v in result.items():
            print(f"  {k:<22} {v}")
        print("=" * 50)

        # Save result JSON
        out_json = output_dir / f"{input_path.stem}_result.json"
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResult saved → {out_json}")

        # Annotated video
        if args.save_video:
            out_vid = output_dir / f"{input_path.stem}_annotated.mp4"
            save_annotated_video(input_path, result, out_vid, face_extractor)

    else:
        raise ValueError(f"Input path does not exist: {input_path}")


if __name__ == "__main__":
    main()