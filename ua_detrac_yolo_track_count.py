"""
UA-DETRAC (image sequences) + YOLOv8 (COCO) + ByteTrack tracking + enter/exit counting + dwell time.

What it does:
- Downloads UA-DETRAC via kagglehub (if you want) OR uses an existing dataset path
- Auto-finds a sequence folder that contains many .jpg frames
- Detects + tracks at least 2 classes (person + vehicles) with different colors
- Produces an annotated MP4 video
- Computes:
  - enter/exit counts per class (crossing a line with direction)
  - average dwell time per class (seconds) for completed tracks

Install:
  pip install ultralytics opencv-python numpy tqdm kagglehub

Run (auto download + process first found sequence):
  python ua_detrac_yolo_track_count.py --download

Run (use existing downloaded path):
  python ua_detrac_yolo_track_count.py --dataset_path "YOUR_PATH_FROM_kagglehub" --sequence_auto

Run (choose a specific sequence folder name substring):
  python ua_detrac_yolo_track_count.py --dataset_path "..." --sequence_hint "MVI_20011"

Notes:
- Classes tracked (COCO): person(0), car(2), motorcycle(3), bus(5), truck(7)
- Counting line defaults to a horizontal line at 60% image height (you can override).
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# ---------- Optional download ----------
def maybe_download(download: bool) -> Optional[str]:
    if not download:
        return None
    import kagglehub  # noqa: F401
    import kagglehub

    path = kagglehub.dataset_download("bratjay/ua-detrac-orig")
    print("Path to dataset files:", path)
    return path


# ---------- Dataset discovery ----------
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS


def find_sequence_dirs(root: Path, min_frames: int = 200) -> List[Path]:
    """
    Find candidate sequence directories: folders that contain many image frames.
    UA-DETRAC typically has nested structure; we search recursively.
    """
    candidates = []
    for d in root.rglob("*"):
        if d.is_dir():
            # Quick check: count images (cap for speed)
            imgs = []
            try:
                for f in d.iterdir():
                    if f.is_file() and is_image_file(f):
                        imgs.append(f)
                        if len(imgs) >= min_frames:
                            break
            except PermissionError:
                continue

            if len(imgs) >= min_frames:
                candidates.append(d)

    # Prefer "img" or "images" like folders (if any), else keep all
    def score(p: Path) -> int:
        name = p.name.lower()
        s = 0
        if "img" in name:
            s += 2
        if "image" in name:
            s += 2
        if "seq" in name or "mvi" in name:
            s += 1
        return s

    candidates.sort(key=lambda p: score(p), reverse=True)
    return candidates


def pick_sequence_dir(
    dataset_path: Path,
    sequence_hint: Optional[str],
    min_frames: int,
) -> Path:
    #  first check if dataset_path itself is already a frames folder
    if dataset_path.is_dir():
        imgs_here = [f for f in dataset_path.iterdir() if f.is_file() and is_image_file(f)]
        if len(imgs_here) >= min_frames:
            return dataset_path

    if sequence_hint:
        hint = sequence_hint.lower()
        matches = [p for p in dataset_path.rglob("*") if p.is_dir() and hint in str(p).lower()]
        good = []
        for d in matches:
            imgs = [f for f in d.iterdir() if f.is_file() and is_image_file(f)]
            if len(imgs) >= min_frames:
                good.append(d)
        if good:
            good.sort(key=lambda d: len([f for f in d.iterdir() if f.is_file() and is_image_file(f)]), reverse=True)
            return good[0]

    candidates = find_sequence_dirs(dataset_path, min_frames=min_frames)
    if not candidates:
        raise FileNotFoundError(
            f"No sequence dirs found under {dataset_path} with >= {min_frames} image frames.\n"
            "Try lowering --min_frames or provide --sequence_hint."
        )
    return candidates[0]



def list_frames(seq_dir: Path) -> List[Path]:
    frames = [p for p in seq_dir.iterdir() if p.is_file() and is_image_file(p)]
    if not frames:
        raise FileNotFoundError(f"No image frames found in {seq_dir}")
    # Sort naturally by filename
    frames.sort(key=lambda p: p.name)
    return frames


# ---------- Geometry for counting line ----------
def side_of_line(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
    """
    Returns signed area (cross product) => sign indicates which side of line AB point P is on.
    """
    return float((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]))


# ---------- Main tracking + counting ----------
def run(
    dataset_path: Path,
    out_dir: Path,
    model_name: str,
    conf: float,
    iou: float,
    tracker_cfg: str,
    fps: float,
    max_lost_seconds: float,
    min_frames: int,
    sequence_hint: Optional[str],
    line_x1: Optional[int],
    line_y1: Optional[int],
    line_x2: Optional[int],
    line_y2: Optional[int],
    line_y_frac: float,
    enter_rule: str,  # "neg_to_pos" or "pos_to_neg"
    resize_width: Optional[int],
) -> None:
    from ultralytics import YOLO

    out_dir.mkdir(parents=True, exist_ok=True)

    seq_dir = pick_sequence_dir(dataset_path, sequence_hint=sequence_hint, min_frames=min_frames)
    frames = list_frames(seq_dir)

    # Read first frame to get size
    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"Failed to read first frame: {frames[0]}")
    h0, w0 = first.shape[:2]

    # Optional resize
    if resize_width and resize_width > 0 and resize_width != w0:
        scale = resize_width / float(w0)
        out_w = int(resize_width)
        out_h = int(round(h0 * scale))
    else:
        out_w, out_h = w0, h0

    # Default counting line: horizontal at y = line_y_frac * height
    if line_x1 is None or line_y1 is None or line_x2 is None or line_y2 is None:
        y = int(round(line_y_frac * out_h))
        a = np.array([0, y], dtype=np.float32)
        b = np.array([out_w - 1, y], dtype=np.float32)
    else:
        a = np.array([line_x1, line_y1], dtype=np.float32)
        b = np.array([line_x2, line_y2], dtype=np.float32)

    # Classes: person + vehicles (COCO)
    # 0 person, 2 car, 3 motorcycle, 5 bus, 7 truck
    class_ids = [0, 2, 3, 5, 7]
    class_names = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    class_group = {
        0: "person",
        2: "vehicle",
        3: "vehicle",
        5: "vehicle",
        7: "vehicle",
    }

    # Colors per group (BGR for OpenCV)
    colors = {
        "person": (0, 255, 0),    # green
        "vehicle": (255, 0, 0),   # blue
        "other": (0, 255, 255),   # yellow
    }

    model = YOLO(model_name)

    # Video writer
    out_video = out_dir / "annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_video), fourcc, fps, (out_w, out_h))

    max_lost_frames = int(round(max_lost_seconds * fps))

    # Track state
    # state[tid] = {
    #   "cls": int,
    #   "group": str,
    #   "first_frame": int,
    #   "last_frame": int,
    #   "last_seen": int,
    #   "prev_side": float,
    #   "counted_enter": bool,
    #   "counted_exit": bool,
    # }
    state: Dict[int, Dict] = {}

    enter_counts = {"person": 0, "vehicle": 0}
    exit_counts = {"person": 0, "vehicle": 0}

    dwell_times = {"person": [], "vehicle": []}  # seconds for completed tracks

    def finalize_track(tid: int, cur_frame_idx: int):
        info = state.get(tid)
        if not info:
            return
        # If it hasn't been seen for > max_lost_frames, finalize
        if cur_frame_idx - info["last_seen"] <= max_lost_frames:
            return
        duration_frames = info["last_frame"] - info["first_frame"] + 1
        duration_s = duration_frames / fps
        grp = info["group"]
        dwell_times[grp].append(duration_s)
        # Remove from active
        del state[tid]

    def avg(lst: List[float]) -> float:
        return float(sum(lst) / len(lst)) if lst else 0.0

    # For determining direction:
    # - side_of_line returns positive on one side, negative on other.
    # - define ENTER as neg->pos or pos->neg.
    enter_neg_to_pos = (enter_rule == "neg_to_pos")

    # Process frames
    for i, fp in enumerate(tqdm(frames, desc=f"Processing {seq_dir.name}", unit="frame")):
        frame = cv2.imread(str(fp))
        if frame is None:
            print(f"Warning: failed to read {fp}, skipping")
            continue

        if (out_w, out_h) != (w0, h0):
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

        # Track with persistence
        # Note: model.track accepts numpy image; persist=True keeps track IDs across frames
        results = model.track(
            source=frame,
            persist=True,
            tracker=tracker_cfg,
            conf=conf,
            iou=iou,
            classes=class_ids,
            verbose=False,
        )

        # Ultralytics returns a list; we use first
        r = results[0]
        boxes = r.boxes

        # Mark which track IDs were updated this frame
        updated_ids = set()

        if boxes is not None and boxes.id is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            tids = boxes.id.cpu().numpy().astype(int)
            clss = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), tid, cls_id, cf in zip(xyxy, tids, clss, confs):
                updated_ids.add(tid)
                grp = class_group.get(int(cls_id), "other")

                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                p = np.array([cx, cy], dtype=np.float32)

                cur_side = side_of_line(a, b, p)

                if tid not in state:
                    state[tid] = {
                        "cls": int(cls_id),
                        "group": grp,
                        "first_frame": i,
                        "last_frame": i,
                        "last_seen": i,
                        "prev_side": cur_side,
                        "counted_enter": False,
                        "counted_exit": False,
                        "last_center": (cx, cy),
                    }
                else:
                    st = state[tid]
                    st["cls"] = int(cls_id)  # update if detector flips (rare)
                    st["group"] = grp
                    st["last_frame"] = i
                    st["last_seen"] = i

                    prev_side = st["prev_side"]
                    prev_cx, prev_cy = st.get("last_center", (cx, cy))
                    st["last_center"] = (cx, cy)

                    # Check crossing only if it truly changed side and moved enough
                    moved = abs(cx - prev_cx) + abs(cy - prev_cy)
                    if (prev_side == 0) or (cur_side == 0):
                        crossed = False
                    else:
                        crossed = (prev_side > 0) != (cur_side > 0)

                    if crossed and moved > 5.0:
                        # Determine direction by sign change
                        if enter_neg_to_pos:
                            is_enter = (prev_side < 0) and (cur_side > 0)
                        else:
                            is_enter = (prev_side > 0) and (cur_side < 0)
                        is_exit = not is_enter

                        if is_enter and (not st["counted_enter"]):
                            enter_counts[grp] += 1
                            st["counted_enter"] = True
                        elif is_exit and (not st["counted_exit"]):
                            exit_counts[grp] += 1
                            st["counted_exit"] = True

                    st["prev_side"] = cur_side

                # Draw bbox
                color = colors.get(grp, colors["other"])
                x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), color, 2)

                label = f"{class_names.get(int(cls_id), str(int(cls_id)))} #{tid} {cf:.2f}"
                # Label background
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                y_text = max(0, y1i - th - 4)
                cv2.rectangle(frame, (x1i, y_text), (x1i + tw + 6, y_text + th + 6), color, -1)
                cv2.putText(frame, label, (x1i + 3, y_text + th + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Finalize tracks that have been lost
        # (we do it every frame; cheap for moderate #tracks)
        for tid in list(state.keys()):
            finalize_track(tid, i)

        # Draw counting line
        cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 0, 255), 2)

        # Overlay stats
        avg_person = avg(dwell_times["person"])
        avg_vehicle = avg(dwell_times["vehicle"])
        stats_lines = [
            f"ENTER person={enter_counts['person']} vehicle={enter_counts['vehicle']}",
            f"EXIT  person={exit_counts['person']} vehicle={exit_counts['vehicle']}",
            f"AVG DWELL (completed) person={avg_person:.1f}s vehicle={avg_vehicle:.1f}s",
            f"Active tracks: {len(state)}",
        ]
        y0 = 20
        for s in stats_lines:
            cv2.putText(frame, s, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, s, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            y0 += 24

        vw.write(frame)

    # Finalize all remaining tracks at end (treat as ended at last frame)
    last_i = len(frames) - 1
    for tid, info in list(state.items()):
        duration_frames = info["last_frame"] - info["first_frame"] + 1
        duration_s = duration_frames / fps
        dwell_times[info["group"]].append(duration_s)
        del state[tid]

    vw.release()

    # Save summary + per-track dwell values
    summary = {
        "dataset_path": str(dataset_path),
        "sequence_dir": str(seq_dir),
        "output_video": str(out_video),
        "fps_used": fps,
        "counting_line": {"x1": float(a[0]), "y1": float(a[1]), "x2": float(b[0]), "y2": float(b[1])},
        "enter_rule": enter_rule,
        "enter_counts": enter_counts,
        "exit_counts": exit_counts,
        "avg_dwell_seconds": {
            "person": (sum(dwell_times["person"]) / len(dwell_times["person"])) if dwell_times["person"] else 0.0,
            "vehicle": (sum(dwell_times["vehicle"]) / len(dwell_times["vehicle"])) if dwell_times["vehicle"] else 0.0,
        },
        "num_completed_tracks": {"person": len(dwell_times["person"]), "vehicle": len(dwell_times["vehicle"])},
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Also dump dwell samples
    with open(out_dir / "dwell_times.json", "w", encoding="utf-8") as f:
        json.dump(dwell_times, f, indent=2)

    print("\nDone.")
    print("Annotated video:", out_video)
    print("Summary:", out_dir / "summary.json")
    print("Dwell times:", out_dir / "dwell_times.json")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="Download dataset with kagglehub and use it.")
    ap.add_argument("--dataset_path", type=str, default="", help="Path printed by kagglehub.dataset_download(...)")
    ap.add_argument("--out_dir", type=str, default="outputs", help="Output directory")
    ap.add_argument("--model", type=str, default="yolov8n.pt", help="Ultralytics model name or path")
    ap.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    ap.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Ultralytics tracker config (bytetrack.yaml/botsort.yaml)")
    ap.add_argument("--fps", type=float, default=25.0, help="FPS used for dwell time + output video (UA-DETRAC often ~25)")
    ap.add_argument("--max_lost_seconds", type=float, default=1.0, help="How long an ID can disappear before finalizing dwell")
    ap.add_argument("--min_frames", type=int, default=200, help="Min #frames to consider a folder a valid sequence")
    ap.add_argument("--sequence_hint", type=str, default=None, help="Substring to locate a specific sequence folder")
    ap.add_argument("--sequence_auto", action="store_true", help="Auto-pick a sequence folder (default if no hint)")

    # Counting line
    ap.add_argument("--line_x1", type=int, default=None)
    ap.add_argument("--line_y1", type=int, default=None)
    ap.add_argument("--line_x2", type=int, default=None)
    ap.add_argument("--line_y2", type=int, default=None)
    ap.add_argument("--line_y_frac", type=float, default=0.60, help="If no explicit line given: horizontal line at this fraction of height")
    ap.add_argument("--enter_rule", type=str, default="neg_to_pos", choices=["neg_to_pos", "pos_to_neg"],
                    help="Which side-change counts as ENTER (the opposite counts as EXIT)")

    ap.add_argument("--resize_width", type=int, default=None, help="Optional resize width (keeps aspect). Helps speed.")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dl_path = maybe_download(args.download)
    dataset_path = dl_path or args.dataset_path
    if not dataset_path:
        raise SystemExit("Provide --dataset_path or use --download")

    run(
        dataset_path=Path(dataset_path),
        out_dir=Path(args.out_dir),
        model_name=args.model,
        conf=args.conf,
        iou=args.iou,
        tracker_cfg=args.tracker,
        fps=args.fps,
        max_lost_seconds=args.max_lost_seconds,
        min_frames=args.min_frames,
        sequence_hint=args.sequence_hint,
        line_x1=args.line_x1,
        line_y1=args.line_y1,
        line_x2=args.line_x2,
        line_y2=args.line_y2,
        line_y_frac=args.line_y_frac,
        enter_rule=args.enter_rule,
        resize_width=args.resize_width,
    )
