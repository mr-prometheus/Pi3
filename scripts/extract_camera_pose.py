"""
Per-second camera pose extraction from a walking-tour video using Pi3 / Pi3X.

Extracts 1fps frames from a video, runs Pi3/Pi3X in overlapping batches
(each batch is an independent, jointly-reasoned reconstruction with its own
arbitrary scale/coordinate frame), and writes per-frame camera position +
orientation + confidence to CSV for downstream walking-vs-standing-still
analysis.

Usage:
    python scripts/extract_camera_pose.py <video.mp4> <output_dir> [options]

Run `python scripts/extract_camera_pose.py --help` for all options. Key ones:
    --model {pi3,pi3x}            default: pi3x (recommended by upstream README)
    --batch-size N                requested frames/batch (default 100, ~100s)
    --overlap N                   seconds of overlap between batches (default 12)
    --calibration-walk-range S-E  known-walking window (sec) for auto PASS/FAIL
    --calibration-still-range S-E known-standing-still window (sec)
    --keep-frames                 persist extracted 1fps JPEGs under <output_dir>/frames
    --save-raw                    also persist per-frame local_points/points/conf
                                   as .npz under <output_dir>/raw/<full|calibration>/
                                   (large; off by default)

Output layout (see task spec): manifest.json, camera_pose.csv,
camera_pose_batches.csv, logs/run.log, calibration/, frames/ (optional).
"""
import argparse
import csv
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pi3.utils.geometry import depth_normal_edge

_TO_TENSOR = transforms.ToTensor()


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def now_iso():
    return datetime.now(timezone.utc).isoformat()


def write_manifest(path, data):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def get_repo_commit():
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return commit
    except Exception:
        return 'unknown'


def get_hf_model_size_mb(repo_id):
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
        for repo in info.repos:
            if repo.repo_id == repo_id:
                return round(repo.size_on_disk / (1024 ** 2), 1)
    except Exception:
        pass
    return None


def setup_logger(log_path):
    logger = logging.getLogger('pi3_pose')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(sh)
    return logger


def parse_range(s):
    if not s:
        return None
    parts = s.split('-')
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Range must be START-END, got: {s}")
    return int(parts[0]), int(parts[1])


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def rotation_matrix_to_quaternion(R):
    """3x3 rotation matrix -> unit quaternion (w, x, y, z). Shepperd's method."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        S = math.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z], dtype=np.float64)
    n = np.linalg.norm(q)
    return q / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Video probing / frame extraction
# ---------------------------------------------------------------------------

def probe_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise OSError(f"Cannot open video file: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not fps or fps <= 0:
        raise ValueError(f"Could not determine FPS for {path}")
    return dict(fps=fps, frame_count=frame_count, width=width, height=height,
                duration_sec=frame_count / fps)


class FrameExtractor:
    """
    Decodes a video once and keeps exactly one frame per integer second:
    the first decoded frame whose nominal timestamp (frame_index / fps) is
    >= the target second. Uses cap.grab() (no decode) for skipped frames and
    only cap.retrieve() (decode) for frames we actually keep, so scanning
    past a long lead-in is cheap.

    Assumes constant frame rate (CFR). A variable-frame-rate source would
    make frame_index / fps drift from wall-clock time; this is not detected.
    """

    def __init__(self, video_path, frames_dir, pixel_limit=255_000):
        self.video_path = video_path
        self.frames_dir = frames_dir
        self.pixel_limit = pixel_limit
        os.makedirs(frames_dir, exist_ok=True)

    def extract(self, start_sec=0, end_sec=None, logger=None):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise OSError(f"Cannot open video file: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            cap.release()
            raise ValueError(f"Could not determine FPS for {self.video_path}")

        target_w = target_h = None
        time_secs = []
        frame_idx = 0
        target_sec = start_sec

        while True:
            if end_sec is not None and target_sec > end_sec:
                break
            ok = cap.grab()
            if not ok:
                break
            t = frame_idx / fps
            frame_idx += 1
            if t < target_sec:
                continue

            ok2, frame = cap.retrieve()
            if not ok2:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)

            if target_w is None:
                target_w, target_h = self._compute_target_size(img.width, img.height)
                if logger:
                    logger.info(
                        f"Frame size: source {img.width}x{img.height} -> "
                        f"resized {target_w}x{target_h} (pixel_limit={self.pixel_limit})"
                    )

            img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            img.save(self.frame_path(target_sec), quality=95)
            time_secs.append(target_sec)
            target_sec += 1

        cap.release()
        duration_sec = (frame_idx - 1) / fps if frame_idx > 0 else 0.0
        size = (target_w, target_h) if target_w is not None else (0, 0)
        return time_secs, fps, duration_sec, size

    def _compute_target_size(self, w, h):
        scale = math.sqrt(self.pixel_limit / (w * h)) if w * h > 0 else 1.0
        w_t, h_t = w * scale, h * scale
        k, m = round(w_t / 14), round(h_t / 14)
        while (k * 14) * (m * 14) > self.pixel_limit and k > 1 and m > 1:
            if k / m > w_t / h_t:
                k -= 1
            else:
                m -= 1
        return max(1, k) * 14, max(1, m) * 14

    def frame_path(self, time_sec):
        return os.path.join(self.frames_dir, f"frame_{time_sec:06d}.jpg")


def load_batch_tensor(frame_paths):
    tensors = [_TO_TENSOR(Image.open(p).convert('RGB')) for p in frame_paths]
    return torch.stack(tensors, dim=0)


# ---------------------------------------------------------------------------
# Batch planning
# ---------------------------------------------------------------------------

def compute_batches(n_frames, batch_size, overlap):
    """
    Plan (start, end) [exclusive] index ranges over [0, n_frames) with the
    requested overlap between consecutive batches. If the tail remaining
    after the last full batch would contribute no new (non-overlap) frames,
    it is merged into the previous batch instead of created as its own
    degenerate batch.
    """
    if n_frames <= 0:
        return []
    batch_size = max(1, min(batch_size, n_frames))
    overlap = max(0, min(overlap, batch_size - 1))

    batches = []
    start = 0
    while start < n_frames:
        end = min(start + batch_size, n_frames)
        batches.append([start, end])
        if end >= n_frames:
            break
        next_start = end - overlap
        if n_frames - next_start <= overlap:
            batches[-1][1] = n_frames
            break
        start = next_start
    return [tuple(b) for b in batches]


# ---------------------------------------------------------------------------
# Model loading / capacity probing / inference
# ---------------------------------------------------------------------------

def load_model(model_name, ckpt_path, device):
    if model_name == 'pi3x':
        from pi3.models.pi3x import Pi3X
        if ckpt_path:
            model = Pi3X(use_multimodal=False).eval()
            weight = _load_weight(ckpt_path, device)
            model.load_state_dict(weight, strict=False)
        else:
            model = Pi3X.from_pretrained("yyfz233/Pi3X").eval()
            model.disable_multimodal()
    elif model_name == 'pi3':
        from pi3.models.pi3 import Pi3
        if ckpt_path:
            model = Pi3().eval()
            weight = _load_weight(ckpt_path, device)
            model.load_state_dict(weight)
        else:
            model = Pi3.from_pretrained("yyfz233/Pi3").eval()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device).eval()


def _load_weight(ckpt_path, device):
    if ckpt_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        return load_file(ckpt_path)
    return torch.load(ckpt_path, map_location=device, weights_only=False)


def select_dtype(device):
    if device.type == 'cuda' and torch.cuda.get_device_capability(device)[0] >= 8:
        return torch.bfloat16
    if device.type == 'cuda':
        return torch.float16
    return torch.float32


def _forward(model, imgs_5d, device, dtype):
    with torch.no_grad():
        if device.type == 'cuda':
            with torch.amp.autocast('cuda', dtype=dtype):
                return model(imgs_5d)
        return model(imgs_5d)


def probe_max_batch(model, target_n, H, W, device, dtype, logger, min_n=1):
    """
    Exponential-then-binary search (on synthetic random frames -- content
    doesn't affect memory use) for the largest N <= target_n that fits in
    GPU memory at this resolution. CUDA-only; caller should skip this on
    CPU/MPS.
    """
    def try_n(n):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            imgs = torch.rand(1, n, 3, H, W, device=device, dtype=torch.float32)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = _forward(model, imgs, device, dtype)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            del imgs, out
            torch.cuda.empty_cache()
            return True, peak, elapsed
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return False, -1, -1
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                return False, -1, -1
            raise

    n = min_n
    best, best_peak, best_elapsed = 0, -1, -1
    logger.info(f"Capacity probe: searching for max batch size <= {target_n} at {W}x{H}...")
    while n <= target_n:
        ok, peak, elapsed = try_n(n)
        if ok:
            best, best_peak, best_elapsed = n, peak, elapsed
            logger.info(f"  N={n}: OK ({peak:.0f} MB, {elapsed:.2f}s)")
            if n == target_n:
                break
            n = min(n * 2, target_n)
        else:
            logger.info(f"  N={n}: OOM")
            break

    if 0 < best < n:
        lo, hi = best, n
        while hi - lo > 1:
            mid = (lo + hi) // 2
            ok, peak, elapsed = try_n(mid)
            logger.info(f"  N={mid}: {'OK' if ok else 'OOM'}" + (f" ({peak:.0f} MB)" if ok else ""))
            if ok:
                lo, best_peak, best_elapsed = mid, peak, elapsed
            else:
                hi = mid
        best = lo

    return best, best_peak, best_elapsed


def run_batch_inference(model, imgs, device, dtype, conf_prob_thresh=0.1, edge_rtol=0.03,
                         return_raw=False):
    """
    imgs: (N, 3, H, W) float tensor in [0, 1], already on `device`.
    Returns positions (N,3), quats (N,4) [w,x,y,z], frame_confidence (N,),
    valid_pixel_frac (N,), and optionally a dict of raw per-pixel arrays.
    """
    out = _forward(model, imgs[None], device, dtype)

    camera_poses = out['camera_poses'][0].float().cpu().numpy()  # (N,4,4)
    conf_logits = out['conf'][0].float()                          # (N,H,W,1)
    local_points = out['local_points'][0].float()                 # (N,H,W,3)

    conf_prob = torch.sigmoid(conf_logits)[..., 0]                # (N,H,W)
    base_mask = conf_prob > conf_prob_thresh
    edge = depth_normal_edge(local_points[None], rtol=edge_rtol, mask=base_mask[None])[0]
    reliable_mask = base_mask & (~edge)

    conf_prob_np = conf_prob.cpu().numpy()
    reliable_mask_np = reliable_mask.cpu().numpy()

    N = camera_poses.shape[0]
    frame_confidence = np.full(N, np.nan, dtype=np.float64)
    valid_pixel_frac = np.zeros(N, dtype=np.float64)
    for i in range(N):
        m = reliable_mask_np[i]
        if m.sum() > 0:
            frame_confidence[i] = float(conf_prob_np[i][m].mean())
            valid_pixel_frac[i] = float(m.mean())
        else:
            frame_confidence[i] = float(conf_prob_np[i].mean())

    positions = camera_poses[:, :3, 3]
    quats = np.stack([rotation_matrix_to_quaternion(camera_poses[i, :3, :3]) for i in range(N)])

    raw = None
    if return_raw:
        raw = dict(
            local_points=local_points.cpu().numpy().astype(np.float16),
            points=out['points'][0].float().cpu().numpy().astype(np.float16),
            conf_prob=conf_prob_np.astype(np.float16),
        )
    return positions, quats, frame_confidence, valid_pixel_frac, raw


# ---------------------------------------------------------------------------
# Core per-range pipeline (used for both the calibration clip and full video)
# ---------------------------------------------------------------------------

def run_pipeline_range(video_path, frames_dir, pose_csv_path, batches_csv_path,
                        start_sec, end_sec, model, device, dtype, args, logger,
                        state, manifest, manifest_path, label, raw_dir=None):
    extractor = FrameExtractor(video_path, frames_dir, pixel_limit=args.pixel_limit)
    time_secs, fps, duration_sec, (W, H) = extractor.extract(
        start_sec=start_sec, end_sec=end_sec, logger=logger)
    n_frames = len(time_secs)
    if n_frames == 0:
        raise ValueError(
            f"No frames extracted for range start={start_sec} end={end_sec}; "
            f"check the video length and --calibration-start/--calibration-duration."
        )

    if not state['probed']:
        if device.type == 'cuda':
            n, peak, elapsed = probe_max_batch(model, args.batch_size, H, W, device, dtype, logger)
            if n <= 0:
                raise RuntimeError(
                    f"Model does not fit in GPU memory even at batch size 1 for "
                    f"resolution {W}x{H}. Try a smaller --pixel-limit."
                )
            if n < args.batch_size:
                msg = (f"Capacity probe at {W}x{H}: requested --batch-size {args.batch_size} "
                       f"does not fit; using {n} instead (peak {peak:.0f} MB observed).")
                logger.warning(msg)
            else:
                msg = (f"Capacity probe at {W}x{H}: requested --batch-size {args.batch_size} "
                       f"fits (peak {peak:.0f} MB observed).")
                logger.info(msg)
            manifest['notes'].append(msg)
            state['batch_size'] = min(n, args.batch_size)
        else:
            msg = (f"Running on device type '{device.type}'; GPU memory capacity was not "
                   f"probed. If you hit an out-of-memory/system-memory error, rerun with "
                   f"a smaller --batch-size.")
            logger.warning(msg)
            manifest['notes'].append(msg)
        state['probed'] = True
        write_manifest(manifest_path, manifest)

    batch_size = state['batch_size']
    batches = compute_batches(n_frames, batch_size, args.overlap)
    logger.info(f"[{label}] {n_frames} frames -> {len(batches)} batches "
                f"(batch_size={batch_size}, overlap={args.overlap}s)")

    pose_f = open(pose_csv_path, 'w', newline='')
    pose_writer = csv.DictWriter(pose_f, fieldnames=[
        'time_sec', 'batch_id', 'batch_frame_idx',
        'pos_x', 'pos_y', 'pos_z',
        'quat_w', 'quat_x', 'quat_y', 'quat_z',
        'confidence', 'valid',
    ])
    pose_writer.writeheader()

    batches_f = batches_writer = None
    if batches_csv_path:
        batches_f = open(batches_csv_path, 'w', newline='')
        batches_writer = csv.DictWriter(batches_f, fieldnames=[
            'batch_id', 'start_time_sec', 'end_time_sec', 'n_frames', 'overlap_prev_sec',
        ])
        batches_writer.writeheader()

    if raw_dir:
        os.makedirs(raw_dir, exist_ok=True)

    try:
        for batch_id, (s, e) in enumerate(batches):
            t0 = time.time()
            frame_paths = [extractor.frame_path(time_secs[k]) for k in range(s, e)]
            overlap_prev = 0 if batch_id == 0 else batches[batch_id - 1][1] - s

            error = None
            raw = None
            try:
                imgs = load_batch_tensor(frame_paths).to(device)
                positions, quats, conf, valid_frac, raw = run_batch_inference(
                    model, imgs, device, dtype, return_raw=bool(raw_dir))
            except Exception as ex:
                logger.exception(f"[{label}] batch {batch_id} inference failed")
                error = str(ex)
                nb = e - s
                positions = np.full((nb, 3), np.nan)
                quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (nb, 1))
                conf = np.full(nb, np.nan)
                valid_frac = np.zeros(nb)
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            canonical_start = s if batch_id == 0 else s + overlap_prev
            for k in range(canonical_start, e):
                local_idx = k - s
                t_sec = time_secs[k]
                c = conf[local_idx]
                is_valid = (
                    error is None
                    and np.isfinite(c)
                    and c >= args.conf_threshold
                    and valid_frac[local_idx] >= args.min_valid_pixel_frac
                )
                pose_writer.writerow(dict(
                    time_sec=t_sec, batch_id=batch_id, batch_frame_idx=local_idx,
                    pos_x=f"{positions[local_idx, 0]:.6f}",
                    pos_y=f"{positions[local_idx, 1]:.6f}",
                    pos_z=f"{positions[local_idx, 2]:.6f}",
                    quat_w=f"{quats[local_idx, 0]:.6f}",
                    quat_x=f"{quats[local_idx, 1]:.6f}",
                    quat_y=f"{quats[local_idx, 2]:.6f}",
                    quat_z=f"{quats[local_idx, 3]:.6f}",
                    confidence=("" if not np.isfinite(c) else f"{c:.6f}"),
                    valid=is_valid,
                ))
                if raw_dir and raw is not None:
                    np.savez_compressed(
                        os.path.join(raw_dir, f"batch{batch_id:03d}_frame_{t_sec:06d}.npz"),
                        local_points=raw['local_points'][local_idx],
                        points=raw['points'][local_idx],
                        conf_prob=raw['conf_prob'][local_idx],
                    )
            pose_f.flush()

            if batches_writer:
                batches_writer.writerow(dict(
                    batch_id=batch_id, start_time_sec=time_secs[s], end_time_sec=time_secs[e - 1],
                    n_frames=e - s, overlap_prev_sec=overlap_prev,
                ))
                batches_f.flush()

            elapsed = time.time() - t0
            mean_conf = float(np.nanmean(conf)) if np.isfinite(conf).any() else float('nan')
            gpu_peak = ""
            if device.type == 'cuda':
                gpu_peak = f" gpu_peak_mb={torch.cuda.max_memory_allocated(device) / (1024 ** 2):.0f}"
            logger.info(
                f"[{label}] batch={batch_id} range=[{time_secs[s]}-{time_secs[e - 1]}]s "
                f"n={e - s} overlap_prev={overlap_prev}s elapsed={elapsed:.1f}s "
                f"mean_conf={mean_conf:.3f}{gpu_peak}" + (f" ERROR={error}" if error else "")
            )
    finally:
        pose_f.close()
        if batches_f:
            batches_f.close()

    return len(batches), dict(n_frames=n_frames, fps=fps, duration_sec=duration_sec, size=(W, H))


# ---------------------------------------------------------------------------
# Calibration analysis / report
# ---------------------------------------------------------------------------

def summarize_batches_from_pose_csv(path):
    groups = defaultdict(list)
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            groups[int(r['batch_id'])].append(r)
    out = []
    for bid in sorted(groups):
        rows = groups[bid]
        times = [int(r['time_sec']) for r in rows]
        confs = [float(r['confidence']) for r in rows if r['confidence'] not in ('', None)]
        valids = [r['valid'] == 'True' for r in rows]
        out.append(dict(
            batch_id=bid, start=min(times), end=max(times), n=len(rows),
            mean_conf=(sum(confs) / len(confs) if confs else float('nan')),
            valid_frac=(sum(valids) / len(valids) if valids else 0.0),
        ))
    return out


def analyze_calibration(pose_csv_path, walk_range, still_range):
    rows = []
    with open(pose_csv_path, newline='') as f:
        rows = list(csv.DictReader(f))

    by_batch = defaultdict(list)
    for r in rows:
        by_batch[int(r['batch_id'])].append(r)

    displacement_by_time = {}
    for bid, brows in by_batch.items():
        brows.sort(key=lambda r: int(r['batch_frame_idx']))
        for i in range(1, len(brows)):
            a, b = brows[i - 1], brows[i]
            if a['valid'] != 'True' or b['valid'] != 'True':
                continue
            pa = np.array([float(a['pos_x']), float(a['pos_y']), float(a['pos_z'])])
            pb = np.array([float(b['pos_x']), float(b['pos_y']), float(b['pos_z'])])
            displacement_by_time[int(b['time_sec'])] = float(np.linalg.norm(pb - pa))

    def range_stats(rng):
        if rng is None:
            return None
        s, e = rng
        vals = [displacement_by_time[t] for t in range(s, e + 1) if t in displacement_by_time]
        if not vals:
            return None
        return dict(n=len(vals), mean=float(np.mean(vals)), median=float(np.median(vals)),
                    max=float(np.max(vals)))

    walk_stats = range_stats(walk_range)
    still_stats = range_stats(still_range)

    verdict, reason = 'INCONCLUSIVE', (
        "No --calibration-walk-range/--calibration-still-range provided (or no valid "
        "frames within them), so an automatic pass/fail verdict could not be computed. "
        "Inspect calibration_pose.csv manually: within each batch_id, walking seconds "
        "should show clear, sustained frame-to-frame displacement, while standing-still "
        "seconds should stay clustered near a fixed point."
    )
    if walk_stats and still_stats:
        ratio = walk_stats['mean'] / (still_stats['mean'] + 1e-9)
        if ratio >= 3.0:
            verdict = 'PASS'
            reason = (f"Walking-range mean per-second displacement ({walk_stats['mean']:.4g}) "
                       f"is {ratio:.1f}x the still-range mean ({still_stats['mean']:.4g}); the "
                       f"trajectory shows clear, sustained movement while walking and stays "
                       f"clustered while standing still.")
        else:
            verdict = 'FAIL'
            reason = (f"Walking-range mean displacement ({walk_stats['mean']:.4g}) is only "
                       f"{ratio:.1f}x the still-range mean ({still_stats['mean']:.4g}); expected "
                       f"a clear separation (>= 3x). The model may not be resolving the "
                       f"rotation/translation ambiguity as intended for this clip.")

    return dict(displacement_by_time=displacement_by_time, walk_stats=walk_stats,
                still_stats=still_stats, verdict=verdict, reason=reason)


def write_calibration_report(path, args, meta, analysis, batch_summary):
    lines = []
    lines.append("# Pi3 Camera Pose Calibration Report")
    lines.append("")
    lines.append(f"- Video: `{args.video_path}`")
    lines.append(f"- Clip checked: {args.calibration_start}s to "
                 f"{args.calibration_start + args.calibration_duration - 1}s "
                 f"({meta['n_frames']} seconds sampled at 1fps)")
    lines.append(f"- Model: {args.model} (device={args.device})")
    lines.append(f"- Frame size used: {meta['size'][0]}x{meta['size'][1]}")
    lines.append("")
    lines.append(f"## Verdict: **{analysis['verdict']}**")
    lines.append("")
    lines.append(analysis['reason'])
    lines.append("")
    if analysis['walk_stats']:
        ws = analysis['walk_stats']
        lines.append(f"- Walk range `{args.calibration_walk_range}`: n={ws['n']}s, "
                     f"mean displacement/sec={ws['mean']:.4g}, median={ws['median']:.4g}, "
                     f"max={ws['max']:.4g} (batch-local units, not meters)")
    if analysis['still_stats']:
        ss = analysis['still_stats']
        lines.append(f"- Still range `{args.calibration_still_range}`: n={ss['n']}s, "
                     f"mean displacement/sec={ss['mean']:.4g}, median={ss['median']:.4g}, "
                     f"max={ss['max']:.4g} (batch-local units, not meters)")
    lines.append("")
    lines.append("## Batches in this calibration run")
    lines.append("")
    lines.append("| batch_id | time range (s) | n_frames | mean confidence | valid fraction |")
    lines.append("|---|---|---|---|---|")
    for b in batch_summary:
        lines.append(f"| {b['batch_id']} | {b['start']}-{b['end']} | {b['n']} | "
                     f"{b['mean_conf']:.3f} | {b['valid_frac']:.2f} |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Position/scale are only meaningful *within* a single `batch_id` -- "
                 "do not compare displacement magnitudes across batches.")
    lines.append("- Re-run with `--calibration-walk-range` and `--calibration-still-range` "
                 "(seconds, cut-video time) pointing at a genuine walking segment and a "
                 "genuine standing-still-while-panning segment to get an automatic "
                 "PASS/FAIL verdict.")
    if analysis['verdict'] == 'FAIL':
        lines.append("- **Do not proceed to the full-video run** without investigating: "
                     "check `logs/run.log` for warnings, try a different clip, or inspect "
                     "`calibration_pose.csv` directly. Use `--force-full-on-calibration-fail` "
                     "to override once you've confirmed it's safe to proceed.")
    if analysis['verdict'] == 'INCONCLUSIVE':
        lines.append("- This run did not get a pass/fail verdict -- treat the full-video "
                     "run's results with caution until a real calibration clip with known "
                     "walk/stand-still segments has been checked.")

    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI / orchestration
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract per-second camera pose (position, orientation, confidence) "
                    "from a walking-tour video using Pi3/Pi3X, in overlapping batches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('video_path', help="Path to the input .mp4 video.")
    p.add_argument('output_dir', help="Directory to write manifest.json, CSVs, logs/, calibration/.")

    p.add_argument('--model', choices=['pi3', 'pi3x'], default='pi3x',
                    help="Pi3 (original) or Pi3X (recommended: smoother, more reliable confidence).")
    p.add_argument('--ckpt', default=None, help="Local checkpoint path. Default: download from HF hub.")
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    p.add_argument('--batch-size', type=int, default=100,
                    help="Requested frames per inference batch (~seconds). Reduced automatically "
                        "if it doesn't fit in GPU memory (CUDA only); the actual value used is "
                        "recorded in manifest.json.")
    p.add_argument('--overlap', type=int, default=12,
                    help="Seconds of frame overlap between consecutive batches (10-15s recommended).")
    p.add_argument('--pixel-limit', type=int, default=255_000,
                    help="Target pixel budget per frame after resize (matches Pi3's own default).")

    p.add_argument('--conf-threshold', type=float, default=0.15,
                    help="Minimum mean per-frame confidence (sigmoid prob, 0-1) to mark valid=true.")
    p.add_argument('--min-valid-pixel-frac', type=float, default=0.02,
                    help="Minimum fraction of confident/non-edge pixels required to mark valid=true.")

    p.add_argument('--keep-frames', action='store_true',
                    help="Persist extracted 1fps JPEGs under <output_dir>/frames/. Off by default.")
    p.add_argument('--save-raw', action='store_true',
                    help="Also persist per-frame local_points/points/conf as float16 .npz files "
                        "under <output_dir>/raw/<full|calibration>/batch<NNN>_frame_<time_sec>.npz. "
                        "Large; off by default.")

    p.add_argument('--skip-calibration', action='store_true',
                    help="Skip the calibration pre-flight check and go straight to the full run.")
    p.add_argument('--calibration-only', action='store_true',
                    help="Run only the calibration clip, then stop (don't process the full video).")
    p.add_argument('--calibration-start', type=int, default=0,
                    help="Start second (cut-video time) of the calibration clip.")
    p.add_argument('--calibration-duration', type=int, default=240,
                    help="Length in seconds of the calibration clip (default 4 min).")
    p.add_argument('--calibration-walk-range', default=None,
                    help="START-END seconds (cut-video time) known to be real walking, e.g. 30-90. "
                        "Enables an automatic PASS/FAIL verdict.")
    p.add_argument('--calibration-still-range', default=None,
                    help="START-END seconds known to be standing still while panning, e.g. 100-160.")
    p.add_argument('--force-full-on-calibration-fail', action='store_true',
                    help="Proceed to the full-video run even if calibration verdict is FAIL.")

    args = p.parse_args()
    if args.skip_calibration and args.calibration_only:
        p.error("--skip-calibration and --calibration-only are mutually exclusive.")
    return args


def main():
    args = parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logs_dir = os.path.join(output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    logger = setup_logger(os.path.join(logs_dir, 'run.log'))

    manifest_path = os.path.join(output_dir, 'manifest.json')
    manifest = {
        "video_path": os.path.abspath(args.video_path),
        "video_duration_sec": None,
        "model": args.model,
        "model_version_or_commit": get_repo_commit(),
        "fps_sampled": 1,
        "batch_size_frames": args.batch_size,
        "overlap_sec": args.overlap,
        "n_batches": None,
        "started_at": now_iso(),
        "finished_at": None,
        "status": "running",
        "notes": [],
    }
    write_manifest(manifest_path, manifest)

    try:
        probe = probe_video(args.video_path)
        manifest['video_duration_sec'] = round(probe['duration_sec'], 3)
        write_manifest(manifest_path, manifest)
        logger.info(f"Video probe: fps={probe['fps']:.3f} frames~={probe['frame_count']:.0f} "
                    f"duration~={probe['duration_sec']:.1f}s size={probe['width']}x{probe['height']}")

        device = torch.device(args.device)
        dtype = select_dtype(device)
        logger.info(f"Device={device} dtype={dtype}")

        logger.info(f"Loading model ({args.model})...")
        model = load_model(args.model, args.ckpt, device)
        if device.type == 'cuda':
            manifest['notes'].append(
                f"Model VRAM after load: {torch.cuda.memory_allocated(device) / (1024 ** 2):.0f} MB"
            )
        else:
            manifest['notes'].append(f"Running on device type '{device.type}' (no GPU capacity probing).")
        if args.ckpt is None:
            repo_id = "yyfz233/Pi3X" if args.model == 'pi3x' else "yyfz233/Pi3"
            size_mb = get_hf_model_size_mb(repo_id)
            if size_mb is not None:
                manifest['notes'].append(f"Downloaded model weights ({repo_id}): {size_mb} MB on disk.")
        write_manifest(manifest_path, manifest)

        state = dict(batch_size=args.batch_size, probed=False)
        calibration_ok = True

        if not args.skip_calibration:
            logger.info("=== Calibration run ===")
            cal_dir = os.path.join(output_dir, 'calibration')
            os.makedirs(cal_dir, exist_ok=True)
            cal_start = args.calibration_start
            cal_end = args.calibration_start + args.calibration_duration - 1
            cal_frames_dir = (os.path.join(output_dir, 'frames') if args.keep_frames
                               else tempfile.mkdtemp(prefix='pi3_frames_calibration_'))
            cal_raw_dir = os.path.join(output_dir, 'raw', 'calibration') if args.save_raw else None

            n_batches_cal, cal_meta = run_pipeline_range(
                video_path=args.video_path, frames_dir=cal_frames_dir,
                pose_csv_path=os.path.join(cal_dir, 'calibration_pose.csv'),
                batches_csv_path=None,
                start_sec=cal_start, end_sec=cal_end,
                model=model, device=device, dtype=dtype, args=args, logger=logger,
                state=state, manifest=manifest, manifest_path=manifest_path,
                label='calibration', raw_dir=cal_raw_dir,
            )
            if not args.keep_frames:
                shutil.rmtree(cal_frames_dir, ignore_errors=True)

            walk_range = parse_range(args.calibration_walk_range)
            still_range = parse_range(args.calibration_still_range)
            analysis = analyze_calibration(
                os.path.join(cal_dir, 'calibration_pose.csv'), walk_range, still_range)
            batch_summary = summarize_batches_from_pose_csv(
                os.path.join(cal_dir, 'calibration_pose.csv'))
            write_calibration_report(
                os.path.join(cal_dir, 'calibration_report.md'), args, cal_meta, analysis, batch_summary)

            logger.info(f"Calibration verdict: {analysis['verdict']} - {analysis['reason']}")
            manifest['notes'].append(f"Calibration verdict: {analysis['verdict']}")
            write_manifest(manifest_path, manifest)

            if analysis['verdict'] == 'FAIL' and not args.force_full_on_calibration_fail:
                calibration_ok = False
                logger.error(
                    "Calibration FAILED. Aborting before the full-video run. Pass "
                    "--force-full-on-calibration-fail to override, or --skip-calibration to bypass."
                )

        if args.calibration_only:
            manifest['status'] = 'complete'
            manifest['finished_at'] = now_iso()
            manifest['notes'].append("Ran in --calibration-only mode; full video was not processed.")
            write_manifest(manifest_path, manifest)
            logger.info("Done (calibration-only).")
            return

        if not calibration_ok:
            manifest['status'] = 'failed'
            manifest['finished_at'] = now_iso()
            manifest['notes'].append(
                "Aborted before full run because calibration FAILED "
                "(see calibration/calibration_report.md)."
            )
            write_manifest(manifest_path, manifest)
            sys.exit(2)

        logger.info("=== Full video run ===")
        frames_dir = (os.path.join(output_dir, 'frames') if args.keep_frames
                      else tempfile.mkdtemp(prefix='pi3_frames_full_'))
        raw_dir = os.path.join(output_dir, 'raw', 'full') if args.save_raw else None
        pose_csv_path = os.path.join(output_dir, 'camera_pose.csv')
        batches_csv_path = os.path.join(output_dir, 'camera_pose_batches.csv')

        n_batches, full_meta = run_pipeline_range(
            video_path=args.video_path, frames_dir=frames_dir,
            pose_csv_path=pose_csv_path, batches_csv_path=batches_csv_path,
            start_sec=0, end_sec=None,
            model=model, device=device, dtype=dtype, args=args, logger=logger,
            state=state, manifest=manifest, manifest_path=manifest_path,
            label='full', raw_dir=raw_dir,
        )
        if not args.keep_frames:
            shutil.rmtree(frames_dir, ignore_errors=True)

        manifest['n_batches'] = n_batches
        manifest['batch_size_frames'] = state['batch_size']
        manifest['video_duration_sec'] = round(full_meta['duration_sec'], 3)
        manifest['status'] = 'complete'
        manifest['finished_at'] = now_iso()
        manifest['notes'].append(
            f"Full run: {n_batches} batches, {full_meta['n_frames']} frames at 1fps, "
            f"frame size {full_meta['size'][0]}x{full_meta['size'][1]}."
        )
        write_manifest(manifest_path, manifest)
        logger.info("Done.")

    except SystemExit:
        raise
    except Exception:
        logger.exception("Run failed with an unhandled exception.")
        manifest['status'] = 'failed'
        manifest['finished_at'] = now_iso()
        manifest['notes'].append("Unhandled exception; see logs/run.log for the traceback.")
        write_manifest(manifest_path, manifest)
        raise


if __name__ == '__main__':
    main()
