"""Turn a capture (a folder of photos or a video) into a COLMAP dataset that Scene can load.

Usage:
    python scripts/run_colmap.py data/NemeSys data/nemesys
    python scripts/run_colmap.py "data/nemesys vid.mp4" data/nemesys_vid --fps 3

Output layout (the same as 3DGS's convert.py):
    <out>/input/        the capture as JPEGs: HEIC converted, EXIF rotation applied, long side capped
                        at --max_size; for a video, the sharpest frame in each 1/fps-second window
    <out>/distorted/    COLMAP database and the raw reconstruction (OPENCV camera)
    <out>/images/       undistorted images (what train.py reads)
    <out>/sparse/0/     PINHOLE camera, poses and SfM points
    <out>/images_N/     copies downsampled by N (--downsample), for --images images_N --resolution 1
    <out>/colmap_summary.json

All images share one camera, whose intrinsics COLMAP estimates. Feature extraction and matching
run on the GPU (no CPU fallback). Matching is exhaustive up to EXHAUSTIVE_MAX images.
"""

import argparse
import glob
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np
import pillow_heif
import pycolmap
from PIL import Image, ImageOps

pillow_heif.register_heif_opener()

PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
# Sequential matching only pairs nearby frames and has no loop closure without a vocab tree, so
# one blurry stretch splits the model (96 frames of the NemeSys video: 4 pieces sequential, all
# registered exhaustive in ~90 s). Exhaustive is O(n^2) pairs; use it while that stays cheap.
EXHAUSTIVE_MAX = 300


def _resize_to(img, max_size):
    if max_size and max(img.size) > max_size:
        scale = max_size / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    return img


def prepare_photos(src_dir, input_dir, max_size):
    """Write every photo in src_dir to input_dir as an upright RGB JPEG, keeping its EXIF
    (COLMAP takes its focal length prior from it). Already converted files are skipped."""
    names = sorted(f for f in os.listdir(src_dir) if os.path.splitext(f)[1].lower() in PHOTO_EXTS)
    if not names:
        sys.exit(f"No photos ({', '.join(sorted(PHOTO_EXTS))}) in {src_dir}")
    os.makedirs(input_dir, exist_ok=True)
    for i, name in enumerate(names):
        dst = os.path.join(input_dir, os.path.splitext(name)[0] + ".jpg")
        if os.path.exists(dst):
            continue
        with Image.open(os.path.join(src_dir, name)) as img:
            exif = img.getexif()
            img = _resize_to(ImageOps.exif_transpose(img).convert("RGB"), max_size)
        exif[0x0112] = 1  # pixels are now upright
        img.save(dst, quality=95, exif=exif.tobytes())
        print(f"\r  converted {i + 1}/{len(names)}", end="", flush=True)
    print(f"\r  {len(names)} photos in {input_dir}" + " " * 20)
    return len(names)


def extract_video_frames(video, input_dir, fps, max_size):
    """Keep the sharpest frame (variance of the Laplacian) in each 1/fps-second window."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"OpenCV cannot open {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    window = max(1, round(src_fps / fps))
    os.makedirs(input_dir, exist_ok=True)
    best, best_score, n_written, idx = None, -1.0, 0, 0
    sharpness = []

    def flush():
        nonlocal best, best_score, n_written
        if best is not None:
            n_written += 1
            img = _resize_to(Image.fromarray(cv2.cvtColor(best, cv2.COLOR_BGR2RGB)), max_size)
            img.save(os.path.join(input_dir, f"frame_{n_written:05d}.jpg"), quality=95)
            sharpness.append(best_score)
        best, best_score = None, -1.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        score = cv2.Laplacian(small, cv2.CV_64F).var()
        if score > best_score:
            best, best_score = frame, score
        idx += 1
        if idx % window == 0:
            flush()
            print(f"\r  {idx} frames read, {n_written} kept", end="", flush=True)
    flush()
    cap.release()
    print(f"\r  {idx} frames at {src_fps:.1f} fps -> {n_written} kept (1 per {window} frames) in {input_dir}")
    return n_written, sharpness


def run_sfm(input_dir, distorted_dir, matcher, mapper, camera_model):
    """Features, matches and a sparse reconstruction. Returns (reconstruction, its folder)."""
    os.makedirs(distorted_dir, exist_ok=True)
    database = os.path.join(distorted_dir, "database.db")
    if os.path.exists(database):
        os.remove(database)  # stale features from an earlier run would be reused silently
    sparse_dir = os.path.join(distorted_dir, "sparse")
    shutil.rmtree(sparse_dir, ignore_errors=True)
    os.makedirs(sparse_dir)

    t = time.perf_counter()
    pycolmap.extract_features(database, input_dir,
                              camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=pycolmap.ImageReaderOptions(camera_model=camera_model),
                              device=pycolmap.Device.cuda)
    print(f"  features: {time.perf_counter() - t:.0f}s")

    t = time.perf_counter()
    if matcher == "sequential":
        pycolmap.match_sequential(database, device=pycolmap.Device.cuda)
    else:
        pycolmap.match_exhaustive(database, device=pycolmap.Device.cuda)
    print(f"  {matcher} matching: {time.perf_counter() - t:.0f}s")

    t = time.perf_counter()
    if mapper == "global":
        recs = pycolmap.global_mapping(database, input_dir, sparse_dir)
    else:
        recs = pycolmap.incremental_mapping(database, input_dir, sparse_dir)
    print(f"  {mapper} mapping: {time.perf_counter() - t:.0f}s")
    if not recs:
        sys.exit("COLMAP found no reconstruction. Too little overlap or texture between images?")
    if len(recs) > 1:
        sizes = ", ".join(str(r.num_reg_images()) for r in recs.values())
        print(f"  WARNING: {len(recs)} disconnected models (images: {sizes}); keeping the largest")
    idx, rec = max(recs.items(), key=lambda kv: kv[1].num_reg_images())
    return rec, os.path.join(sparse_dir, str(idx))


def undistort(out_dir, model_dir, input_dir):
    """Undistort into <out>/images and <out>/sparse/0 with a PINHOLE camera."""
    for d in ["images", "sparse", "stereo"] + glob.glob(os.path.join(out_dir, "images_*")):
        shutil.rmtree(os.path.join(out_dir, d), ignore_errors=True)  # images_N would be stale
    pycolmap.undistort_images(out_dir, model_dir, input_dir)
    # COLMAP writes the model straight into sparse/; the loader reads sparse/0/.
    sparse = os.path.join(out_dir, "sparse")
    os.makedirs(os.path.join(sparse, "0"))
    for f in os.listdir(sparse):
        if f != "0":
            shutil.move(os.path.join(sparse, f), os.path.join(sparse, "0", f))
    shutil.rmtree(os.path.join(out_dir, "stereo"), ignore_errors=True)  # only used by dense MVS
    for f in glob.glob(os.path.join(out_dir, "run-colmap-*.sh")):
        os.remove(f)
    return pycolmap.Reconstruction(os.path.join(sparse, "0"))


def downsample(out_dir, factors):
    names = sorted(os.listdir(os.path.join(out_dir, "images")))
    for k in factors:
        dst = os.path.join(out_dir, f"images_{k}")
        shutil.rmtree(dst, ignore_errors=True)
        os.makedirs(dst)
        for name in names:
            with Image.open(os.path.join(out_dir, "images", name)) as img:
                img.resize((round(img.width / k), round(img.height / k)), Image.LANCZOS) \
                   .save(os.path.join(dst, name), quality=95)
        print(f"  images_{k}: {len(names)} images")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture", help="folder of photos, or a video file")
    parser.add_argument("out_dir")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="video only: frames kept per second of video (sharpest in each window)")
    parser.add_argument("--max_size", type=int, default=3200,
                        help="cap the long side of the input images (0 = keep full size)")
    parser.add_argument("--matcher", choices=["auto", "exhaustive", "sequential"], default="auto",
                        help=f"auto: exhaustive up to {EXHAUSTIVE_MAX} images, sequential above")
    parser.add_argument("--mapper", choices=["incremental", "global"], default="incremental")
    parser.add_argument("--camera_model", default="OPENCV",
                        help="distorted camera model COLMAP fits (undistorted output is always PINHOLE)")
    parser.add_argument("--downsample", type=int, nargs="*", default=[2, 4],
                        help="also write images_N folders for these factors")
    parser.add_argument("--skip_input", action="store_true",
                        help="reuse <out>/input as is (e.g. after deleting bad frames by hand)")
    args = parser.parse_args(argv)

    if not pycolmap.has_cuda:
        sys.exit("This pycolmap build has no CUDA support; install pycolmap-cuda12 (see pyproject.toml)")

    is_video = os.path.isfile(args.capture) and os.path.splitext(args.capture)[1].lower() in VIDEO_EXTS
    if not is_video and not os.path.isdir(args.capture):
        sys.exit(f"{args.capture} is neither a folder nor a video ({', '.join(sorted(VIDEO_EXTS))})")
    input_dir = os.path.join(args.out_dir, "input")
    summary = {"capture": os.path.abspath(args.capture), "mapper": args.mapper,
               "camera_model": args.camera_model, "max_size": args.max_size}
    t0 = time.perf_counter()

    print("== preparing images")
    if args.skip_input:
        n_input = len(os.listdir(input_dir))
    elif is_video:
        n_input, sharpness = extract_video_frames(args.capture, input_dir, args.fps, args.max_size)
        summary.update(fps=args.fps, frame_sharpness_min=float(np.min(sharpness)),
                       frame_sharpness_median=float(np.median(sharpness)))
    else:
        n_input = prepare_photos(args.capture, input_dir, args.max_size)

    matcher = args.matcher if args.matcher != "auto" else \
        ("exhaustive" if n_input <= EXHAUSTIVE_MAX else "sequential")
    summary["matcher"] = matcher

    print("== structure from motion")
    rec, model_dir = run_sfm(input_dir, os.path.join(args.out_dir, "distorted"), matcher, args.mapper,
                             args.camera_model)
    cam = next(iter(rec.cameras.values()))
    summary.update(
        num_input_images=n_input,
        num_registered=rec.num_reg_images(),
        num_points3D=rec.num_points3D(),
        mean_reprojection_error_px=rec.compute_mean_reprojection_error(),
        mean_track_length=rec.compute_mean_track_length(),
        distorted_camera={"model": cam.model.name, "width": cam.width, "height": cam.height,
                          "params": [float(p) for p in cam.params]},
    )
    registered = {img.name for img in rec.images.values()}
    summary["unregistered_images"] = sorted(set(os.listdir(input_dir)) - registered)

    print("== undistorting")
    und = undistort(args.out_dir, model_dir, input_dir)
    ucam = next(iter(und.cameras.values()))
    summary["camera"] = {"model": ucam.model.name, "width": ucam.width, "height": ucam.height,
                         "params": [float(p) for p in ucam.params]}
    if args.downsample:
        downsample(args.out_dir, args.downsample)

    summary["seconds"] = round(time.perf_counter() - t0, 1)
    with open(os.path.join(args.out_dir, "colmap_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nRegistered {summary['num_registered']}/{n_input} images, {summary['num_points3D']} points, "
          f"mean reprojection error {summary['mean_reprojection_error_px']:.2f} px, "
          f"{summary['seconds']:.0f}s total")
    if summary["unregistered_images"]:
        print(f"Not registered ({len(summary['unregistered_images'])}): "
              + ", ".join(summary["unregistered_images"][:10])
              + (" ..." if len(summary["unregistered_images"]) > 10 else ""))
    print(f"Camera: {ucam.model.name} {ucam.width}x{ucam.height}, params {np.round(ucam.params, 1).tolist()}")
    print(f"Dataset ready: --source_path={args.out_dir}")


if __name__ == "__main__":
    main()
