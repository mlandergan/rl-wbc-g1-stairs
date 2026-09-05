"""Render a "Hiking in the Wild"-style retargeted parkour motion npz to an mp4, using MuJoCo's
offscreen renderer + ffmpeg. Companion to visualize_parkour_npz_rerun.py (same npz schema,
same FK/reindex logic) -- this script is for producing a fixed video asset (e.g. for a blog
post) rather than an interactive scrub-able session.

The npz is a concatenation of many separate mocap/MPC clips (~47 segments, detected via large
frame-to-frame base-position jumps) rather than one continuous trajectory -- use --start/--end
to render a single segment. A chase camera follows the robot's base every frame, so segment
boundaries just show up as a camera recenter, not a crash.

Usage:
    python render_parkour_npz_mp4.py \
        --npz parkour_motion_without_run_retargetted.npz \
        --mjcf ../../rl-wbc-g1-amp/third_party/gmr/assets/unitree_g1/g1_mocap_29dof.xml \
        --start 900 --end 2975 --out climbing_highlight.mp4

    # list detected clip segments sorted by base-height range (climbing candidates first)
    python render_parkour_npz_mp4.py --npz parkour_motion_without_run_retargetted.npz --list-segments
"""

import argparse
import subprocess
from pathlib import Path

import mujoco
import numpy as np


def detect_segments(base_pos: np.ndarray, fps: float, jump_threshold: float = 0.5):
    """Split a concatenated-clips npz into (start, end) segments via large frame-to-frame jumps."""
    dp = np.linalg.norm(np.diff(base_pos, axis=0), axis=1)
    jump_idx = (np.where(dp > jump_threshold)[0] + 1).tolist()
    bounds = [0] + jump_idx + [len(base_pos)]
    segs = []
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        z = base_pos[s:e, 2]
        segs.append({"start": s, "end": e, "frames": e - s, "duration_s": (e - s) / fps,
                     "z_min": float(z.min()), "z_max": float(z.max()), "z_range": float(z.max() - z.min())})
    return segs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--mjcf", required=True)
    parser.add_argument("--out", default="parkour_motion.mp4")
    parser.add_argument("--fps", type=float, default=None, help="Override output fps (default: clip's own fps)")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--list-segments", action="store_true",
                         help="Print detected clip segments (sorted by base-height range) and exit")
    parser.add_argument("--camera-mode", choices=["static", "chase"], default="static",
                         help="static: fixed camera framing the whole rendered range (default). "
                              "chase: camera lookat follows the base every frame.")
    parser.add_argument("--cam-lookat", type=float, nargs=3, default=None,
                         help="Static-mode only: fixed lookat point (default: trajectory centroid)")
    parser.add_argument("--cam-distance", type=float, default=None,
                         help="Static-mode only: camera distance (default: auto-fit to trajectory extent)")
    parser.add_argument("--cam-azimuth", type=float, default=120.0)
    parser.add_argument("--cam-elevation", type=float, default=-15.0)
    args = parser.parse_args()

    d = np.load(args.npz)
    joint_names_npz = list(d["joint_names"])
    base_pos_all = d["base_pos_w"]
    fps_native = float(d["framerate"])

    if args.list_segments:
        segs = detect_segments(base_pos_all, fps_native)
        segs.sort(key=lambda s: -s["z_range"])
        print(f"{len(segs)} segments (sorted by base-height range, most likely climbing first)\n")
        print(f"{'start':>6} {'end':>6} {'frames':>7} {'dur(s)':>7} {'z_min':>7} {'z_max':>7} {'z_range':>8}")
        for s in segs:
            print(f"{s['start']:6d} {s['end']:6d} {s['frames']:7d} {s['duration_s']:7.2f} "
                  f"{s['z_min']:7.3f} {s['z_max']:7.3f} {s['z_range']:8.3f}")
        return

    start = args.start
    end = args.end or base_pos_all.shape[0]
    base_pos = base_pos_all[start:end]
    base_quat_wxyz = d["base_quat_w"][start:end]
    joint_pos = d["joint_pos"][start:end]
    fps = args.fps or fps_native
    n_frames = joint_pos.shape[0]

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    data = mujoco.MjData(model)

    mjcf_joint_names = []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            mjcf_joint_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))
    reindex = [joint_names_npz.index(name) for name in mjcf_joint_names]

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = args.cam_azimuth
    cam.elevation = args.cam_elevation

    static_lookat = None
    if args.camera_mode == "static":
        static_lookat = np.array(args.cam_lookat) if args.cam_lookat else base_pos.mean(axis=0)
        if args.cam_distance is not None:
            cam.distance = args.cam_distance
        else:
            # auto-fit: distance covers the largest extent (y or z) of the rendered range,
            # with margin so the robot doesn't clip the frame edges
            extent = max(base_pos.max(axis=0) - base_pos.min(axis=0))
            cam.distance = max(2.5, extent * 1.8 + 1.5)
        cam.lookat[:] = static_lookat

    out_path = Path(args.out)
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{args.width}x{args.height}", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(out_path),
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    for t in range(n_frames):
        qpos = np.zeros(model.nq)
        qpos[0:3] = base_pos[t]
        qpos[3:7] = base_quat_wxyz[t]
        qpos[7:] = joint_pos[t][reindex]
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)

        if args.camera_mode == "chase":
            # recenters every frame -- handles segment-boundary position jumps cleanly, but
            # camera moves throughout, unlike static mode's fixed framing
            cam.lookat[:] = base_pos[t]

        renderer.update_scene(data, camera=cam)
        frame = renderer.render()  # (H, W, 3) uint8 RGB
        proc.stdin.write(frame.tobytes())

        if (t + 1) % 500 == 0:
            print(f"  rendered {t + 1}/{n_frames} frames")

    proc.stdin.close()
    proc.wait()
    print(f"Wrote {out_path} ({n_frames} frames @ {fps} fps = {n_frames/fps:.1f}s)")


if __name__ == "__main__":
    main()
