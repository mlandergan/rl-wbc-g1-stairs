"""Convert the retargeted parkour motion into the .npz schema MotionLoader expects.

AMP_data/parkour_motion_without_run_retargetted.npz carries only joint_pos, base_pos_w and
base_quat_w, so MotionLoader cannot read it. This runs MuJoCo forward kinematics to recover the
per-body Cartesian data and finite-differences the velocities, the same way convert_gmr_to_npz.py
does for a GMR .pkl.

The input is many clips concatenated end to end, not one trajectory, so it is split on
frame-to-frame base-position jumps and every velocity is computed per segment -- otherwise each
join produces a huge bogus velocity spike. Only ascending segments are kept: this task always
climbs toward the centre platform, so descent references would teach the wrong behaviour.

MotionLoader still treats the output as one clip and can interpolate across a join, but the AMP
feature vector is entirely joint-space, root-relative or per-segment velocity, so the absolute
position discontinuity there is not an input to the discriminator.

Usage:
    python convert_parkour_npz_to_amp_npz.py \
        --npz ../../../AMP_data/parkour_motion_without_run_retargetted.npz \
        --mjcf <path to>/gmr/assets/unitree_g1/g1_mocap_29dof.xml \
        --out G1_parkour_climb.npz
"""
import argparse

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d

BODY_NAME_PAIRS = [
    ("pelvis", "pelvis"),
    ("left_shoulder_pitch_link", "left_shoulder_pitch_link"),
    ("right_shoulder_pitch_link", "right_shoulder_pitch_link"),
    ("left_elbow_link", "left_elbow_link"),
    ("right_elbow_link", "right_elbow_link"),
    ("right_hip_yaw_link", "right_hip_yaw_link"),
    ("left_hip_yaw_link", "left_hip_yaw_link"),
    ("right_hand_palm_link", "right_rubber_hand"),
    ("left_hand_palm_link", "left_rubber_hand"),
    ("right_ankle_roll_link", "right_ankle_roll_link"),
    ("left_ankle_roll_link", "left_ankle_roll_link"),
]
BODY_NAMES = [a for a, _ in BODY_NAME_PAIRS]
MJCF_BODY_NAMES = [b for _, b in BODY_NAME_PAIRS]

FOOT_SOLE_OFFSET_M = 0.035


def quat_angular_velocity(q_prev: np.ndarray, q_next: np.ndarray, dt: float) -> np.ndarray:
    """wxyz quaternions in, *world-frame* angular velocity (rad/s) out.

    Copied in behaviour from convert_gmr_to_npz.py's function of the same name: uses the
    world-frame relative rotation q_next * q_prev^-1 so the extracted axis is in world axes,
    matching Isaac Lab's body_ang_vel_w, which is what these values are compared against.
    """
    w0, x0, y0, z0 = q_prev
    inv = np.array([w0, -x0, -y0, -z0]) / max(w0 * w0 + x0 * x0 + y0 * y0 + z0 * z0, 1e-8)
    w1, x1, y1, z1 = q_next
    w2, x2, y2, z2 = inv
    q_rel = np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])
    n = np.linalg.norm(q_rel)
    if n < 1e-8:
        return np.zeros(3)
    q_rel /= n
    w = np.clip(q_rel[0], -1.0, 1.0)
    sin_half = np.sqrt(max(1.0 - w * w, 0.0))
    if sin_half < 1e-8:
        return np.zeros(3)
    return (2.0 * np.arccos(w) / dt) * (q_rel[1:] / sin_half)


def find_segments(base_pos: np.ndarray, jump_threshold: float, min_frames: int):
    """Split the concatenated clip on frame-to-frame base-position jumps.

    Same detection rule as AMP_data/render_parkour_npz_mp4.py:29-37.
    """
    step = np.linalg.norm(np.diff(base_pos, axis=0), axis=1)
    bounds = [0] + (np.where(step > jump_threshold)[0] + 1).tolist() + [len(base_pos)]
    return [(s, e) for s, e in zip(bounds[:-1], bounds[1:]) if e - s >= min_frames]


def central_diff(x: np.ndarray, dt: float) -> np.ndarray:
    """Central difference along axis 0, forward/backward at the boundaries."""
    v = np.zeros_like(x, dtype=np.float64)
    if x.shape[0] < 2:
        return v
    v[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
    v[0] = (x[1] - x[0]) / dt
    v[-1] = (x[-1] - x[-2]) / dt
    return v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True, help="retargeted parkour npz (joint_pos/base_pos_w/base_quat_w)")
    p.add_argument("--mjcf", required=True, help="GMR's g1_mocap_29dof.xml, for forward kinematics")
    p.add_argument("--out", required=True)
    p.add_argument("--smooth_sigma", type=float, default=1.0)
    p.add_argument("--jump_threshold", type=float, default=0.5,
                   help="frame-to-frame base displacement (m) above which a clip boundary is declared")
    p.add_argument("--min_segment_frames", type=int, default=50)
    p.add_argument("--min_net_rise", type=float, default=0.40,
                   help="keep only segments whose base rises at least this much end-to-end (m). "
                        "Ascents only: this project's robot always spawns on the border ring and "
                        "climbs toward the centre platform, so descent references would teach the "
                        "wrong behaviour. Set to a large negative number to keep everything.")
    p.add_argument("--list_only", action="store_true", help="print the segment table and exit")
    args = p.parse_args()

    d = np.load(args.npz)
    joint_pos_all = d["joint_pos"].astype(np.float64)
    base_pos_all = d["base_pos_w"].astype(np.float64)
    base_quat_all = d["base_quat_w"].astype(np.float64)
    npz_joint_names = [str(x) for x in d["joint_names"]]
    fps = float(d["framerate"])
    dt = 1.0 / fps

    segments = find_segments(base_pos_all, args.jump_threshold, args.min_segment_frames)
    kept = []
    print(f"{len(segments)} segments >= {args.min_segment_frames} frames "
          f"(of {len(base_pos_all)} frames @ {fps:g} fps)")
    print(f"{'start':>7}{'end':>7}{'frames':>7}{'sec':>7}{'net rise':>10}{'travel':>8}  keep")
    for s, e in segments:
        z = base_pos_all[s:e, 2]
        net = float(z[-1] - z[0])
        travel = float(np.linalg.norm(base_pos_all[e - 1, :2] - base_pos_all[s, :2]))
        keep = net >= args.min_net_rise
        if keep:
            kept.append((s, e))
        print(f"{s:>7}{e:>7}{e-s:>7}{(e-s)/fps:>7.1f}{net:>+10.2f}{travel:>8.2f}  {'YES' if keep else '-'}")

    if not kept:
        raise SystemExit(f"No segment has net rise >= {args.min_net_rise} m; nothing to convert.")
    total = sum(e - s for s, e in kept)
    print(f"\nkeeping {len(kept)} ascending segment(s), {total} frames = {total/fps:.1f} s")
    if args.list_only:
        return

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    data = mujoco.MjData(model)

    mjcf_joint_names = []
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            mjcf_joint_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))
    missing = [n for n in mjcf_joint_names if n not in npz_joint_names]
    if missing:
        raise ValueError(f"MJCF joints absent from the npz: {missing}")
    to_mjcf = [npz_joint_names.index(n) for n in mjcf_joint_names]

    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in MJCF_BODY_NAMES]
    if any(b == -1 for b in body_ids):
        raise ValueError(f"Body name(s) not in MJCF: "
                         f"{[n for n, b in zip(MJCF_BODY_NAMES, body_ids) if b == -1]}")

    nb = len(BODY_NAMES)
    seg_dof_pos, seg_body_pos, seg_body_rot = [], [], []
    seg_dof_vel, seg_body_lin, seg_body_ang = [], [], []

    for s, e in kept:
        n = e - s
        dof_pos = joint_pos_all[s:e]
        body_pos = np.zeros((n, nb, 3))
        body_rot = np.zeros((n, nb, 4))
        for t in range(n):
            qpos = np.zeros(model.nq)
            qpos[0:3] = base_pos_all[s + t]
            qpos[3:7] = base_quat_all[s + t]
            qpos[7:] = dof_pos[t][to_mjcf]
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            for j, bid in enumerate(body_ids):
                body_pos[t, j] = data.xpos[bid]
                body_rot[t, j] = data.xquat[bid]

        dof_vel = gaussian_filter1d(central_diff(dof_pos, dt), sigma=args.smooth_sigma, axis=0)
        body_lin = gaussian_filter1d(central_diff(body_pos, dt), sigma=args.smooth_sigma, axis=0)
        body_ang = np.zeros((n, nb, 3))
        for j in range(nb):
            q = body_rot[:, j, :]
            av = np.zeros((n, 3))
            if n > 1:
                av[0] = quat_angular_velocity(q[0], q[1], dt)
                av[-1] = quat_angular_velocity(q[-2], q[-1], dt)
            for k in range(1, n - 1):
                av[k] = 0.5 * (quat_angular_velocity(q[k - 1], q[k], dt)
                               + quat_angular_velocity(q[k], q[k + 1], dt))
            body_ang[:, j, :] = gaussian_filter1d(av, sigma=args.smooth_sigma, axis=0)

        seg_dof_pos.append(dof_pos)
        seg_body_pos.append(body_pos)
        seg_body_rot.append(body_rot)
        seg_dof_vel.append(dof_vel)
        seg_body_lin.append(body_lin)
        seg_body_ang.append(body_ang)

    dof_positions = np.concatenate(seg_dof_pos, axis=0)
    dof_velocities = np.concatenate(seg_dof_vel, axis=0)
    body_positions = np.concatenate(seg_body_pos, axis=0)
    body_rotations = np.concatenate(seg_body_rot, axis=0)
    body_linear_velocities = np.concatenate(seg_body_lin, axis=0)
    body_angular_velocities = np.concatenate(seg_body_ang, axis=0)

    foot_idx = [BODY_NAMES.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
    offset = float(body_positions[:, foot_idx, 2].min()) - FOOT_SOLE_OFFSET_M
    body_positions[:, :, 2] -= offset
    print(f"Ground-aligned: shifted z by {-offset:+.4f} m (lowest sole now at 0)")

    pelvis = BODY_NAMES.index("pelvis")
    print(f"pelvis height over clip: min {body_positions[:, pelvis, 2].min():.3f} m  "
          f"max {body_positions[:, pelvis, 2].max():.3f} m")

    np.savez(
        args.out,
        fps=np.array(fps),
        dof_names=np.array(npz_joint_names),
        body_names=np.array(BODY_NAMES),
        dof_positions=dof_positions.astype(np.float32),
        dof_velocities=dof_velocities.astype(np.float32),
        body_positions=body_positions.astype(np.float32),
        body_rotations=body_rotations.astype(np.float32),
        body_linear_velocities=body_linear_velocities.astype(np.float32),
        body_angular_velocities=body_angular_velocities.astype(np.float32),
    )
    print(f"\nWrote {args.out}: {dof_positions.shape[0]} frames, {dof_positions.shape[0]/fps:.1f} s, "
          f"{len(BODY_NAMES)} bodies, {len(npz_joint_names)} dofs")


if __name__ == "__main__":
    main()
