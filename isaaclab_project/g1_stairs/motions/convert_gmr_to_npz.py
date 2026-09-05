"""Convert a GMR-retargeted G1 motion (.pkl, from third_party/gmr's bvh_to_robot.py)
into the .npz schema Isaac Lab's stock AMP MotionLoader expects.

GMR's pkl only carries root_pos/root_rot/dof_pos (no per-body Cartesian data), so this
computes body positions/rotations via MuJoCo forward kinematics against the same G1
model GMR itself uses for retargeting, then derives velocities by finite difference
(central difference, forward/backward at the clip boundaries), matching the approach
used by reference converters in this space (e.g. linden713/humanoid_amp's data_convert.py,
which uses Pinocchio instead of MuJoCo for the same FK step).

Usage:
    python convert_gmr_to_npz.py --pkl ../../../data/mixamo/Strut_Walking_loop_g1.pkl \
        --mjcf ../../../third_party/gmr/assets/unitree_g1/g1_mocap_29dof.xml \
        --out motions/G1_strut_walk.npz
"""

import argparse
import pickle

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d

DOF_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# (npz_name, mjcf_name) pairs. npz_name is what we store in the output file and what
# g1_stairs_env.py's key_body_names looks up by (must match Isaac Lab's G1_29DOF_CFG body
# names). mjcf_name is what GMR's own g1_mocap_29dof.xml calls that body for FK purposes.
# These differ only for the hands: Isaac Lab's asset calls it "*_hand_palm_link", GMR's
# calls it "*_rubber_hand" -- same body, two different naming conventions, matched here
# by hand since nothing else ties them together automatically.
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
BODY_NAMES = [npz_name for npz_name, _ in BODY_NAME_PAIRS]
MJCF_BODY_NAMES = [mjcf_name for _, mjcf_name in BODY_NAME_PAIRS]

# Vertical distance from the ankle_roll_link frame origin down to the foot sole contact plane:
# g1_mocap_29dof.xml models the sole as four contact spheres at z=-0.03 with radius 0.005.
# Used by the ground-alignment step to decide where "foot touching the floor" actually is.
FOOT_SOLE_OFFSET_M = 0.035


def quat_angular_velocity(q_prev: np.ndarray, q_next: np.ndarray, dt: float) -> np.ndarray:
    """wxyz quaternions in, *world-frame* angular velocity (rad/s) out.

    Uses the world-frame relative rotation q_next * q_prev^-1, so the extracted rotation
    axis is expressed in world axes -- matching Isaac Lab's body_ang_vel_w, which is what
    these values get compared against in the AMP observation. (An earlier version computed
    q_prev^-1 * q_next, whose axis is in the *body* frame -- a small but systematic frame
    mismatch between the reference data and the simulated robot.)
    """
    w0, x0, y0, z0 = q_prev
    inv = np.array([w0, -x0, -y0, -z0]) / max(w0 * w0 + x0 * x0 + y0 * y0 + z0 * z0, 1e-8)
    w1, x1, y1, z1 = q_next
    w2, x2, y2, z2 = inv
    # q_rel = q_next * inv (Hamilton product, world-frame delta)
    rw = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    rx = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    ry = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    rz = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    q_rel = np.array([rw, rx, ry, rz])
    n = np.linalg.norm(q_rel)
    if n < 1e-8:
        return np.zeros(3)
    q_rel /= n
    w = np.clip(q_rel[0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(max(1.0 - w * w, 0.0))
    if sin_half < 1e-8:
        return np.zeros(3)
    axis = q_rel[1:] / sin_half
    return (angle / dt) * axis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--mjcf", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--smooth_sigma", type=float, default=1.0)
    parser.add_argument("--align_x", action="store_true", default=True,
                         help="Rotate the clip so net travel direction points along world +X (default: on)")
    parser.add_argument("--no-align_x", dest="align_x", action="store_false")
    parser.add_argument("--ground_align", action="store_true", default=True,
                         help="Shift the clip vertically so the lowest foot-sole point over the whole clip "
                              "touches z=0 (default: on). GMR output is not guaranteed to be ground-registered: "
                              "the Strut Walking clip came out floating ~0.08-0.09 m, which made every training "
                              "reset a free-fall drop and put the reference root height physically out of reach "
                              "of a robot actually standing on the floor.")
    parser.add_argument("--no-ground_align", dest="ground_align", action="store_false")
    args = parser.parse_args()

    with open(args.pkl, "rb") as f:
        motion = pickle.load(f)

    root_pos = motion["root_pos"].copy()
    root_rot = motion["root_rot"].copy()  # xyzw
    dof_pos = motion["dof_pos"]

    if args.align_x:
        # GMR/Blender's world frame doesn't guarantee the clip travels along +X, but Isaac Lab's
        # AMP reference impl (linden713/humanoid_amp) uses *world-frame* root velocity/orientation
        # in the discriminator observation (not root-relative), so align net travel direction with
        # +X to match Isaac Lab's forward convention and avoid a world-frame heading mismatch.
        disp = root_pos[-1, :2] - root_pos[0, :2]
        yaw = np.arctan2(disp[1], disp[0])
        c, s = np.cos(-yaw), np.sin(-yaw)
        rot_mat = np.array([[c, -s], [s, c]])
        root_pos[:, :2] = root_pos[:, :2] @ rot_mat.T
        # compose with a yaw quaternion (xyzw) on the left: q_align * q_root
        half = -yaw / 2.0
        qz, qw = np.sin(half), np.cos(half)
        x0, y0, z0, w0 = root_rot[:, 0], root_rot[:, 1], root_rot[:, 2], root_rot[:, 3]
        root_rot = np.stack([
            qw * x0 - qz * y0,
            qw * y0 + qz * x0,
            qw * z0 + qz * w0,
            qw * w0 - qz * z0,
        ], axis=-1)
        print(f"Aligned clip to +X: rotated by {np.degrees(-yaw):.1f} deg")
    fps = motion["fps"]
    dt = 1.0 / fps
    n_frames = dof_pos.shape[0]

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    data = mujoco.MjData(model)
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in MJCF_BODY_NAMES]
    if any(b == -1 for b in body_ids):
        missing = [n for n, b in zip(MJCF_BODY_NAMES, body_ids) if b == -1]
        raise ValueError(f"Body name(s) not found in MJCF: {missing}")

    body_positions = np.zeros((n_frames, len(BODY_NAMES), 3), dtype=np.float32)
    body_rotations = np.zeros((n_frames, len(BODY_NAMES), 4), dtype=np.float32)  # wxyz

    for t in range(n_frames):
        qpos = np.zeros(model.nq)
        qpos[0:3] = root_pos[t]
        x, y, z, w = root_rot[t]
        qpos[3:7] = [w, x, y, z]
        qpos[7:] = dof_pos[t]
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        for j, bid in enumerate(body_ids):
            body_positions[t, j] = data.xpos[bid]
            body_rotations[t, j] = data.xquat[bid]

    if args.ground_align:
        # constant vertical shift for the whole clip (per-frame shifting would distort the
        # dynamics): lowest sole-contact point across all frames lands exactly on z=0.
        foot_indexes = [BODY_NAMES.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
        offset = float(body_positions[:, foot_indexes, 2].min()) - FOOT_SOLE_OFFSET_M
        body_positions[:, :, 2] -= offset
        print(f"Ground-aligned clip: shifted z by {-offset:+.4f} m (lowest foot sole now at 0)")

    # dof velocities: central difference, forward/backward at boundaries, then smooth
    dof_velocities = np.zeros_like(dof_pos, dtype=np.float32)
    dof_velocities[1:-1] = (dof_pos[2:] - dof_pos[:-2]) / (2 * dt)
    dof_velocities[0] = (dof_pos[1] - dof_pos[0]) / dt
    dof_velocities[-1] = (dof_pos[-1] - dof_pos[-2]) / dt
    dof_velocities = gaussian_filter1d(dof_velocities, sigma=args.smooth_sigma, axis=0).astype(np.float32)

    body_linear_velocities = np.zeros_like(body_positions)
    body_linear_velocities[1:-1] = (body_positions[2:] - body_positions[:-2]) / (2 * dt)
    body_linear_velocities[0] = (body_positions[1] - body_positions[0]) / dt
    body_linear_velocities[-1] = (body_positions[-1] - body_positions[-2]) / dt
    body_linear_velocities = gaussian_filter1d(body_linear_velocities, sigma=args.smooth_sigma, axis=0).astype(np.float32)

    body_angular_velocities = np.zeros((n_frames, len(BODY_NAMES), 3), dtype=np.float32)
    for j in range(len(BODY_NAMES)):
        quats = body_rotations[:, j, :]
        av = np.zeros((n_frames, 3), dtype=np.float32)
        if n_frames > 1:
            av[0] = quat_angular_velocity(quats[0], quats[1], dt)
            av[-1] = quat_angular_velocity(quats[-2], quats[-1], dt)
        for k in range(1, n_frames - 1):
            av1 = quat_angular_velocity(quats[k - 1], quats[k], dt)
            av2 = quat_angular_velocity(quats[k], quats[k + 1], dt)
            av[k] = 0.5 * (av1 + av2)
        body_angular_velocities[:, j, :] = gaussian_filter1d(av, sigma=args.smooth_sigma, axis=0)

    np.savez(
        args.out,
        fps=fps,
        dof_names=np.array(DOF_NAMES, dtype=np.str_),
        body_names=np.array(BODY_NAMES, dtype=np.str_),
        dof_positions=dof_pos.astype(np.float32),
        dof_velocities=dof_velocities,
        body_positions=body_positions,
        body_rotations=body_rotations,
        body_linear_velocities=body_linear_velocities,
        body_angular_velocities=body_angular_velocities,
    )
    print(f"Wrote {args.out}: {n_frames} frames @ {fps} fps, {len(BODY_NAMES)} bodies, {len(DOF_NAMES)} dofs")


if __name__ == "__main__":
    main()
