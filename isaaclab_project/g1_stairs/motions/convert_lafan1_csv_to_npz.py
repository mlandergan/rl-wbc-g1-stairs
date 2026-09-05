"""Convert a LAFAN1-retargeted G1 motion (.csv, from the lvhaidong/LAFAN1_Retargeting_Dataset
mirror on Hugging Face of Unitree's own retargeting) into the .npz schema Isaac Lab's stock
AMP MotionLoader expects. Same FK/velocity approach as convert_gmr_to_npz.py -- see that
script's docstring for the general rationale.

Why this exists: to validate whether our "alien gait" / open-loop-fall issue is specific to
our own Strut_Walking_loop_g1 retargeting, or a property of any retargeted motion on this
robot/pipeline. Unitree's own dataset card is explicit that this data "only accounted for
kinematic constraints and did not include dynamic constraints or actuator limitations" --
i.e. it is not guaranteed to be open-loop stable either, so this is a *relative* comparison
tool, not a pass/fail validator.

The CSV's own column order already matches convert_gmr_to_npz.py's DOF_NAMES exactly (root
XYZ + quat xyzw, then the same 29 G1 joints in the same order) -- confirmed against the
dataset's README -- so no joint reordering is needed here, just a different loader.

Usage:
    python convert_lafan1_csv_to_npz.py --csv ../../../data/lafan1/walk1_subject1.csv \
        --mjcf ../../../third_party/gmr/assets/unitree_g1/g1_mocap_29dof.xml \
        --out motions/G1_lafan1_walk1.npz --start_frame 0 --num_frames 300
"""

import argparse

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d

from convert_gmr_to_npz import (
    BODY_NAMES,
    DOF_NAMES,
    FOOT_SOLE_OFFSET_M,
    MJCF_BODY_NAMES,
    quat_angular_velocity,
)

FPS = 30.0  # fixed by the dataset (see its README)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--mjcf", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=300, help="~10s at 30fps by default")
    parser.add_argument("--smooth_sigma", type=float, default=1.0)
    parser.add_argument("--ground_align", action="store_true", default=True,
                         help="Shift the clip vertically so the lowest foot-sole point over the selected "
                              "frame range touches z=0 (default: on) -- see convert_gmr_to_npz.py.")
    parser.add_argument("--no-ground_align", dest="ground_align", action="store_false")
    args = parser.parse_args()

    raw = np.loadtxt(args.csv, delimiter=",")
    end = args.start_frame + args.num_frames
    raw = raw[args.start_frame:end]

    root_pos = raw[:, 0:3].copy()
    root_rot = raw[:, 3:7].copy()  # xyzw, matches convert_gmr_to_npz.py's pkl convention
    dof_pos = raw[:, 7:36].copy()
    assert dof_pos.shape[1] == len(DOF_NAMES), f"expected {len(DOF_NAMES)} dofs, got {dof_pos.shape[1]}"

    dt = 1.0 / FPS
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
        foot_indexes = [BODY_NAMES.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
        offset = float(body_positions[:, foot_indexes, 2].min()) - FOOT_SOLE_OFFSET_M
        body_positions[:, :, 2] -= offset
        print(f"Ground-aligned clip: shifted z by {-offset:+.4f} m (lowest foot sole now at 0)")

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
        fps=FPS,
        dof_names=np.array(DOF_NAMES, dtype=np.str_),
        body_names=np.array(BODY_NAMES, dtype=np.str_),
        dof_positions=dof_pos.astype(np.float32),
        dof_velocities=dof_velocities,
        body_positions=body_positions,
        body_rotations=body_rotations,
        body_linear_velocities=body_linear_velocities,
        body_angular_velocities=body_angular_velocities,
    )
    print(f"Wrote {args.out}: {n_frames} frames @ {FPS} fps, {len(BODY_NAMES)} bodies, {len(DOF_NAMES)} dofs")


if __name__ == "__main__":
    main()
