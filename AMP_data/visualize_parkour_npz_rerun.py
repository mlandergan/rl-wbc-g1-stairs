"""Replay a "Hiking in the Wild"-style retargeted parkour motion npz in Rerun, using the
G1 MuJoCo model for forward kinematics and rendering the actual robot mesh geometry.

This npz schema (framerate, joint_names, joint_pos, base_pos_w, base_quat_w) differs from
rl-wbc-g1-amp's MotionLoader .npz convention (fps, dof_names, body_names, dof_positions,
body_positions, ...) -- it's a raw retargeted clip, pre-FK, with no key-body/velocity data
computed yet. base_quat_w is wxyz (confirmed empirically: interpreting it as wxyz gives an
upright robot -- R[2,2] ~ 1 -- across the whole clip; xyzw does not).

Usage:
    python visualize_parkour_npz_rerun.py \
        --npz parkour_motion_without_run_retargetted.npz \
        --mjcf ../../rl-wbc-g1-amp/third_party/gmr/assets/unitree_g1/g1_mocap_29dof.xml
"""

import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import rerun as rr


def parse_mesh_files(mjcf_path: str) -> dict[str, str]:
    """Map MJCF <mesh name=...> to its resolved file path, honoring <compiler meshdir=...>."""
    root = ET.parse(mjcf_path).getroot()
    mjcf_dir = Path(mjcf_path).resolve().parent
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    mesh_files = {}
    for mesh in root.iter("mesh"):
        name = mesh.get("name")
        file = mesh.get("file")
        if name and file:
            mesh_files[name] = str(mjcf_dir / meshdir / file)
    return mesh_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--mjcf", required=True)
    parser.add_argument("--fps", type=float, default=None, help="Override playback fps")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--end", type=int, default=None, help="End frame index (exclusive)")
    parser.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=False,
                         help="Pace logging to real motion duration (default: off, log as fast "
                              "as possible and scrub the timeline in the viewer instead)")
    args = parser.parse_args()

    d = np.load(args.npz)
    joint_names_npz = list(d["joint_names"])
    base_pos = d["base_pos_w"]  # (N, 3)
    base_quat_wxyz = d["base_quat_w"]  # (N, 4), wxyz
    joint_pos = d["joint_pos"]  # (N, 29)
    fps = args.fps or float(d["framerate"])

    start = args.start
    end = args.end or joint_pos.shape[0]
    base_pos, base_quat_wxyz, joint_pos = base_pos[start:end], base_quat_wxyz[start:end], joint_pos[start:end]
    n_frames = joint_pos.shape[0]

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    data = mujoco.MjData(model)

    # map npz joint_pos columns -> MJCF qpos order (freejoint occupies qpos[0:7])
    mjcf_joint_names = []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            mjcf_joint_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))
    reindex = [joint_names_npz.index(name) for name in mjcf_joint_names]
    if mjcf_joint_names != joint_names_npz:
        print(f"NOTE: npz joint order != MJCF hinge order, reindexing {len(reindex)} joints.")

    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}" for i in range(model.nbody)]
    mesh_files = parse_mesh_files(args.mjcf)

    body_mesh_file = {}
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        body_id = model.geom_bodyid[g]
        mesh_id = model.geom_dataid[g]
        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
        if mesh_name in mesh_files:
            body_mesh_file[body_id] = mesh_files[mesh_name]

    rr.init("g1_parkour_motion", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    grid = []
    for i in range(-2, 11):
        grid.append([[i, -2.0, 0.0], [i, 3.0, 0.0]])
    for j in range(-2, 4):
        grid.append([[-2.0, j, 0.0], [10.0, j, 0.0]])
    rr.log("world/ground_grid", rr.LineStrips3D(grid, colors=(90, 90, 90), radii=0.002), static=True)

    for body_id, mesh_path in body_mesh_file.items():
        name = body_names[body_id]
        rr.log(f"g1/{name}/mesh", rr.Asset3D(path=mesh_path), static=True)

    parents = [model.body_parentid[i] for i in range(model.nbody)]
    edges = [(i, parents[i]) for i in range(1, model.nbody) if parents[i] >= 0]

    frame_dt = 1.0 / fps
    t_start = time.time()
    for t in range(n_frames):
        rr.set_time_seconds("motion_time", t / fps)

        qpos = np.zeros(model.nq)
        qpos[0:3] = base_pos[t]
        qpos[3:7] = base_quat_wxyz[t]  # already wxyz, matches MuJoCo freejoint convention
        qpos[7:] = joint_pos[t][reindex]
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)

        for body_id in body_mesh_file:
            name = body_names[body_id]
            rr.log(
                f"g1/{name}",
                rr.Transform3D(translation=data.xpos[body_id], quaternion=data.xquat[body_id][[1, 2, 3, 0]]),
            )

        xpos = data.xpos.copy()
        segments = [[xpos[c], xpos[p]] for c, p in edges]
        rr.log("g1/skeleton", rr.LineStrips3D(segments, colors=(58, 90, 122), radii=0.004))
        rr.log("g1/base_trail", rr.Points3D(base_pos[max(0, t - 200):t + 1], colors=(230, 140, 30), radii=0.006))

        if args.realtime:
            target = t_start + (t + 1) * frame_dt
            sleep_s = target - time.time()
            if sleep_s > 0:
                time.sleep(sleep_s)

    print(f"Logged {n_frames} frames ({n_frames/fps:.1f}s) at {fps} fps to Rerun viewer.")
    print(f"Rendered mesh for {len(body_mesh_file)}/{model.nbody} bodies.")
    print(f"Base height range: [{base_pos[:,2].min():.3f}, {base_pos[:,2].max():.3f}] m")


if __name__ == "__main__":
    main()
