# AMP motion reference

`parkour_motion_without_run_retargetted.npz` (+ `parkour_motion_without_run.yaml`, the clip
selection list) is the retargeted human-motion source for the AMP style reward. It carries only
`joint_pos`, `base_pos_w` and `base_quat_w`, so it is not directly loadable — it is many separate
clips concatenated end to end, and MotionLoader needs per-body Cartesian data and velocities.

`isaaclab_project/g1_stairs/motions/convert_parkour_npz_to_amp_npz.py` turns it into the training
clip (`G1_parkour_climb.npz`): MuJoCo forward kinematics for the missing body data, per-segment
finite-difference velocities, and a filter down to ascending segments only.

Two preview tools: `render_parkour_npz_mp4.py` renders the raw clip to an MP4,
`visualize_parkour_npz_rerun.py` replays it interactively via [rerun](https://rerun.io/).
