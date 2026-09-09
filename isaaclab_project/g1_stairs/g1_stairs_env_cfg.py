"""Config for the vision-based AMP stair-climbing environment.

The robot spawns on the flat border ring of a procedural pyramid-stairs tile and is commanded
toward the raised centre platform. Reward is velocity tracking plus regularization, combined with
an AMP style reward against a climbing reference clip (see g1_stairs_env.py).

Terrain is a curriculum: `step_height_range` is interpolated across STAIRS_NUM_LEVELS difficulty
rows, from a 5 cm step a walking gait already clears up to 20 cm. Envs are promoted or demoted on
velocity-tracking score by G1StairsEnv._update_terrain_curriculum.

Reward weights, event ranges, sensor geometry and the depth pipeline follow project-instinct's
InstinctLab parkour task, which solves the same problem on the same robot; see
THIRD_PARTY_NOTICES.md for attribution (design reference only, nothing vendored).

Robot asset is Isaac Lab's G1_29DOF_CFG, not G1_MINIMAL_CFG: its joint convention
(waist yaw/roll/pitch split, single-DOF elbow, wrist roll/pitch/yaw) matches the retargeted
motion data this project line uses.
"""

from __future__ import annotations

import copy
import math
import os

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCameraCfg, RayCasterCfg, patterns
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import FlatPatchSamplingCfg, TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.terrains.height_field import HfPyramidStairsTerrainCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg

from isaaclab_assets import G1_29DOF_CFG

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")

STAIRS_STEP_HEIGHT_RANGE = (0.05, 0.20)
STAIRS_NUM_LEVELS = 10
STAIRS_NUM_TYPE_COLS = 4
STAIRS_MAX_INIT_LEVEL = 2

_STAIRS_RISER_HEIGHT_M = 0.178
_STAIRS_TREAD_DEPTH_M = 0.2794
_STAIRS_NUM_STEPS = 5
_STAIRS_BORDER_WIDTH_M = 0.5
_STAIRS_PLATFORM_WIDTH_M = 1.2
_STAIRS_TILE_SIZE_M = 2.0 * (
    _STAIRS_NUM_STEPS * _STAIRS_TREAD_DEPTH_M + _STAIRS_BORDER_WIDTH_M + _STAIRS_PLATFORM_WIDTH_M / 2.0
)
_STAIRS_HORIZONTAL_SCALE_M = 0.05
_STAIRS_VERTICAL_SCALE_M = 0.005

STAIRS_SPAWN_RADIUS_M = _STAIRS_TILE_SIZE_M / 2.0 - _STAIRS_BORDER_WIDTH_M / 2.0
STAIRS_SPAWN_LATERAL_MARGIN_M = _STAIRS_TILE_SIZE_M / 2.0 - _STAIRS_BORDER_WIDTH_M
STAIRS_STAIR_REGION_HALF_EXTENT_M = _STAIRS_TILE_SIZE_M / 2.0 - _STAIRS_BORDER_WIDTH_M
STAIRS_TREAD_DEPTH_ACTUAL_M = (
    round(_STAIRS_TREAD_DEPTH_M / _STAIRS_HORIZONTAL_SCALE_M) * _STAIRS_HORIZONTAL_SCALE_M
)

STAIRS_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(_STAIRS_TILE_SIZE_M, _STAIRS_TILE_SIZE_M),
    border_width=2.0,
    num_rows=STAIRS_NUM_LEVELS,
    num_cols=STAIRS_NUM_TYPE_COLS,
    curriculum=True,
    horizontal_scale=_STAIRS_HORIZONTAL_SCALE_M,
    vertical_scale=_STAIRS_VERTICAL_SCALE_M,
    sub_terrains={
        "stairs": HfPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=STAIRS_STEP_HEIGHT_RANGE,
            step_width=_STAIRS_TREAD_DEPTH_M,
            platform_width=_STAIRS_PLATFORM_WIDTH_M,
            border_width=_STAIRS_BORDER_WIDTH_M,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=16, patch_radius=0.4, max_height_diff=0.05),
            },
        ),
    },
)

_G1_29DOF_SOFT_ARMS_CFG = copy.deepcopy(G1_29DOF_CFG)
_G1_29DOF_SOFT_ARMS_CFG.prim_path = "/World/envs/env_.*/Robot"
_G1_29DOF_SOFT_ARMS_CFG.actuators["arms"].stiffness = 40.0
_G1_29DOF_SOFT_ARMS_CFG.actuators["arms"].damping = 10.0
_G1_29DOF_SOFT_ARMS_CFG.actuators["waist"].stiffness = 200.0
_G1_29DOF_SOFT_ARMS_CFG.actuators["waist"].damping = 5.0
_G1_29DOF_SOFT_ARMS_CFG.spawn.activate_contact_sensors = True

# DelayedPDActuator is an EXPLICIT actuator model where the stock G1 config uses implicit PhysX
# PD. Same gains either way (as InstinctLab does it), but the dynamics are not identical -- set
# this False first if a run is unstable in a way nothing else explains.
# DISABLED 2026-09-08 for a termination-diagnostic run: a 4090 run reported episodes lasting
# ~3 s even at 6k iterations, which is exactly the "unstable in a way nothing else explains"
# case this flag's own comment says to rule out first. Explicit/implicit actuator models do not
# produce identical dynamics, so it is isolated here rather than left confounded with the
# termination logic under investigation. Re-enable once episode length is understood.
ENABLE_ACTUATOR_DELAY = False
ACTUATOR_MIN_DELAY_STEPS = 0
ACTUATOR_MAX_DELAY_STEPS = 1

if ENABLE_ACTUATOR_DELAY:
    from isaaclab.actuators import DelayedPDActuatorCfg

    _delayed_actuators = {}
    for _group_name, _actuator in _G1_29DOF_SOFT_ARMS_CFG.actuators.items():
        _fields = {
            f: getattr(_actuator, f)
            for f in ("joint_names_expr", "effort_limit", "velocity_limit", "stiffness", "damping",
                      "armature", "friction")
            if getattr(_actuator, f, None) is not None
        }
        _delayed_actuators[_group_name] = DelayedPDActuatorCfg(
            min_delay=ACTUATOR_MIN_DELAY_STEPS,
            max_delay=ACTUATOR_MAX_DELAY_STEPS,
            **_fields,
        )
    _G1_29DOF_SOFT_ARMS_CFG.actuators = _delayed_actuators


@configclass
class G1StairsEnvCfg(DirectRLEnvCfg):
    """G1 stairs environment config: velocity-tracking task reward + AMP style reward."""

    rew_lin_vel_xy = 2.0
    rew_ang_vel_z = 2.0
    rew_track_sigma = 0.25

    rew_heading_error = -1.0

    rew_termination = 0.0
    rew_is_alive = 3.0
    rew_action_rate_l2 = -0.005
    rew_joint_pos_limits = -1.0
    rew_joint_acc_l2 = -1.25e-7
    rew_dof_torques_l2 = -1.5e-7
    rew_flat_orientation_l2 = -3.0
    rew_lin_vel_z_l2 = -0.2
    rew_ang_vel_xy_l2 = -0.05
    rew_joint_deviation_hip = -0.5
    rew_joint_deviation_arms = -0.004
    rew_joint_deviation_torso = -0.004

    # Positive weight but still a penalty: the formula is exp(-clamp(thr - dy, 0)/std^2) - 1,
    # which is <= 0 and rises to 0 as the feet separate. Kept sign-for-sign as InstinctLab has it.
    rew_feet_close_xy = 0.4
    feet_close_xy_threshold_m = 0.12
    feet_close_xy_std = math.sqrt(0.05)

    rew_feet_air_time = 0.5
    feet_air_time_vel_threshold_mps = 0.15
    rew_feet_slide = -0.4
    rew_energy = -5.0e-5
    rew_torque_limits = -0.01
    torque_limit_ratio = 0.8
    rew_undesired_contacts = -1.0
    undesired_contact_threshold_n = 1.0
    rew_dof_vel_limits = -1.0
    dof_vel_limit_soft_ratio = 0.9
    rew_dof_vel_l2 = -1.0e-4

    rew_feet_height_error = -0.1
    feet_height_margin_m = 0.035
    feet_height_clip_max_m = 0.3

    rew_edge_penetration = -4.0
    edge_safety_margin_m = 0.05

    rew_dont_wait = -0.5
    dont_wait_cmd_threshold_mps = 0.3
    rew_stand_still = -0.3
    stand_still_offset = 4.0
    stand_still_cmd_threshold = 0.15

    rew_max_height = 0.0

    feet_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*_ankle_roll_link",
        history_length=3,
        track_air_time=True,
    )

    body_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        track_air_time=False,
    )

    left_height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/left_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.04, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.12, size=[0.12, 0.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=0.02,
    )
    right_height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/right_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.04, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.12, size=[0.12, 0.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=0.02,
    )
    base_height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/pelvis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.0, 0.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=0.02,
    )

    DEPTH_RAW_WIDTH = 64
    DEPTH_RAW_HEIGHT = 36
    DEPTH_HFOV_DEG = 89.51
    DEPTH_VFOV_DEG = 58.29
    DEPTH_RANGE_M = (0.0, 2.5)
    DEPTH_CROP_REGION = (18, 0, 16, 16)
    DEPTH_FINAL_SIZE = 16
    DEPTH_BLUR_KERNEL_SIZE = 3
    DEPTH_BLUR_SIGMA = 1.0

    depth_camera: RayCasterCameraCfg = RayCasterCameraCfg(
        prim_path="/World/envs/env_.*/Robot/torso_link",
        mesh_prim_paths=["/World/ground"],
        ray_alignment="yaw",
        pattern_cfg=PinholeCameraPatternCfg(
            focal_length=1.0,
            horizontal_aperture=2 * math.tan(math.radians(DEPTH_HFOV_DEG) / 2),
            vertical_aperture=2 * math.tan(math.radians(DEPTH_VFOV_DEG) / 2),
            width=DEPTH_RAW_WIDTH,
            height=DEPTH_RAW_HEIGHT,
        ),
        debug_vis=False,
        data_types=["distance_to_image_plane"],
        update_period=0.02,
        depth_clipping_behavior="max",
        max_distance=DEPTH_RANGE_M[1],
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=(0.0487988662332928, 0.01, 0.4378029937970051),
            rot=(0.9135367613482678, 0.004363309284746571, 0.4067366430758002, 0.0),
            convention="world",
        ),
    )

    command_pos_k_v = 2.0
    command_pos_k_w = 2.0
    command_lin_vel_x_max = 0.8
    command_ang_vel_z_max = 1.0
    command_target_dist_threshold_m = 0.4
    command_resample_time_s = 20.0

    debug_vis_goal: bool = False

    episode_length_s = 20.0
    decimation = 4

    policy_proprio_history_len = 8

    depth_history_len = 8
    depth_history_skip = 5
    depth_history_buffer_len = 37

    privileged_obs_dim = 5

    observation_space = (
        policy_proprio_history_len * (70 + 3 * 10)
        + 3
        + depth_history_len * DEPTH_FINAL_SIZE**2
        + privileged_obs_dim
    )
    action_space = 29
    state_space = 0
    num_amp_observations = 10
    amp_observation_space = 70 + 3 * 10

    action_scale = 0.5

    early_termination = True
    termination_height = 0.5
    """Minimum root height, now measured RELATIVE TO THE LOCAL TERRAIN under the robot rather than
    in absolute world Z (changed 2026-09-06, audit finding #5). The old absolute check drifted with
    terrain height: standing on the centre platform, the pelvis had to fall ~1.3 m to trigger it,
    so termination got steadily more lenient the higher the robot climbed. InstinctLab's own
    version is terrain-relative too (`root_height_below_env_origin_minimum`,
    tasks/parkour/mdp/terminations.py:44-51)."""

    termination_bad_orientation_rad = 1.0
    termination_contact_body_names = ["torso_link"]
    termination_contact_threshold_n = 1.0

    terrain_curriculum_lin_vel_threshold = (0.3, 0.6)

    event_static_friction_range = (0.3, 1.6)
    event_dynamic_friction_range = (0.3, 1.6)
    event_restitution_range = (0.05, 0.5)
    event_material_num_buckets = 64
    event_reset_pos_range_m = 0.1
    event_reset_yaw_range_rad = 0.1
    event_reset_lin_vel_range = 0.2
    event_reset_ang_vel_range = 0.2
    event_reset_joint_pos_range_rad = 0.15

    motion_file: str = os.path.join(MOTIONS_DIR, "G1_parkour_climb.npz")
    """AMP reference motion. Changed 2026-09-06 (audit finding #3) from `G1_strut_walk.npz`.

    The strut-walk clip was FLAT-GROUND walking, which meant the discriminator was rewarding
    near-constant pelvis height, small hip/knee flexion and a level trunk -- the opposite of a
    stair step-up on every axis -- so the style reward was actively pointing away from the task,
    and marching in place at the base of the stairs was close to optimal under it.

    `G1_parkour_climb.npz` is built by motions/convert_parkour_npz_to_amp_npz.py from
    AMP_data/parkour_motion_without_run_retargetted.npz, which was already retargeted but had
    never been converted into MotionLoader's schema (it carries only joint_pos/base_pos_w/
    base_quat_w -- no per-body Cartesian data or velocities -- so MotionLoader simply could not
    read it, which is why the config still pointed at the flat-ground clip). The converter runs
    MuJoCo forward kinematics for the missing body data, splits the file on its clip joins, and
    keeps only the ASCENDING segments: 7 segments, 1048 frames, 21.0 s at 50 fps, pelvis rising
    0.777 m -> 1.402 m. (The 0.777 m floor is a good independent check that the FK and the
    pelvis-root assumption are right -- that is G1's real standing pelvis height.)"""
    reference_body = "pelvis"
    reset_strategy = "random"
    """Strategy to be followed when resetting each environment (humanoid's pose and joint states).

    * default: pose and joint states are set to the initial state of the asset.
    * random: pose and joint states are set by sampling motions at random, uniform times.
    * random-start: pose and joint states are set by sampling motion at the start (time zero).
    """

    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
            gpu_collision_stack_size=2**28,
            gpu_max_rigid_patch_count=10 * 2**15,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=STAIRS_TERRAIN_CFG,
        max_init_terrain_level=STAIRS_MAX_INIT_LEVEL,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = _G1_29DOF_SOFT_ARMS_CFG


@configclass
class G1StairsEnvCfg_PLAY(G1StairsEnvCfg):
    """Smaller-scene eval variant, matching Isaac Lab's `_PLAY` convention.

    Viewer follows env 0's robot root (close-up follow-cam for videos). Set here rather than
    via hydra override because Isaac Lab's config updater rejects overrides of None-defaulted
    fields (viewer.asset_name) with a NoneType type-check error."""

    def __post_init__(self):
        self.scene.num_envs = 32
        self.scene.env_spacing = 3.0
        self.debug_vis_goal = True
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (4.0, 0.0, 3.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)
