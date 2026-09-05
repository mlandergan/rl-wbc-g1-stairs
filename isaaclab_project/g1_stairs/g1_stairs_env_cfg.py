"""Vision-based AMP stair climbing environment config — velocity-tracking task reward layered
on an AMP style reward (see g1_stairs_env.py for the reward/observation logic), forked from
rl-wbc-g1-amp's g1_amp_env_cfg.py.

Adapted (via rl-wbc-g1-amp) from linden713/humanoid_amp (BSD-3-Clause), which wires Isaac Lab's
stock AMP machinery to a direct-workflow G1 env but only for pure imitation (all task reward
scales are zero there).

Robot asset: Isaac Lab's own G1_29DOF_CFG (isaaclab_assets), NOT G1_MINIMAL_CFG (what
rl-wbc-g1-baseline / stock G1FlatEnvCfg actually uses) -- matches this project line's
GMR-retargeted motion data convention (waist_yaw/roll/pitch split, single-DOF elbow,
wrist_roll/pitch/yaw). See project_description.md's Robot section.

**Scaffolding status (see project_description.md's Initial Tasks):** Task 1 (fork/scaffold),
Task 2 (this file's STAIRS_TERRAIN_CFG / G1StairsEnvCfg.terrain), Task 3 (position-based,
direction-biased velocity command -- command_pos_k_v/k_w below, computed each step in
g1_stairs_env.py's _update_position_based_commands), and Task 4 (border-ring spawn placement --
STAIRS_SPAWN_RADIUS_M / _LATERAL_MARGIN_M / TOTAL_RISE_M below, consumed by
g1_stairs_env.py's _sample_border_spawn_positions) are done and GPU-VM-verified (2026-08-11/12:
terrain renders correctly, goal marker confirmed on-platform, a real training run reached
5228/50000 steps with sane trend directions and one flagged numerical-instability anomaly early
on -- see project memory, not repeated here). AMP reference motion is still the bootstrap Strut
Walking clip (G1_strut_walk.npz) -- stairs-specific reference motion is a separate open task
(Dataset section). Terrain-aware rewards (feet_contact_sensor, rew_feet_height_error,
rew_edge_penetration below, added 2026-08-12) are new and NOT yet GPU-VM-verified. Everything
past that (reward/termination tuning proper, full training run) is not done.

**Not runnable/verified locally -- Isaac Sim isn't installed on this machine.** The terrain
geometry (STAIRS_TERRAIN_CFG below) and the env/terrain-origin wiring in g1_stairs_env.py's
_setup_scene ARE now GPU-VM-verified (see above). Still unverified: feet_contact_sensor's
prim_path/activate_contact_sensors correctness (see its own comment below for the specific past
bug class this is trying to avoid), and the terrain-height/edge-distance formulas'
step-index/quantization math (g1_stairs_env.py's _terrain_height_at / _edge_distance_at) --
these lean on STAIRS_TREAD_DEPTH_ACTUAL_M / STAIRS_RISER_HEIGHT_ACTUAL_M, themselves computed
from an *assumed* quantization rule (round-to-nearest-cell) that hasn't been checked against the
real generated mesh either. See STAIRS_TOTAL_RISE_M's comment for a related, still-open
discrepancy (measured env_origins Z of 1.05 vs. this file's own 0.89 assumption) that the new
terrain-height formula was deliberately designed to not depend on either way.
"""

from __future__ import annotations

import copy
import os

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import FlatPatchSamplingCfg, TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.terrains.height_field import HfPyramidStairsTerrainCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg

from isaaclab_assets import G1_29DOF_CFG

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")

# --- Procedural 5-step American stairs (project_description.md's Terrain section) ---
#
# Standard US residential stair code: 7 in (0.178 m) riser, 11 in (0.2794 m) tread depth.
# step_height_range is degenerate (min == max) rather than curriculum-scaled by difficulty --
# this project deliberately has one fixed terrain, not a difficulty curriculum (curriculum=False
# on the generator below too).
#
# Size derivation, from Isaac Lab's own pyramid-stairs heightfield algorithm (confirmed by
# reading InstinctLab's `perlin_pyramid_stairs_terrain`, which extends -- not reimplements --
# Isaac Lab's base algorithm, so the core stepping loop is the same): starting from a
# (size - 2*border_width) square, each step shrinks the remaining un-stepped region by
# 2*step_width (step_width removed from each side) until what's left is <= platform_width.
# Solving for the tile size that yields exactly 5 steps:
#   size = 2 * (NUM_STEPS * TREAD_DEPTH_M + BORDER_WIDTH_M + PLATFORM_WIDTH_M / 2)
# NOT yet confirmed against a real Isaac Lab run (see module docstring) -- heightfield pixel
# quantization (horizontal_scale below) could shift the exact step count by one in edge cases.
_STAIRS_RISER_HEIGHT_M = 0.178  # 7 in
_STAIRS_TREAD_DEPTH_M = 0.2794  # 11 in
_STAIRS_NUM_STEPS = 5
_STAIRS_BORDER_WIDTH_M = 0.5  # flat per-tile margin before the stairs begin (spawn zone)
_STAIRS_PLATFORM_WIDTH_M = 1.2  # flat center landing -- also this project's command-generation
# goal target (see project_description.md's Command Generation section / Task 3)
_STAIRS_TILE_SIZE_M = 2.0 * (
    _STAIRS_NUM_STEPS * _STAIRS_TREAD_DEPTH_M + _STAIRS_BORDER_WIDTH_M + _STAIRS_PLATFORM_WIDTH_M / 2.0
)
_STAIRS_HORIZONTAL_SCALE_M = 0.05  # heightfield cell size -- see STAIRS_TERRAIN_CFG's own
# horizontal_scale comment for why 5cm was chosen. Named here (not just inline on the cfg below)
# because the *actual* generated tread depth is this value quantized by horizontal_scale, needed
# by g1_stairs_env.py's edge-distance reward to match where risers really land, not just the
# pre-quantization target.
_STAIRS_VERTICAL_SCALE_M = 0.005  # heightfield height quantization -- see STAIRS_TERRAIN_CFG.

# Public (non-underscore) derived constants g1_stairs_env.py needs for border-ring spawn
# placement (Task 4 -- see g1_stairs_env.py's _sample_border_spawn_positions) and for the
# terrain-relative reward terms (feet height error / edge penetration, added after the 2026-08-12
# training run): Isaac Lab's pyramid-stairs sub-terrain `origin` is the CENTER PLATFORM (our
# command-generation goal, see STAIRS_TERRAIN_CFG's comment below), not the border/spawn ring.
STAIRS_TOTAL_RISE_M = _STAIRS_NUM_STEPS * _STAIRS_RISER_HEIGHT_M  # platform height above the
# border ring's flat ground level -- subtract this from env_origins' Z to get back to ground.
#
# ** OPEN DISCREPANCY, not yet resolved (found 2026-08-12 while deriving the terrain-height
# formula below): a real GPU-VM run measured `terrain.env_origins[:, 2] == 1.05`, not the 0.89
# this constant computes. The earlier "goal_z - spawn_z = 0.89 exactly, confirmed correct"
# claim (same session) was PARTLY CIRCULAR -- _sample_border_spawn_positions itself subtracts
# this exact constant from env_origins' Z, so that check could only ever reproduce whatever
# value this constant already has; it wasn't independent evidence the constant matches the real
# generated mesh. Root cause not identified (candidates: sub-terrain `origin` uses a different
# convention than "riser_height * num_steps" above the border ring; vertical_scale quantization
# compounding differently than expected; something about the outer grid border_width). NOT fixed
# here -- the new terrain-height formula below is deliberately written to avoid depending on
# this constant being correct (anchored to the live env_origins value instead, see
# g1_stairs_env.py's _terrain_height_at). The border-spawn code above still uses this constant
# unchanged (visually looked fine in the reviewed video, but that's not the same as confirmed
# correct) -- revisit on the next GPU-VM session, e.g. by directly inspecting the generated
# heightfield/mesh bounds rather than inferring from env_origins.
STAIRS_SPAWN_RADIUS_M = _STAIRS_TILE_SIZE_M / 2.0 - _STAIRS_BORDER_WIDTH_M / 2.0  # distance from
# tile center to the middle of the flat border ring, along whichever face's axis is chosen.
STAIRS_SPAWN_LATERAL_MARGIN_M = _STAIRS_TILE_SIZE_M / 2.0 - _STAIRS_BORDER_WIDTH_M  # half-width
# of the flat region along a chosen border face (the cross-axis sampling range); staying within
# this keeps the sampled point away from the (unverified, see module docstring) exact corner
# geometry where two faces' border regions meet.
STAIRS_STAIR_REGION_HALF_EXTENT_M = _STAIRS_TILE_SIZE_M / 2.0 - _STAIRS_BORDER_WIDTH_M  # Chebyshev
# distance from tile center to where the outermost (lowest) step begins -- the "d' = 0" datum
# used by both the edge-distance and terrain-height formulas below.
STAIRS_TREAD_DEPTH_ACTUAL_M = (
    round(_STAIRS_TREAD_DEPTH_M / _STAIRS_HORIZONTAL_SCALE_M) * _STAIRS_HORIZONTAL_SCALE_M
)  # as-generated tread depth after heightfield quantization: round(0.2794/0.05)*0.05 = 0.30 m,
# ~7% over the 11in target (see STAIRS_TERRAIN_CFG's horizontal_scale comment) -- riser edges in
# the real mesh sit at multiples of THIS value from STAIRS_STAIR_REGION_HALF_EXTENT_M, not the
# pre-quantization target, so this (not _STAIRS_TREAD_DEPTH_M) is what the edge-distance formula
# must use to match where edges actually are.
STAIRS_RISER_HEIGHT_ACTUAL_M = (
    round(_STAIRS_RISER_HEIGHT_M / _STAIRS_VERTICAL_SCALE_M) * _STAIRS_VERTICAL_SCALE_M
)  # as-generated riser height: round(0.178/0.005)*0.005 = 0.180 m, ~1% over target -- vertical
# quantization is far finer than horizontal here so this is close to the target, unlike tread
# depth above.
STAIRS_NUM_STEPS = _STAIRS_NUM_STEPS  # exposed for the terrain-height step-index clamp

STAIRS_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(_STAIRS_TILE_SIZE_M, _STAIRS_TILE_SIZE_M),
    border_width=2.0,  # flat safety margin around the whole generated grid (outside all tiles).
    # Confirmed via a real run on the GPU VM (2026-08-11) that heightfield-to-mesh generation is
    # CPU-bound and NOT fast at scale in this installed Isaac Lab version -- the original 5.0m
    # border + 2x4=8 tiles + 0.01m horizontal_scale produced a multi-million-cell heightfield
    # that didn't finish meshing in 15+ minutes (killed, not confirmed to ever finish). This is
    # a ~12x cell-count reduction from that (border 5.0->2.0, tiles 8->4, resolution 0.01->0.02).
    num_rows=2,
    num_cols=2,  # 4 identical tiles, purely so 4096 parallel envs spread visually across
    # multiple world-space regions instead of literally all overlapping at one tile (see
    # module docstring -- GPU per-env collision filtering makes envs sharing a tile physically
    # fine either way, but overlapping renders are useless for videos/eval). Same fixed
    # terrain repeated, not a curriculum (curriculum=False below).
    curriculum=False,
    horizontal_scale=_STAIRS_HORIZONTAL_SCALE_M,  # 5cm heightfield resolution -- matches the
    # reference paper's own choice (Appendix A.4) and Isaac Lab's common default, prioritizing
    # generation/mesh speed (confirmed CPU-bound and slow at finer resolutions on the actual GPU
    # VM, see above) over exact riser/tread dimensions. Tradeoff, not free: as-generated tread
    # depth is STAIRS_TREAD_DEPTH_ACTUAL_M = 0.30m (~7% over the 27.94cm target) -- a real,
    # visible deviation from literal standard-code dimensions, accepted here for iteration speed.
    # (An earlier version of this comment incorrectly attributed the riser-height quantization to
    # this field too -- riser height is quantized by vertical_scale below, not horizontal_scale;
    # see STAIRS_RISER_HEIGHT_ACTUAL_M, which is only ~1% off target, not ~12%.)
    vertical_scale=_STAIRS_VERTICAL_SCALE_M,  # matches the reference paper's own vertical
    # resolution (Appendix A.4); see STAIRS_RISER_HEIGHT_ACTUAL_M for the as-generated riser height.
    sub_terrains={
        "stairs": HfPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(_STAIRS_RISER_HEIGHT_M, _STAIRS_RISER_HEIGHT_M),
            step_width=_STAIRS_TREAD_DEPTH_M,
            platform_width=_STAIRS_PLATFORM_WIDTH_M,
            border_width=_STAIRS_BORDER_WIDTH_M,
            # NOTE: no `holes` kwarg -- confirmed via a real run on the GPU VM (2026-08-11) that
            # this installed Isaac Lab version's HfPyramidStairsTerrainCfg doesn't accept one
            # (TypeError: unexpected keyword argument 'holes'), unlike what the module docstring
            # originally assumed from reading InstinctLab's extended variant. No holes were ever
            # wanted here anyway (see this file's opening comment), so simply omitting it is the
            # correct fix, not a workaround.
            # Candidate flat-patch locations for Task 3's position-based command: this samples
            # ALL sufficiently-flat regions of the tile (both the outer border ring near z=0 and
            # the center plateau near z=NUM_STEPS*RISER_HEIGHT), not just the goal -- Task 3
            # distinguishes spawn vs. goal patches by height (the two clusters are well
            # separated: ~0 m vs ~0.89 m), not by this cfg. max_height_diff small enough to
            # reject any patch straddling a step edge.
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=16, patch_radius=0.4, max_height_diff=0.05),
            },
        ),
    },
)

# G1_29DOF_CFG ships mocap-tracking gains for the upper body (arms stiffness 3000, waist 5000
# -- position-hold values for kinematic replay, not RL). With noisy policy position targets
# (std 1.0 at action_scale 0.5 -> ~±0.5 rad) and a 300 N*m effort limit, gains that stiff turn
# the arms into a destabilizing torque source on the torso every step. Soften arms/waist to
# rl-wbc-g1-baseline's values (arms 40/10, torso 200/5) for parity; legs/feet keep this asset's
# own values, already in that project's ballpark.
# deepcopy rather than .replace(): configclass replace() copies shallowly for fields not being
# replaced, so mutating .actuators on a replace()'d copy would write through to the shared
# G1_29DOF_CFG instance.
_G1_29DOF_SOFT_ARMS_CFG = copy.deepcopy(G1_29DOF_CFG)
_G1_29DOF_SOFT_ARMS_CFG.prim_path = "/World/envs/env_.*/Robot"
_G1_29DOF_SOFT_ARMS_CFG.actuators["arms"].stiffness = 40.0
_G1_29DOF_SOFT_ARMS_CFG.actuators["arms"].damping = 10.0
_G1_29DOF_SOFT_ARMS_CFG.actuators["waist"].stiffness = 200.0
_G1_29DOF_SOFT_ARMS_CFG.actuators["waist"].damping = 5.0
# Required for the feet ContactSensor below to report anything (added 2026-08-12, for the feet
# height error / edge penetration reward terms) -- per rl-wbc-g1-amp-force's own hard-won lesson
# (project_description.md Task 6 notes), activate_contact_sensors must be set on the spawn cfg of
# a sensed body. Only set on the robot side here (not the terrain): that lesson was specifically
# about two DYNAMIC RigidObjects contacting each other (a box and a hand); our feet contact the
# terrain's static heightfield collider, which is a different case and may not need this on the
# terrain side too -- NOT verified against a real Isaac Lab run, first thing to check if the
# sensor reports zero contacts.
_G1_29DOF_SOFT_ARMS_CFG.spawn.activate_contact_sensors = True


@configclass
class G1StairsEnvCfg(DirectRLEnvCfg):
    """G1 stairs environment config: velocity-tracking task reward + AMP style reward."""

    # task reward (velocity tracking, mirrors rl-wbc-g1-baseline's track_lin_vel_xy_exp/
    # track_ang_vel_z_exp scales/shape, reimplemented in g1_stairs_env.py since this is a
    # direct-workflow env, not a manager-based one, so there's no reward-manager term to reuse)
    rew_lin_vel_xy = 1.0
    rew_ang_vel_z = 1.0  # matches stock G1FlatEnvCfg's override (G1Rewards' own default is 2.0)
    # this is std**2, not std itself — Isaac Lab's own track_*_exp mdp functions take `std` and
    # divide error by std**2 internally
    rew_track_sigma = 0.25

    # regularization — matches rl-wbc-g1-baseline's stock g1_flat run (the full reward set).
    # feet_air_time and feet_slide are still NOT included (both would need the ContactSensor's
    # air-time tracking, not just its contact force) -- deferred as a follow-up, not dropped for
    # a design reason.
    rew_termination = -200.0
    rew_action_rate_l2 = -0.005
    rew_joint_pos_limits = -1.0  # ankle pitch/roll only
    rew_joint_acc_l2 = -1.0e-7  # hip + knee only
    rew_dof_torques_l2 = -2.0e-6  # hip + knee only
    rew_flat_orientation_l2 = -1.0
    rew_lin_vel_z_l2 = -0.2
    rew_ang_vel_xy_l2 = -0.05
    rew_joint_deviation_hip = -0.1  # hip_yaw + hip_roll only (hip_pitch drives the actual gait)
    rew_joint_deviation_arms = -0.1  # shoulder pitch/roll/yaw + elbow (single-DOF here, not pitch+roll)
    rew_joint_deviation_torso = -0.1  # all 3 waist joints
    # joint_deviation_fingers has no equivalent here: fingers aren't in action_dof_indexes, so
    # they never move from default_joint_pos regardless -- the term would always be exactly 0.

    # Terrain-aware regularization (added 2026-08-12, adapted from the paper's Table IV -- see
    # project_description.md's discussion of what's stair-specific vs. generic). Both need
    # feet_contact_sensor below. Neither is a literal port of the paper's formula -- both are
    # ADAPTED to use our own analytically-known stair geometry (Chebyshev-distance formulas in
    # g1_stairs_env.py's _terrain_height_at / _edge_distance_at) rather than the paper's
    # generic mesh-based Terrain Edge Detector + Volume Points, which we don't need since we
    # generated this terrain ourselves. Weights are starting-point guesses (loosely anchored to
    # the paper's own -0.1 / -4.0 relative magnitudes, not validated against our reward scale).
    #
    # Feet Height Error: penalizes a foot being higher than the local stair-tread surface by more
    # than feet_height_margin_m at the moment of contact -- paper's own reasoning (Table IV) is
    # this discourages landing partway up a riser face/edge rather than flat on a tread; contact
    # gating matters (an ungated version would also fight normal swing-phase foot lift).
    rew_feet_height_error = -0.1
    feet_height_margin_m = 0.035  # paper's own value (Table IV formula), not re-derived for G1
    feet_height_clip_max_m = 0.3  # paper's own value, caps a single bad landing's penalty

    # Edge Penetration (paper's r_vol, Eq. 7): penalizes a foot being close to a riser edge while
    # in contact, scaled by the foot's own velocity (a fast/careless contact near an edge is
    # penalized more than a slow, controlled one) -- adapted here as a planar edge-distance
    # threshold rather than the paper's true 3D volume-point penetration depth (see
    # g1_stairs_env.py's _edge_distance_at docstring for why that's a reasonable substitution
    # given we know the terrain analytically).
    rew_edge_penetration = -4.0
    edge_safety_margin_m = 0.05  # foot must be at least this far (planar) from a riser line while
    # in contact to avoid this penalty -- not from the paper (their Volume Points use the robot's
    # actual foot collision geometry, not a hand-picked margin), a starting guess.

    # Anti-dawdling terms (added 2026-08-13, ported verbatim from InstinctLab's real reward
    # functions -- reference/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/mdp/
    # rewards.py:38-52 (stand_still) and :101-110 (dont_wait), not re-derived -- to counter the
    # "camps at the base of the stairs to avoid rew_termination" failure mode found via video
    # review of the 2026-08-12/13 training run (see instinctlab_comparison.md / project memory).
    # Weights/offsets/thresholds below are InstinctLab's own real config values, not guesses.
    rew_dont_wait = -0.5  # penalizes standing/moving backward while commanded to move forward
    dont_wait_cmd_threshold_mps = 0.3  # only applies once commands[:,0] exceeds this
    rew_stand_still = -0.3  # penalizes joint deviation from default pose while commanded to stand
    stand_still_offset = 4.0  # allowed slack (sum |joint_pos - default|, rad) before this bites
    stand_still_cmd_threshold = 0.15  # both lin/ang command must be below this to count as "standing"

    # Height-progress diagnostic (added 2026-08-14): root height relative to the goal/center
    # platform (0 at goal, negative below it). Default weight is deliberately 0.0 -- this exists
    # purely so mean height can be watched in TensorBoard (reward_log's diag_mean_height_rel_to_goal,
    # always logged unweighted regardless of this value) as a direct "how far up the stairs" signal,
    # without changing training behavior. Raise above 0.0 later if height-shaping the reward
    # (on top of the existing velocity-tracking task reward) turns out to help; not yet tried.
    rew_max_height = 0.0

    # Foot contact sensor (added 2026-08-12, prerequisite for both terrain-aware terms above and
    # for a future feet_air_time/feet_slide addition). NOT verified against a real Isaac Lab
    # install -- prim_path pattern and activate_contact_sensors (see _G1_29DOF_SOFT_ARMS_CFG
    # above) follow rl-wbc-g1-amp-force's own hard-won lesson (project_description.md Task 6:
    # exact body path had to be confirmed against a live USD stage dump there, not assumed) but
    # this project hasn't done that same confirmation yet -- first thing to check on the next
    # GPU-VM session if the sensor reports zero contacts.
    feet_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*_ankle_roll_link",
        history_length=3,
        track_air_time=True,
    )

    # Position-based, direction-biased velocity command (project_description.md's Command
    # Generation section; paper's Eq. 8-9): v_x = clip(k_v * x_g, 0, v_max),
    # w_z = clip(k_w * atan2(y_g, x_g), -w_max, w_max), where (x_g, y_g) is the goal position
    # in the robot's yaw-heading frame, recomputed every step (not resampled-and-held) so the
    # command continuously drives the robot toward the goal and tapers as it approaches. v_y is
    # fixed at 0 -- forward-facing camera only, matching the paper's own stated limitation.
    # Goal itself (self.goal_pos_w in g1_stairs_env.py) is resampled every command_resample_time_s
    # and is always the terrain tile's center plateau (self.terrain.env_origins) -- the
    # "difficult direction" is always through the stairs, since that's the only terrain feature,
    # unlike the paper's multi-terrain flat-patch selection (see g1_stairs_env_cfg.py's
    # STAIRS_TERRAIN_CFG comment on why the platform IS the tile's Isaac Lab `origin`).
    # k_v/k_w are NOT from the paper (not numerically specified there, only described as
    # "linear and angular stiffness coefficients") -- these are starting-point guesses, not yet
    # tuned against real training (see project_description.md's scaffolding-status notes).
    command_pos_k_v = 0.5  # 1/s
    command_pos_k_w = 2.0  # unitless (rad output per rad of bearing error)
    command_lin_vel_x_max = 1.0  # m/s
    command_ang_vel_z_max = 1.0  # rad/s
    command_resample_time_s = 10.0

    # debug visualization: draws a marker at each env's goal_pos_w (the position-based command's
    # target -- see g1_stairs_env.py's _update_position_based_commands) every step. Off by
    # default (real cost at num_envs=4096, and no visual point to it during headless training);
    # on in G1StairsEnvCfg_PLAY below for eval/video runs.
    debug_vis_goal: bool = False

    # env
    episode_length_s = 10.0
    decimation = 2

    # spaces — 71 task-obs dims (29 dof_pos + 29 dof_vel + 1 root height + 6 tangent/normal
    # + 3 root lin vel + 3 root ang vel) + 3*10 key-body relative positions + 3 velocity command
    observation_space = 71 + 3 * 10 + 3
    action_space = 29
    state_space = 0
    num_amp_observations = 2
    amp_observation_space = 71 + 3 * 10  # AMP obs stays style-only, no command — see g1_stairs_env.py

    # target = default_joint_pos + action_scale * action (JointPositionAction, scale=0.5,
    # use_default_offset=true) -- NOT the original AMP reference implementation's
    # "joint_limit_midpoint + full_joint_range * action" convention (see the note in
    # g1_stairs_env.py's __init__ for why that convention was the likely root cause of episodes
    # reliably dying within ~6-8 steps in rl-wbc-g1-amp's own debugging history).
    action_scale = 0.5

    early_termination = True
    termination_height = 0.5

    motion_file: str = os.path.join(MOTIONS_DIR, "G1_strut_walk.npz")
    """Bootstrap placeholder AMP reference motion -- rl-wbc-g1-amp's Strut Walking clip, not
    stairs-specific. Replace once stairs reference motion is sourced/converted (open task, see
    project_description.md's Dataset section)."""
    reference_body = "pelvis"
    reset_strategy = "random"  # default, random, random-start
    """Strategy to be followed when resetting each environment (humanoid's pose and joint states).

    * default: pose and joint states are set to the initial state of the asset.
    * random: pose and joint states are set by sampling motions at random, uniform times.
    * random-start: pose and joint states are set by sampling motion at the start (time zero).
    """

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # terrain — procedural 5-step American stairs (STAIRS_TERRAIN_CFG above). num_envs/
    # env_spacing get filled in from `scene` at runtime in g1_stairs_env.py's _setup_scene,
    # matching Isaac Lab's own direct-workflow rough-terrain pattern -- NOT
    # InteractiveSceneCfg's own nested `terrain` field, which this repo's DirectRLEnv-based
    # _setup_scene doesn't use (it builds the scene by hand, same as it hand-builds the robot
    # Articulation below, rather than relying on InteractiveScene's automatic entity wiring).
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=STAIRS_TERRAIN_CFG,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
        debug_vis=False,
    )

    # robot — Isaac Lab's own G1_29DOF_CFG (matching our retargeted motion data's joint
    # convention; see module docstring), with arm/waist actuator gains softened to
    # rl-wbc-g1-baseline's values (see _G1_29DOF_SOFT_ARMS_CFG).
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
        # front-on, further back: camera sits ahead of the robot along its direction of
        # travel (+X, the clip's own heading), so it walks toward camera. lookat is deliberately
        # low (near ankle height, not chest height) -- aiming at the robot's center pushes the
        # feet to the bottom edge of frame since the robot has much more height above the hips
        # than below them; aiming low keeps feet clear with margin at the cost of some empty
        # headroom above the head, the safer direction to err in.
        # TODO(cosmetic, not yet done): the robot now spawns on a randomly-chosen one of the
        # pyramid terrain's 4 faces (g1_stairs_env.py's _sample_border_spawn_positions), not
        # always walking in +X, so this fixed front-on eye/lookat won't track env 0's robot
        # correctly for 3 out of 4 spawn faces. Revisit once eval videos are actually produced.
        self.viewer.eye = (4.5, 0.0, 0.85)
        self.viewer.lookat = (0.0, 0.0, 0.35)
