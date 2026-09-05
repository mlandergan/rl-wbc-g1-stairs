"""Vision-based AMP stair climbing environment — forked from rl-wbc-g1-amp's g1_amp_env.py,
which is itself adapted from linden713/humanoid_amp's G1AmpEnv (BSD-3-Clause, pure imitation,
all task reward scales zero, no command).

Inherited unchanged from that fork:
  - the command appended to the *policy* observation only — the AMP observation buffer
    stays exactly the reference's style-only features, so the discriminator never sees
    the command and can't use it to shortcut style scoring
  - a track_lin_vel_xy_exp / track_ang_vel_z_exp task reward (reimplemented directly since
    this is a direct-workflow env with no reward manager)

Changed from that fork (see project_description.md's Initial Tasks):
  - Task 2 (done): _setup_scene now imports a procedural stairs TerrainImporter
    (g1_stairs_env_cfg.py's STAIRS_TERRAIN_CFG) instead of a flat GroundPlaneCfg.
  - Task 3 (done): self.commands is no longer sampled uniformly and held for
    command_resample_time_s. It's now DERIVED every step from a persistent per-env goal
    (self.goal_pos_w) and the live robot pose, via the paper's Eq. 8-9 position-based command
    (_update_position_based_commands). _resample_commands now only resamples the goal (always
    the terrain tile's center plateau -- see its docstring), not the command directly.
  - Task 4 (done): reset logic (_reset_strategy_default / _reset_strategy_random) spawns robots
    on the stairs tile's flat border ring (_sample_border_spawn_positions), not at the terrain
    tile's own Isaac Lab `origin` (which is the center platform / this env's command goal, not
    a spawn point -- see g1_stairs_env_cfg.py's STAIRS_TERRAIN_CFG comment). This is what makes
    every episode actually start at the bottom of the stairs and walk up toward the goal.
  - Debug visualization (2026-08-12, added after the terrain/command/spawn geometry was
    verified correct on the GPU VM): a green sphere marker (GOAL_MARKER_CFG) drawn at each
    env's goal_pos_w every step, gated by cfg.debug_vis_goal (off by default, on in
    G1StairsEnvCfg_PLAY). VERIFIED on the GPU VM (2026-08-12): marker renders exactly on each
    tile's center platform, matching the numeric goal verification.
  - Terrain-aware rewards (2026-08-12, after a training run reached step 5228/50000 cleanly --
    see project_description.md's reward-comparison discussion for why these two specifically):
    a feet ContactSensor (self.contact_sensor, cfg.feet_contact_sensor) plus two new reward
    terms in _get_rewards, rew_feet_height_error and rew_edge_penetration, both ADAPTED from
    the paper's Table IV (not literal ports -- see _terrain_height_at / _edge_distance_at's
    docstrings) using this project's own analytically-known stairs geometry rather than the
    paper's generic mesh-based Terrain Edge Detector + Volume Points. NOT yet run on the GPU
    VM -- the ContactSensor body-name resolution, activate_contact_sensors requirement, and the
    terrain-height formula's step-index math are all new and unverified (see
    g1_stairs_env_cfg.py's feet_contact_sensor comment for the specific past bug class this is
    trying to avoid by resolving body names explicitly rather than assuming index alignment).
  - Spawn heading alignment (2026-08-13, after reviewing a full training run's rollout video --
    robot started ~90 degrees off-heading on most spawns and never learned to walk): reset now
    rotates the reference clip's sampled root orientation and linear/angular velocity by a
    per-env yaw delta so the clip's forward direction points from the spawn face toward the goal,
    instead of always the clip's fixed +X heading. See _reset_strategy_random's inline comment.
  - AMP frame fix + anti-dawdling rewards (2026-08-13, "Finding B" from the same video-review
    deep-dive, confirmed via a direct code comparison against InstinctLab -- see
    instinctlab_comparison.md): compute_obs now heading-relativizes velocity/orientation/key-body
    features instead of using raw world-frame quantities (see its own docstring), so the AMP
    discriminator judges gait style independent of which of the 4 stairs faces the robot is
    approaching from. Also added rew_dont_wait/rew_stand_still (_get_rewards), ported verbatim
    from InstinctLab's real reward functions, to counter the "camps at the base to avoid
    rew_termination" pattern seen in that same video review -- a separate, compounding cause from
    the frame bug, not fixed by it. VERIFIED on the GPU VM (2026-08-13): checkpoint video at step
    13000/20000 showed real goal-reaching (robots on the top platform, not just camping at the
    base), a qualitatively different and better pattern than pre-fix runs.
  - Height-progress diagnostic (2026-08-14): rew_max_height (default weight 0.0, no training
    effect) plus an always-logged raw diag_mean_height_rel_to_goal scalar in _get_rewards, so
    mean climbing progress can be watched directly in TensorBoard regardless of reward weighting.
    See g1_stairs_env_cfg.py's comment for details.

NOTE: the task-reward terms use *base-frame* velocity (root_lin_vel_b/root_ang_vel_b) —
what the command means. The AMP style features now ALSO use base-frame velocity (plus
yaw-relativized orientation/key-body features) as of the 2026-08-13 fix above -- previously they
used raw *world-frame* velocity (body_lin_vel_w/body_ang_vel_w), unchanged from the reference
implementation, which was fine there (single-direction flat-ground task) but fought goal-directed
motion on 3 of 4 approach faces here. See convert_gmr_to_npz.py's note on why the reference clip
itself is still recorded +X-aligned -- that's no longer load-bearing for AMP correctness now that
compute_obs relativizes both the rollout and reference sides symmetrically, but it's harmless to
leave as-is.
"""

from __future__ import annotations

import re

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import ContactSensor, RayCasterCamera
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.math import (
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from .g1_stairs_env_cfg import (
    STAIRS_NUM_STEPS,
    STAIRS_SPAWN_LATERAL_MARGIN_M,
    STAIRS_SPAWN_RADIUS_M,
    STAIRS_STAIR_REGION_HALF_EXTENT_M,
    STAIRS_TOTAL_RISE_M,
    STAIRS_TREAD_DEPTH_ACTUAL_M,
    STAIRS_RISER_HEIGHT_ACTUAL_M,
    G1StairsEnvCfg,
)
from .motions import MotionLoader

# Debug-vis marker for the position-based command's goal (self.goal_pos_w) -- always the
# terrain tile's center plateau, see _resample_commands' docstring. Off by default
# (G1StairsEnvCfg.debug_vis_goal=False), on in the _PLAY cfg for eval/video runs.
GOAL_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Command/stairs_goal",
    markers={
        "goal": sim_utils.SphereCfg(
            radius=0.12,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 1.0, 0.2)),
        ),
    },
)


class G1StairsEnv(DirectRLEnv):
    cfg: G1StairsEnvCfg

    def __init__(self, cfg: G1StairsEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # fixed depth-blur kernel for _process_depth_image, precomputed once (not per-step) --
        # a standard discretized isotropic Gaussian, matching InstinctLab's own
        # kernel_size=3/sigma=1 blur parameters (WORKING_NOTES.md). (1, 1, k, k) shape for
        # torch.nn.functional.conv2d's (out_channels, in_channels/groups, kH, kW) convention.
        k, sigma = self.cfg.DEPTH_BLUR_KERNEL_SIZE, self.cfg.DEPTH_BLUR_SIGMA
        coords = torch.arange(k, dtype=torch.float32, device=self.device) - (k - 1) / 2.0
        gauss_1d = torch.exp(-(coords**2) / (2 * sigma**2))
        gauss_2d = torch.outer(gauss_1d, gauss_1d)
        gauss_2d = gauss_2d / gauss_2d.sum()
        self._depth_blur_kernel = gauss_2d.view(1, 1, k, k)

        # load motion
        self._motion_loader = MotionLoader(motion_file=self.cfg.motion_file, device=self.device)

        # G1_29DOF_CFG has 43 joints (29 "real" body joints + 14 finger joints); our motion
        # data only covers the 29 real ones (retargeted with no finger animation). action_
        # dof_indexes is where those 29 land in the robot's full 43-joint ordering -- fingers
        # are excluded from the action space entirely and held at their default pose (see
        # _apply_action/_reset_strategy_random), since nothing here cares about finger motion.
        motion_joint_names = [n for n in self.robot.data.joint_names if n in self._motion_loader.dof_names]
        assert len(motion_joint_names) == self.cfg.action_space, (
            f"Expected {self.cfg.action_space} robot joints covered by the motion file, "
            f"found {len(motion_joint_names)}: {motion_joint_names}"
        )
        self.action_dof_indexes = [self.robot.data.joint_names.index(n) for n in motion_joint_names]
        self.motion_dof_indexes = self._motion_loader.get_dof_index(motion_joint_names)

        # joint-group index subsets (positions *within* the 29-long action_dof_indexes slice, not
        # into the robot's full 43-joint space) for the regularization terms below that only apply
        # to specific joint groups, matching Project 1's asset_cfg(joint_names=...) restrictions.
        def _match(patterns: list[str]) -> list[int]:
            return [i for i, n in enumerate(motion_joint_names) if any(re.fullmatch(p, n) for p in patterns)]

        self.hip_knee_dof_indexes = _match([r".*_hip_.*", r".*_knee_joint"])
        self.ankle_dof_indexes = _match([r".*_ankle_pitch_joint", r".*_ankle_roll_joint"])
        self.hip_deviation_dof_indexes = _match([r".*_hip_yaw_joint", r".*_hip_roll_joint"])
        self.arm_deviation_dof_indexes = _match([
            r".*_shoulder_pitch_joint", r".*_shoulder_roll_joint", r".*_shoulder_yaw_joint", r".*_elbow_joint",
        ])
        self.torso_deviation_dof_indexes = _match([r"waist_yaw_joint", r"waist_roll_joint", r"waist_pitch_joint"])

        # NOTE: no per-joint action_offset/action_scale tensors here (that was the original
        # AMP-reference-implementation convention: target = joint_limit_midpoint + full_joint_range
        # * action). Project 1's actual action space is target = default_joint_pos + 0.5 * action
        # (a small, bounded nudge from the robot's own standing pose, not a swing across the whole
        # range of motion centered on the joint-limit midpoint) -- see cfg.action_scale and
        # _apply_action. The full-range version was traced as the likely root cause of episodes
        # reliably dying within ~6-8 steps across every reward/PPO-hyperparameter variant tried:
        # even a single std of Gaussian action noise could swing a PD target across a joint's
        # entire range, untethered from the actual current pose, regardless of policy quality.

        # DOF and key body indexes
        key_body_names = [
            "left_shoulder_pitch_link", "right_shoulder_pitch_link",
            "left_elbow_link", "right_elbow_link",
            "right_hip_yaw_link", "left_hip_yaw_link",
            "right_hand_palm_link", "left_hand_palm_link",
            "right_ankle_roll_link", "left_ankle_roll_link",
        ]

        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in key_body_names]
        self.motion_ref_body_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        self.motion_key_body_indexes = self._motion_loader.get_body_index(key_body_names)

        # Feet body indexes, in TWO SEPARATE index spaces that are not guaranteed to agree (added
        # 2026-08-12, for the feet-height-error / edge-penetration reward terms): self.robot's own
        # body_names order (for position/velocity queries via self.robot.data.*) vs.
        # self.contact_sensor's own discovered body order (for force queries via
        # self.contact_sensor.data.*, populated by matching feet_contact_sensor's prim_path regex
        # against the live USD stage -- not guaranteed to match self.robot.data.body_names' order).
        # Resolving both explicitly by name here, rather than assuming index alignment, is
        # deliberately paranoid: rl-wbc-g1-amp-force hit a real bug from exactly this kind of
        # assumption (wrong link path resolved to a body with zero collision geometry, silently
        # sensed nothing -- see project_description.md's Task 6 notes). NOT verified against a
        # real Isaac Lab install; first thing to check on the next GPU-VM session if
        # self.contact_sensor.body_names doesn't contain what's expected here.
        feet_body_names = ["right_ankle_roll_link", "left_ankle_roll_link"]
        self.feet_body_indexes = [self.robot.data.body_names.index(name) for name in feet_body_names]
        self.feet_contact_body_indexes = [self.contact_sensor.body_names.index(name) for name in feet_body_names]

        # reconfigure AMP observation space according to the number of observations and create the buffer
        # (style-only — the velocity command is never part of this, see module docstring)
        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        )

        # position-based velocity command (paper's Eq. 8-9, see g1_stairs_env_cfg.py's module
        # docstring): goal_pos_w is the persistent per-env world-space target, resampled every
        # command_resample_time_s in _resample_commands; commands (lin_vel_x, lin_vel_y,
        # ang_vel_z, robot base frame) are DERIVED from goal_pos_w + the live robot pose every
        # step in _update_position_based_commands, not resampled-and-held.
        self.commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_marker = VisualizationMarkers(GOAL_MARKER_CFG) if self.cfg.debug_vis_goal else None
        self._resample_commands(torch.arange(self.num_envs, device=self.device))
        self._update_position_based_commands()

        # action buffers for action_rate_l2 (penalizes actions changing too fast between steps,
        # matching Project 1's mdp.action_rate_l2 term -- there's no action manager here to hold
        # this for us since this is a direct-workflow env, so it's tracked by hand)
        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)

        # procedural stairs terrain (g1_stairs_env_cfg.py's STAIRS_TERRAIN_CFG) — replaces the
        # flat GroundPlaneCfg this env forked from. num_envs/env_spacing filled in from the
        # scene cfg, matching Isaac Lab's own direct-workflow rough-terrain pattern (e.g. its
        # Anymal-C/Go2/H1 rough envs). NOT verified against a real Isaac Lab install -- see
        # g1_stairs_env_cfg.py's module docstring.
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = TerrainImporter(self.cfg.terrain)

        # feet contact sensor (added 2026-08-12, for the feet-height-error / edge-penetration
        # reward terms below) -- NOT verified against a real Isaac Lab install, see
        # g1_stairs_env_cfg.py's feet_contact_sensor comment.
        self.contact_sensor = ContactSensor(self.cfg.feet_contact_sensor)
        self.scene.sensors["feet_contact_sensor"] = self.contact_sensor

        # depth camera (WORKING_NOTES.md's "Exact camera configuration") -- a RayCasterCamera,
        # so this raycasts against the mesh_prim_paths given in cfg.depth_camera ("/World/ground",
        # which after clone_environments below resolves per-env) rather than going through the
        # RTX render pipeline. NOT verified against a real Isaac Lab install -- first thing to
        # check on the next GPU-VM session if _process_depth_image ever sees an all-zero/all-inf
        # image (see that method's own docstring for the specific output-shape assumption at risk).
        self.depth_camera = RayCasterCamera(self.cfg.depth_camera)
        self.scene.sensors["depth_camera"] = self.depth_camera

        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone()

        # resample the goal for envs whose window has elapsed (mid-episode, in general; with the
        # default config command_resample_time_s == episode_length_s so this only actually
        # fires on reset, but stays correct if the two are ever set independently). Currently a
        # no-op in practice -- every env's goal is always the same center plateau (see
        # _resample_commands' docstring) -- kept as a hook for a future multi-goal design
        # (e.g. alternating ascent/descent) rather than deleted as dead code.
        resample_steps = int(self.cfg.command_resample_time_s / (self.cfg.sim.dt * self.cfg.decimation))
        due = (self.episode_length_buf % max(resample_steps, 1) == 0) & (self.episode_length_buf > 0)
        env_ids = due.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self._resample_commands(env_ids)

    def _resample_commands(self, env_ids: torch.Tensor):
        """Resample the per-env GOAL, not the velocity command directly (contrast with
        rl-wbc-g1-amp's uniform-velocity version of this method). The goal is always the
        terrain tile's center plateau -- `self.terrain.env_origins` already resolves to that
        point in world coordinates per env, by construction of Isaac Lab's pyramid-stairs
        sub-terrain `origin` convention (see g1_stairs_env_cfg.py's STAIRS_TERRAIN_CFG comment)
        -- so there's no goal *selection* to make (unlike the paper's multi-terrain flat-patch
        choice): the difficult direction is always through the stairs, since that's the only
        terrain feature. self.commands itself is derived from this goal every step in
        _update_position_based_commands, not here.
        """
        if len(env_ids) == 0:
            return
        self.goal_pos_w[env_ids] = self.terrain.env_origins[env_ids]

    def _update_position_based_commands(self):
        """Paper's Eq. 8-9: v_x = clip(k_v * x_g, 0, v_max), w_z = clip(k_w * atan2(y_g, x_g),
        -w_max, w_max), where (x_g, y_g) is the goal position in the robot's yaw-heading frame.
        Recomputed for ALL envs every step (cheap, and simpler/more obviously correct than only
        updating just-resampled envs) so the command continuously tracks the live goal-relative
        position as the robot moves -- this is what makes it drive the robot toward the goal and
        taper as it approaches, unlike a fixed command held for the whole resample window.
        v_y stays fixed at 0 (forward-facing camera only, matching the paper's own limitation).
        """
        r = self.cfg
        goal_vec_w = self.goal_pos_w - self.robot.data.root_pos_w
        heading_w = yaw_quat(self.robot.data.root_quat_w)
        goal_vec_b = quat_apply_inverse(heading_w, goal_vec_w)
        x_g, y_g = goal_vec_b[:, 0], goal_vec_b[:, 1]

        self.commands[:, 0] = torch.clamp(r.command_pos_k_v * x_g, min=0.0, max=r.command_lin_vel_x_max)
        self.commands[:, 1] = 0.0
        self.commands[:, 2] = torch.clamp(
            r.command_pos_k_w * torch.atan2(y_g, x_g), min=-r.command_ang_vel_z_max, max=r.command_ang_vel_z_max
        )

        if self.goal_marker is not None:
            self.goal_marker.visualize(translations=self.goal_pos_w)

    def _terrain_height_at(self, pos_w: torch.Tensor) -> torch.Tensor:
        """Analytic local terrain height under an arbitrary world-space position, for this
        project's known pyramid-stairs geometry (added 2026-08-12, for the feet-height-error
        reward). pos_w: (num_envs, ..., 3) -- only XY used, any number of extra dims (e.g. a
        per-foot dim) between the leading num_envs dim and the trailing XYZ dim. Returns
        (num_envs, ...) heights.

        Deliberately anchored to the LIVE self.terrain.env_origins Z value (platform height),
        not an assumed border-ground height -- see g1_stairs_env_cfg.py's STAIRS_TOTAL_RISE_M
        comment for the open discrepancy (measured env_origins Z of 1.05 vs. this project's own
        0.89 assumption) this sidesteps by construction: height is computed as "platform height
        minus N risers," never as "border height plus N risers," so it's correct regardless of
        what the true border height turns out to be once that's resolved.
        """
        origin = self.terrain.env_origins  # (num_envs, 3)
        extra_dims = pos_w.dim() - 2
        view_shape = (origin.shape[0],) + (1,) * extra_dims
        origin_x = origin[:, 0].view(view_shape)
        origin_y = origin[:, 1].view(view_shape)
        origin_z = origin[:, 2].view(view_shape)

        # Chebyshev (L-infinity) distance from tile center -- the same metric the pyramid-stairs
        # generator itself uses to carve nested-square steps (see STAIRS_TERRAIN_CFG's derivation
        # comment), so this is exact for our geometry, not an approximation.
        d = torch.maximum(torch.abs(pos_w[..., 0] - origin_x), torch.abs(pos_w[..., 1] - origin_y))
        d_prime = STAIRS_STAIR_REGION_HALF_EXTENT_M - d  # distance INTO the stair region
        step_index = torch.clamp(
            torch.floor(d_prime / STAIRS_TREAD_DEPTH_ACTUAL_M) + 1.0, min=0.0, max=float(STAIRS_NUM_STEPS)
        )
        height_below_platform = (STAIRS_NUM_STEPS - step_index) * STAIRS_RISER_HEIGHT_ACTUAL_M
        return origin_z - height_below_platform

    def _edge_distance_at(self, pos_w: torch.Tensor) -> torch.Tensor:
        """Analytic planar distance from an arbitrary world-space position to the nearest stair
        riser edge line (added 2026-08-12, for the edge-penetration reward). pos_w: same shape
        convention as _terrain_height_at.

        This is our substitute for the paper's Terrain Edge Detector (Section III-C, Algorithms
        2-3): they scan an arbitrary triangle mesh for edges via dihedral-angle thresholding
        because they don't control the terrain generation; we generated ours from known
        parameters, so edge locations are closed-form algebra instead -- riser lines sit at
        exact multiples of STAIRS_TREAD_DEPTH_ACTUAL_M out from the tile's stair-region boundary,
        in the same Chebyshev metric the generator itself uses. Also a substitute for their
        Volume Points (a set of sample points covering the foot's actual collision geometry): we
        use a single point per foot (its body origin) rather than a multi-point volume, since we
        don't have per-point contact/penetration data without building that machinery -- a
        reasonable simplification given the foot is small relative to a 30cm tread, not a
        literal port of Eq. 7.
        """
        origin = self.terrain.env_origins
        extra_dims = pos_w.dim() - 2
        view_shape = (origin.shape[0],) + (1,) * extra_dims
        origin_x = origin[:, 0].view(view_shape)
        origin_y = origin[:, 1].view(view_shape)

        d = torch.maximum(torch.abs(pos_w[..., 0] - origin_x), torch.abs(pos_w[..., 1] - origin_y))
        d_prime = STAIRS_STAIR_REGION_HALF_EXTENT_M - d

        k = torch.arange(STAIRS_NUM_STEPS + 1, device=pos_w.device, dtype=pos_w.dtype)
        edge_positions = k * STAIRS_TREAD_DEPTH_ACTUAL_M  # (num_steps+1,) -- one per riser line
        dist_to_each_edge = torch.abs(d_prime.unsqueeze(-1) - edge_positions)
        return dist_to_each_edge.min(dim=-1).values

    def _apply_action(self):
        # fingers (not in action_dof_indexes) hold their default pose; only the 29
        # motion-covered joints are actually driven by the policy's actions, as a small nudge
        # away from the robot's own default/standing pose -- matches Project 1's actual action
        # space (JointPositionAction, scale=0.5, use_default_offset=true), not the original AMP
        # reference implementation's "swing across the whole joint range" convention (see the
        # note in __init__ for why that was likely the real root cause of instant falls).
        target = self.robot.data.default_joint_pos.clone()
        target[:, self.action_dof_indexes] += self.cfg.action_scale * self.actions
        self.robot.set_joint_position_target(target)

    def _get_observations(self) -> dict:
        # style-only AMP observation, unchanged from the reference implementation except
        # restricted to the 29 motion-covered joints (not the full 43 incl. fingers) so
        # this stays dimensionally consistent with the reference motion's own 29-dim data
        amp_obs = compute_obs(
            self.robot.data.joint_pos[:, self.action_dof_indexes],
            self.robot.data.joint_vel[:, self.action_dof_indexes],
            self.robot.data.body_pos_w[:, self.ref_body_index],
            self.robot.data.body_quat_w[:, self.ref_body_index],
            self.robot.data.body_lin_vel_w[:, self.ref_body_index],
            self.robot.data.body_ang_vel_w[:, self.ref_body_index],
            self.robot.data.body_pos_w[:, self.key_body_indexes],
        )

        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        self.amp_observation_buffer[:, 0] = amp_obs.clone()
        # _get_rewards() runs before _get_observations() in DirectRLEnv.step() and sets
        # self.extras["log"] for the reward breakdown -- update here, don't reassign
        # self.extras wholesale, or that breakdown gets silently wiped out before it ever
        # reaches the trainer's info dict (confirmed: this is exactly what was happening).
        self.extras["amp_obs"] = self.amp_observation_buffer.view(-1, self.amp_observation_size)

        # policy observation = style obs + velocity command + flattened depth image, so the
        # policy can condition its actions on both what it's being asked to do AND what's ahead
        # of it. The discriminator above never sees the command OR the depth image -- confirmed
        # against InstinctLab's own actual discriminator inputs (WORKING_NOTES.md), it judges
        # motion style only. Flat concatenation, not a nested Dict -- DepthAmpPolicy/DepthAmpValue
        # (models.py) slice this back apart in their own compute() (see WORKING_NOTES.md's "flat
        # single policy Box" decision).
        depth_obs = self._process_depth_image().view(self.num_envs, -1)
        policy_obs = torch.cat((amp_obs, self.commands, depth_obs), dim=-1)
        return {"policy": policy_obs}

    def _process_depth_image(self) -> torch.Tensor:
        """Crop/blur/normalize the raw depth camera output into the (N, 16, 16) tensor
        DepthAmpPolicy/DepthAmpValue's Conv2d branch expects -- InstinctLab's own exact
        crop/blur/normalize parameters (WORKING_NOTES.md), reimplemented directly here rather
        than importing their `Noisy*` sensor classes (their own CC BY-NC source).

        NOT verified against a real Isaac Lab install -- specifically the assumption that
        `self.depth_camera.data.output["distance_to_image_plane"]` has shape
        (N, height, width, 1), matching Isaac Lab's other camera sensor classes'
        documented convention (confirmed for TiledCamera via Isaac Lab's own stock
        Cartpole-Depth-Camera example; RayCasterCamera specifically not independently
        confirmed). If this is actually (N, height, width) with no trailing channel dim, the
        `squeeze(-1)` below is a no-op and this still works either way -- but if the height/width
        axis order is swapped, the crop below silently reads the wrong region rather than
        erroring. First thing to check on the next GPU-VM session (log `.shape` once).
        """
        cfg = self.cfg
        raw = self.depth_camera.data.output["distance_to_image_plane"]  # (N, H, W[, 1])
        if raw.dim() == 4:
            raw = raw.squeeze(-1)  # (N, H, W)

        # depth_clipping_behavior="max" should already clamp out-of-range rays, but defensively
        # replace any NaN/inf before cropping/blurring, so a single bad ray can't propagate into
        # the whole blurred neighborhood.
        raw = torch.nan_to_num(raw, nan=cfg.DEPTH_RANGE_M[1], posinf=cfg.DEPTH_RANGE_M[1], neginf=cfg.DEPTH_RANGE_M[0])

        crop_x, crop_y, crop_w, crop_h = cfg.DEPTH_CROP_REGION
        cropped = raw[:, crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]  # (N, 16, 16)

        # Gaussian blur via a fixed, precomputed small kernel (no torchvision dependency) --
        # separable would be cheaper, but at 16x16 a single small conv2d is already trivial.
        blurred = F.conv2d(
            cropped.unsqueeze(1),  # (N, 1, 16, 16)
            self._depth_blur_kernel,
            padding=cfg.DEPTH_BLUR_KERNEL_SIZE // 2,
        ).squeeze(1)  # (N, 16, 16)

        clamped = torch.clamp(blurred, cfg.DEPTH_RANGE_M[0], cfg.DEPTH_RANGE_M[1])
        normalized = (clamped - cfg.DEPTH_RANGE_M[0]) / (cfg.DEPTH_RANGE_M[1] - cfg.DEPTH_RANGE_M[0])
        return normalized

    def _get_rewards(self) -> torch.Tensor:
        # self.commands must reflect THIS step's post-physics robot pose before either the task
        # reward below or _get_observations() (which runs after this and reads self.commands
        # too) consume it -- _get_rewards() is the first consumer each step (DirectRLEnv.step()
        # order: physics -> _get_dones() -> _get_rewards() -> resets -> _get_observations()).
        self._update_position_based_commands()

        # pre-slice per-joint-group tensors here (plain python/torch indexing) rather than inside
        # the jit-scripted compute_rewards, so that function only ever does simple reductions and
        # never needs to know about index lists (TorchScript's typing for that is a headache).
        joint_pos_29 = self.robot.data.joint_pos[:, self.action_dof_indexes]
        default_joint_pos_29 = self.robot.data.default_joint_pos[:, self.action_dof_indexes]
        soft_limits_29 = self.robot.data.soft_joint_pos_limits[:, self.action_dof_indexes]
        joint_acc_29 = self.robot.data.joint_acc[:, self.action_dof_indexes]
        applied_torque_29 = self.robot.data.applied_torque[:, self.action_dof_indexes]

        total_reward, reward_log = compute_rewards(
            self.cfg.rew_lin_vel_xy,
            self.cfg.rew_ang_vel_z,
            self.cfg.rew_track_sigma,
            self.cfg.rew_termination,
            self.cfg.rew_action_rate_l2,
            self.cfg.rew_joint_pos_limits,
            self.cfg.rew_joint_acc_l2,
            self.cfg.rew_dof_torques_l2,
            self.cfg.rew_flat_orientation_l2,
            self.cfg.rew_lin_vel_z_l2,
            self.cfg.rew_ang_vel_xy_l2,
            self.cfg.rew_joint_deviation_hip,
            self.cfg.rew_joint_deviation_arms,
            self.cfg.rew_joint_deviation_torso,
            self.commands,
            self.robot.data.root_lin_vel_b,
            self.robot.data.root_ang_vel_b,
            self.reset_terminated,
            self.actions,
            self.prev_actions,
            self.robot.data.projected_gravity_b,
            joint_acc_29[:, self.hip_knee_dof_indexes],
            applied_torque_29[:, self.hip_knee_dof_indexes],
            joint_pos_29[:, self.ankle_dof_indexes],
            soft_limits_29[:, self.ankle_dof_indexes],
            joint_pos_29[:, self.hip_deviation_dof_indexes] - default_joint_pos_29[:, self.hip_deviation_dof_indexes],
            joint_pos_29[:, self.arm_deviation_dof_indexes] - default_joint_pos_29[:, self.arm_deviation_dof_indexes],
            joint_pos_29[:, self.torso_deviation_dof_indexes] - default_joint_pos_29[:, self.torso_deviation_dof_indexes],
        )

        # Terrain-aware reward terms (added 2026-08-12, see g1_stairs_env_cfg.py's comment on
        # rew_feet_height_error / rew_edge_penetration for what each is adapted from). Computed
        # here in plain Python/torch, not inside the jit-scripted compute_rewards above, to avoid
        # growing that function's already-long signature and to keep this new, untested-on-GPU
        # code isolated and easy to find/remove if it turns out wrong.
        feet_pos_w = self.robot.data.body_pos_w[:, self.feet_body_indexes]  # (N, 2, 3)
        feet_vel_w = self.robot.data.body_lin_vel_w[:, self.feet_body_indexes]  # (N, 2, 3)
        feet_contact_force = self.contact_sensor.data.net_forces_w[:, self.feet_contact_body_indexes]  # (N, 2, 3)
        feet_in_contact = torch.norm(feet_contact_force, dim=-1) > 1.0  # (N, 2), 1N threshold

        terrain_h = self._terrain_height_at(feet_pos_w)  # (N, 2)
        edge_dist = self._edge_distance_at(feet_pos_w)  # (N, 2)

        height_err = torch.clamp(
            feet_pos_w[..., 2] - terrain_h - self.cfg.feet_height_margin_m,
            min=0.0, max=self.cfg.feet_height_clip_max_m,
        )
        rew_feet_height_error = self.cfg.rew_feet_height_error * torch.sum(
            height_err * feet_in_contact.float(), dim=1
        )

        edge_violation = torch.clamp(self.cfg.edge_safety_margin_m - edge_dist, min=0.0)  # (N, 2)
        feet_speed = torch.norm(feet_vel_w, dim=-1)  # (N, 2)
        rew_edge_penetration = self.cfg.rew_edge_penetration * torch.sum(
            edge_violation * (feet_speed + 1.0e-3) * feet_in_contact.float(), dim=1
        )

        # Anti-dawdling terms (added 2026-08-13, ported verbatim from InstinctLab's real reward
        # functions -- see g1_stairs_env_cfg.py's comment on rew_dont_wait/rew_stand_still for
        # exact source citations -- to counter the "camps at the base to avoid rew_termination"
        # failure mode found via video review. Computed here in plain Python/torch, same reason
        # as the terrain-aware terms above: untested on GPU yet, kept easy to isolate/remove.
        lin_vel_cmd_x = self.commands[:, 0]
        lin_vel_x_b = self.robot.data.root_lin_vel_b[:, 0]
        dont_wait_count = (
            (lin_vel_x_b < 0.15).float() + (lin_vel_x_b < 0.0).float() + (lin_vel_x_b < -0.15).float()
        )
        rew_dont_wait = self.cfg.rew_dont_wait * (
            lin_vel_cmd_x > self.cfg.dont_wait_cmd_threshold_mps
        ).float() * dont_wait_count

        dof_error = torch.sum(torch.abs(joint_pos_29 - default_joint_pos_29), dim=1)
        standing_mask = (
            (torch.norm(self.commands[:, :2], dim=1) < self.cfg.stand_still_cmd_threshold).float()
            * (torch.abs(self.commands[:, 2]) < self.cfg.stand_still_cmd_threshold).float()
        )
        rew_stand_still = self.cfg.rew_stand_still * (dof_error - self.cfg.stand_still_offset) * standing_mask

        # Height-progress diagnostic (added 2026-08-14): root height relative to the goal/center
        # platform (0 at goal, negative below it, same terrain-relative convention as
        # _terrain_height_at) -- a direct, cheap proxy for "how far up the stairs has the robot
        # actually gotten," independent of any reward-shaping question. cfg.rew_max_height
        # defaults to 0.0 (no effect on training) specifically so this can be watched in
        # TensorBoard without changing behavior; raise it later if height-shaping turns out to
        # help. Logged as the RAW (unweighted) mean height, not the weighted contribution --
        # at weight 0.0 the weighted contribution is always exactly 0 and wouldn't show anything
        # useful, defeating the point of watching it.
        height_rel_to_goal = self.robot.data.root_pos_w[:, 2] - self.terrain.env_origins[:, 2]
        rew_max_height = self.cfg.rew_max_height * height_rel_to_goal

        total_reward = (
            total_reward + rew_feet_height_error + rew_edge_penetration + rew_dont_wait + rew_stand_still
            + rew_max_height
        )
        reward_log["rew_feet_height_error"] = rew_feet_height_error.mean()
        reward_log["rew_edge_penetration"] = rew_edge_penetration.mean()
        reward_log["rew_dont_wait"] = rew_dont_wait.mean()
        reward_log["rew_stand_still"] = rew_stand_still.mean()
        reward_log["rew_max_height"] = rew_max_height.mean()
        reward_log["diag_mean_height_rel_to_goal"] = height_rel_to_goal.mean()

        self.extras["log"] = reward_log
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.early_termination:
            died = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height
        else:
            died = torch.zeros_like(time_out)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0

        if self.cfg.reset_strategy == "default":
            root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        elif self.cfg.reset_strategy.startswith("random"):
            start = "start" in self.cfg.reset_strategy
            root_state, joint_pos, joint_vel = self._reset_strategy_random(env_ids, start)
        else:
            raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._resample_commands(env_ids)
        # _resample_commands only updates goal_pos_w now (see its docstring) -- unlike the
        # uniform-velocity version this env forked from, self.commands is pose-DEPENDENT, so it
        # must be recomputed here too. Without this, freshly-reset envs would carry a stale
        # command (from before reset) into the very next _get_observations() call, since
        # DirectRLEnv.step() computes obs right after _reset_idx with no other refresh point.
        self._update_position_based_commands()

    # reset strategies

    def _sample_border_spawn_positions(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Random flat spawn point on the stairs tile's border ring (bottom of the stairs), one
        of the pyramid's 4 faces chosen uniformly per env, as an offset from
        self.terrain.env_origins (that tile's own Isaac Lab `origin`, which is the CENTER
        PLATFORM -- our command-generation goal, not a spawn point -- see
        g1_stairs_env_cfg.py's STAIRS_TERRAIN_CFG comment).

        Analytic, not flat-patch-sampled: computed directly from the known stairs dimensions
        (STAIRS_SPAWN_RADIUS_M / _LATERAL_MARGIN_M / TOTAL_RISE_M in g1_stairs_env_cfg.py)
        rather than reading self.terrain.flat_patches, since this project fully controls that
        geometry already and the exact flat_patches tensor shape/frame isn't verified against a
        real Isaac Lab install (see module docstring). Returns world-space (x, y, z) per env_id.
        """
        n = len(env_ids)
        face = torch.randint(0, 4, (n,), device=self.device)  # 0:+X 1:-X 2:+Y 3:-Y
        lateral = sample_uniform(-STAIRS_SPAWN_LATERAL_MARGIN_M, STAIRS_SPAWN_LATERAL_MARGIN_M, (n,), self.device)
        offset = torch.zeros(n, 3, device=self.device)
        offset[face == 0, 0] = STAIRS_SPAWN_RADIUS_M
        offset[face == 1, 0] = -STAIRS_SPAWN_RADIUS_M
        offset[face == 2, 1] = STAIRS_SPAWN_RADIUS_M
        offset[face == 3, 1] = -STAIRS_SPAWN_RADIUS_M
        cross_axis = torch.where(face < 2, torch.ones(n, device=self.device), torch.zeros(n, device=self.device))
        offset[:, 1] += lateral * cross_axis  # faces 0/1: lateral varies along Y
        offset[:, 0] += lateral * (1.0 - cross_axis)  # faces 2/3: lateral varies along X
        offset[:, 2] = -STAIRS_TOTAL_RISE_M  # env_origins' Z is the platform height; bring back
        # down to the border ring's flat ground level
        return self.terrain.env_origins[env_ids] + offset

    def _reset_strategy_default(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self._sample_border_spawn_positions(env_ids)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        return root_state, joint_pos, joint_vel

    def _reset_strategy_random(
        self, env_ids: torch.Tensor, start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_samples = env_ids.shape[0]
        times = np.zeros(num_samples) if start else self._motion_loader.sample_times(num_samples)
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)

        motion_torso_index = self._motion_loader.get_body_index(["pelvis"])[0]
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = body_positions[:, motion_torso_index] + self._sample_border_spawn_positions(env_ids)
        # small clearance so the stance foot doesn't spawn interpenetrating the ground. The
        # motion data itself is ground-aligned by the converter (lowest sole at z=0) -- an
        # earlier version of the npz floated ~0.08-0.09 m AND this offset was 0.05, so every
        # reset started with a ~0.13-0.17 m free-fall drop onto the ground.
        root_state[:, 2] += 0.02

        # Heading alignment (added 2026-08-13, fixing a real bug found via video review of a full
        # training run -- see project memory for the full writeup). convert_gmr_to_npz.py's
        # align_x step hard-aligns the reference clip's own root heading + world-frame velocities
        # to world +X. _sample_border_spawn_positions puts each env on one of 4 faces around the
        # stairs, so copying the clip's fixed +X heading straight into root_state only pointed at
        # the goal for spawns on face 0 (+X); face 1 (-X) started 180 degrees off, faces 2/3 (+-Y)
        # started 90 degrees off. Rotate the clip's sampled root orientation AND its linear/angular
        # velocity by the same per-env yaw delta so the clip's forward direction (and its recorded
        # motion) points from this env's spawn position toward its goal (the tile center platform,
        # self.terrain.env_origins), instead of always world +X. This only corrects the state
        # written at reset -- it does not touch the AMP style features computed every step
        # (compute_obs, world-frame body_lin_vel_w/body_ang_vel_w), which is the separate,
        # deeper issue (discriminator reference data is always +X-relative regardless of which
        # way the robot is actually walking) still open for a structural fix.
        clip_root_rot = body_rotations[:, motion_torso_index]
        clip_lin_vel = body_linear_velocities[:, motion_torso_index]
        clip_ang_vel = body_angular_velocities[:, motion_torso_index]

        goal_xy = self.terrain.env_origins[env_ids, :2]
        to_goal_xy = goal_xy - root_state[:, 0:2]
        target_yaw = torch.atan2(to_goal_xy[:, 1], to_goal_xy[:, 0])
        z_axis = torch.zeros(num_samples, 3, device=self.device)
        z_axis[:, 2] = 1.0
        target_yaw_quat = quat_from_angle_axis(target_yaw, z_axis)
        delta_yaw_quat = quat_mul(target_yaw_quat, quat_conjugate(yaw_quat(clip_root_rot)))

        root_state[:, 3:7] = quat_mul(delta_yaw_quat, clip_root_rot)
        root_state[:, 7:10] = quat_apply(delta_yaw_quat, clip_lin_vel)
        root_state[:, 10:13] = quat_apply(delta_yaw_quat, clip_ang_vel)
        # fingers (not in action_dof_indexes) reset to the robot's own default pose; only
        # the 29 motion-covered joints get values from the sampled reference motion
        dof_pos = self.robot.data.default_joint_pos[env_ids].clone()
        dof_vel = self.robot.data.default_joint_vel[env_ids].clone()
        dof_pos[:, self.action_dof_indexes] = dof_positions[:, self.motion_dof_indexes]
        dof_vel[:, self.action_dof_indexes] = dof_velocities[:, self.motion_dof_indexes]

        amp_observations = self.collect_reference_motions(num_samples, times)
        self.amp_observation_buffer[env_ids] = amp_observations.view(num_samples, self.cfg.num_amp_observations, -1)

        return root_state, dof_pos, dof_vel

    # env methods

    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None) -> torch.Tensor:
        if current_times is None:
            current_times = self._motion_loader.sample_times(num_samples)
        times = (
            np.expand_dims(current_times, axis=-1)
            - self._motion_loader.dt * np.arange(0, self.cfg.num_amp_observations)
        ).flatten()
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)
        amp_observation = compute_obs(
            dof_positions[:, self.motion_dof_indexes],
            dof_velocities[:, self.motion_dof_indexes],
            body_positions[:, self.motion_ref_body_index],
            body_rotations[:, self.motion_ref_body_index],
            body_linear_velocities[:, self.motion_ref_body_index],
            body_angular_velocities[:, self.motion_ref_body_index],
            body_positions[:, self.motion_key_body_indexes],
        )
        return amp_observation.view(-1, self.amp_observation_size)


@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_apply(q, ref_tangent)
    normal = quat_apply(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)


@torch.jit.script
def compute_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    key_body_positions: torch.Tensor,
) -> torch.Tensor:
    # Heading-relativize everything below that would otherwise leak *world-frame* heading into the
    # AMP style features (added 2026-08-13, fixing "Finding B" -- see g1_stairs_env_cfg.py's module
    # docstring and instinctlab_comparison.md Section 4.2 for the full writeup: our AMP discriminator
    # was judging "authenticity" against a world-+X-aligned reference clip, fighting genuine
    # goal-directed motion on 3 of 4 stairs approach faces for the whole episode). Mirrors
    # InstinctLab's own confirmed pattern (reference/InstinctLab/.../reference_as_state.py:64-91,
    # quat_apply_inverse(base_quat_w, vel_w)) -- linear/angular velocity go to FULL base frame
    # (matching that pattern and our own task reward's root_lin_vel_b/root_ang_vel_b convention),
    # while orientation and key-body offsets are relativized by YAW ONLY (full-quaternion removal
    # would make quaternion_to_tangent_and_normal degenerate to a constant and destroy the
    # roll/pitch gait-tilt information the discriminator actually needs -- InstinctLab sidesteps
    # this by using projected gravity, which is physics-yaw-invariant by construction, instead of
    # tangent/normal; yaw-only removal is the direct analogue for the tangent/normal
    # representation this project already uses). This function is shared by both the rollout side
    # (_get_observations) and the reference side (collect_reference_motions), so fixing it here
    # applies the identical transform to both automatically -- no per-call-site changes needed.
    heading_rot = yaw_quat(root_rotations)
    root_rotations_rel = quat_mul(quat_conjugate(heading_rot), root_rotations)
    root_linear_velocities_b = quat_apply_inverse(root_rotations, root_linear_velocities)
    root_angular_velocities_b = quat_apply_inverse(root_rotations, root_angular_velocities)

    key_body_offsets_w = key_body_positions - root_positions.unsqueeze(-2)
    heading_rot_expanded = heading_rot.unsqueeze(-2).expand(
        key_body_offsets_w.shape[0], key_body_offsets_w.shape[1], 4
    )
    key_body_offsets_b = quat_apply_inverse(heading_rot_expanded, key_body_offsets_w)

    obs = torch.cat(
        (
            dof_positions,
            dof_velocities,
            root_positions[:, 2:3],  # root body height
            quaternion_to_tangent_and_normal(root_rotations_rel),
            root_linear_velocities_b,
            root_angular_velocities_b,
            key_body_offsets_b.reshape(key_body_positions.shape[0], -1),
        ),
        dim=-1,
    )
    return obs


@torch.jit.script
def compute_rewards(
    rew_lin_vel_xy: float,
    rew_ang_vel_z: float,
    rew_track_sigma: float,
    rew_scale_termination: float,
    rew_scale_action_rate_l2: float,
    rew_scale_joint_pos_limits: float,
    rew_scale_joint_acc_l2: float,
    rew_scale_dof_torques_l2: float,
    rew_scale_flat_orientation_l2: float,
    rew_scale_lin_vel_z_l2: float,
    rew_scale_ang_vel_xy_l2: float,
    rew_scale_joint_deviation_hip: float,
    rew_scale_joint_deviation_arms: float,
    rew_scale_joint_deviation_torso: float,
    commands: torch.Tensor,
    root_lin_vel_b: torch.Tensor,
    root_ang_vel_b: torch.Tensor,
    reset_terminated: torch.Tensor,
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    projected_gravity_b: torch.Tensor,
    hip_knee_joint_acc: torch.Tensor,
    hip_knee_applied_torque: torch.Tensor,
    ankle_joint_pos: torch.Tensor,
    ankle_soft_joint_pos_limits: torch.Tensor,
    hip_deviation: torch.Tensor,
    arm_deviation: torch.Tensor,
    torso_deviation: torch.Tensor,
):
    # task reward: velocity tracking, base frame — mirrors Project 1's
    # track_lin_vel_xy_exp / track_ang_vel_z_exp manager-based reward terms
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - root_lin_vel_b[:, :2]), dim=1)
    rew_task_lin_vel = rew_lin_vel_xy * torch.exp(-lin_vel_error / rew_track_sigma)

    ang_vel_error = torch.square(commands[:, 2] - root_ang_vel_b[:, 2])
    rew_task_ang_vel = rew_ang_vel_z * torch.exp(-ang_vel_error / rew_track_sigma)

    # regularization — matches Project 1's stock g1_flat run (see g1_stairs_env_cfg.py)
    rew_termination = rew_scale_termination * reset_terminated.float()
    rew_action_rate_l2 = rew_scale_action_rate_l2 * torch.sum(torch.square(actions - prev_actions), dim=1)

    out_of_limits = -(ankle_joint_pos - ankle_soft_joint_pos_limits[:, :, 0]).clip(max=0.0)
    out_of_limits += (ankle_joint_pos - ankle_soft_joint_pos_limits[:, :, 1]).clip(min=0.0)
    rew_joint_pos_limits = rew_scale_joint_pos_limits * torch.sum(out_of_limits, dim=1)

    rew_joint_acc_l2 = rew_scale_joint_acc_l2 * torch.sum(torch.square(hip_knee_joint_acc), dim=1)
    rew_dof_torques_l2 = rew_scale_dof_torques_l2 * torch.sum(torch.square(hip_knee_applied_torque), dim=1)
    rew_flat_orientation_l2 = rew_scale_flat_orientation_l2 * torch.sum(
        torch.square(projected_gravity_b[:, :2]), dim=1
    )
    rew_lin_vel_z_l2 = rew_scale_lin_vel_z_l2 * torch.square(root_lin_vel_b[:, 2])
    rew_ang_vel_xy_l2 = rew_scale_ang_vel_xy_l2 * torch.sum(torch.square(root_ang_vel_b[:, :2]), dim=1)
    rew_joint_deviation_hip = rew_scale_joint_deviation_hip * torch.sum(torch.abs(hip_deviation), dim=1)
    rew_joint_deviation_arms = rew_scale_joint_deviation_arms * torch.sum(torch.abs(arm_deviation), dim=1)
    rew_joint_deviation_torso = rew_scale_joint_deviation_torso * torch.sum(torch.abs(torso_deviation), dim=1)

    total_reward = (
        rew_task_lin_vel + rew_task_ang_vel
        + rew_termination + rew_action_rate_l2 + rew_joint_pos_limits
        + rew_joint_acc_l2 + rew_dof_torques_l2 + rew_flat_orientation_l2
        + rew_lin_vel_z_l2 + rew_ang_vel_xy_l2
        + rew_joint_deviation_hip + rew_joint_deviation_arms + rew_joint_deviation_torso
    )

    log = {
        "rew_task_lin_vel": rew_task_lin_vel.mean(),
        "rew_task_ang_vel": rew_task_ang_vel.mean(),
        "rew_termination": rew_termination.mean(),
        "rew_action_rate_l2": rew_action_rate_l2.mean(),
        "rew_joint_pos_limits": rew_joint_pos_limits.mean(),
        "rew_joint_acc_l2": rew_joint_acc_l2.mean(),
        "rew_dof_torques_l2": rew_dof_torques_l2.mean(),
        "rew_flat_orientation_l2": rew_flat_orientation_l2.mean(),
        "rew_lin_vel_z_l2": rew_lin_vel_z_l2.mean(),
        "rew_ang_vel_xy_l2": rew_ang_vel_xy_l2.mean(),
        "rew_joint_deviation_hip": rew_joint_deviation_hip.mean(),
        "rew_joint_deviation_arms": rew_joint_deviation_arms.mean(),
        "rew_joint_deviation_torso": rew_joint_deviation_torso.mean(),
    }
    return total_reward, log
