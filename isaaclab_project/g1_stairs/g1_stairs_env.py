"""Vision-based AMP stair-climbing environment (Isaac Lab direct workflow).

Each step the policy sees stacked proprioception, a velocity command derived from the goal, a
stack of depth frames, and a small privileged block that only the critic reads. Reward is
velocity tracking plus regularization; an AMP discriminator scores gait style against a climbing
reference clip.

Direct-workflow envs have no manager layer, so the pieces a manager-based env would get for free
are written out here: the terrain curriculum (_update_terrain_curriculum), domain randomization
(_randomize_material_properties and the reset block in _reset_idx), and the position-based
velocity command (_update_position_based_commands).

Adapted from linden713/humanoid_amp (BSD-3-Clause) via rl-wbc-g1-amp; the reward set, event
ranges and sensor geometry follow project-instinct's InstinctLab parkour task. See
THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

import math
import re

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import ContactSensor, RayCaster, RayCasterCamera
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
    STAIRS_NUM_LEVELS,
    STAIRS_SPAWN_LATERAL_MARGIN_M,
    STAIRS_SPAWN_RADIUS_M,
    STAIRS_STAIR_REGION_HALF_EXTENT_M,
    STAIRS_TREAD_DEPTH_ACTUAL_M,
    G1StairsEnvCfg,
)
from .motions import MotionLoader

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

        k, sigma = self.cfg.DEPTH_BLUR_KERNEL_SIZE, self.cfg.DEPTH_BLUR_SIGMA
        coords = torch.arange(k, dtype=torch.float32, device=self.device) - (k - 1) / 2.0
        gauss_1d = torch.exp(-(coords**2) / (2 * sigma**2))
        gauss_2d = torch.outer(gauss_1d, gauss_1d)
        gauss_2d = gauss_2d / gauss_2d.sum()
        self._depth_blur_kernel = gauss_2d.view(1, 1, k, k)

        self._motion_loader = MotionLoader(motion_file=self.cfg.motion_file, device=self.device)

        motion_joint_names = [n for n in self.robot.data.joint_names if n in self._motion_loader.dof_names]
        assert len(motion_joint_names) == self.cfg.action_space, (
            f"Expected {self.cfg.action_space} robot joints covered by the motion file, "
            f"found {len(motion_joint_names)}: {motion_joint_names}"
        )
        self.action_dof_indexes = [self.robot.data.joint_names.index(n) for n in motion_joint_names]
        self.motion_dof_indexes = self._motion_loader.get_dof_index(motion_joint_names)

        def _match(patterns: list[str]) -> list[int]:
            return [i for i, n in enumerate(motion_joint_names) if any(re.fullmatch(p, n) for p in patterns)]

        self.hip_knee_dof_indexes = _match([r".*_hip_.*", r".*_knee_joint"])
        self.ankle_dof_indexes = _match([r".*_ankle_pitch_joint", r".*_ankle_roll_joint"])
        self.hip_knee_ankle_dof_indexes = self.hip_knee_dof_indexes + self.ankle_dof_indexes
        self.hip_knee_ankle_full_dof_indexes = [self.action_dof_indexes[i] for i in self.hip_knee_ankle_dof_indexes]
        self.hip_deviation_dof_indexes = _match([r".*_hip_yaw_joint", r".*_hip_roll_joint"])
        self.arm_deviation_dof_indexes = _match([
            r".*_shoulder_pitch_joint", r".*_shoulder_roll_joint", r".*_shoulder_yaw_joint", r".*_elbow_joint",
            r".*_wrist_roll_joint", r".*_wrist_pitch_joint", r".*_wrist_yaw_joint",
        ])
        self.torso_deviation_dof_indexes = _match([r"waist_yaw_joint", r"waist_roll_joint", r"waist_pitch_joint"])

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

        feet_body_names = ["right_ankle_roll_link", "left_ankle_roll_link"]
        self.feet_body_indexes = [self.robot.data.body_names.index(name) for name in feet_body_names]
        self.feet_contact_body_indexes = [self.contact_sensor.body_names.index(name) for name in feet_body_names]

        self.undesired_contact_body_indexes = [
            i for i, name in enumerate(self.body_contact_sensor.body_names) if name not in feet_body_names
        ]

        self.termination_contact_body_indexes = []
        for name in self.cfg.termination_contact_body_names:
            if name not in self.body_contact_sensor.body_names:
                raise ValueError(
                    f"termination_contact_body_names entry {name!r} not found in the contact "
                    f"sensor's discovered bodies: {self.body_contact_sensor.body_names}"
                )
            self.termination_contact_body_indexes.append(self.body_contact_sensor.body_names.index(name))

        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        )

        self.policy_proprio_buffer = torch.zeros(
            (self.num_envs, self.cfg.policy_proprio_history_len, self.cfg.amp_observation_space),
            device=self.device,
        )

        self.commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_marker = VisualizationMarkers(GOAL_MARKER_CFG) if self.cfg.debug_vis_goal else None
        self._resample_commands(torch.arange(self.num_envs, device=self.device))
        self._update_position_based_commands()

        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)

        self._depth_pixels = self.cfg.DEPTH_FINAL_SIZE**2
        self.depth_history_buffer = torch.zeros(
            (self.num_envs, self.cfg.depth_history_buffer_len, self._depth_pixels), device=self.device
        )
        self._depth_ptr = 0
        self._depth_history_offsets = torch.arange(
            0, self.cfg.depth_history_len * self.cfg.depth_history_skip, self.cfg.depth_history_skip,
            device=self.device,
        )
        assert int(self._depth_history_offsets.max()) < self.cfg.depth_history_buffer_len, (
            f"depth history needs {int(self._depth_history_offsets.max()) + 1} buffered frames "
            f"but depth_history_buffer_len is {self.cfg.depth_history_buffer_len}"
        )

        self._track_score_sum = torch.zeros(self.num_envs, device=self.device)

        # Per-cause termination diagnostics, populated by _get_dones and logged by _get_rewards.
        # Initialized here because _get_rewards can run before _get_dones has ever filled it.
        self._term_cause: dict[str, torch.Tensor] = {}

        self._randomize_material_properties()
        self._log_terrain_geometry()

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = TerrainImporter(self.cfg.terrain)

        self.contact_sensor = ContactSensor(self.cfg.feet_contact_sensor)
        self.scene.sensors["feet_contact_sensor"] = self.contact_sensor

        self.body_contact_sensor = ContactSensor(self.cfg.body_contact_sensor)
        self.scene.sensors["body_contact_sensor"] = self.body_contact_sensor

        self.depth_camera = RayCasterCamera(self.cfg.depth_camera)
        self.scene.sensors["depth_camera"] = self.depth_camera

        self.left_height_scanner = RayCaster(self.cfg.left_height_scanner)
        self.right_height_scanner = RayCaster(self.cfg.right_height_scanner)
        self.base_height_scanner = RayCaster(self.cfg.base_height_scanner)
        self.scene.sensors["left_height_scanner"] = self.left_height_scanner
        self.scene.sensors["right_height_scanner"] = self.right_height_scanner
        self.scene.sensors["base_height_scanner"] = self.base_height_scanner

        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone()

        resample_steps = int(self.cfg.command_resample_time_s / (self.cfg.sim.dt * self.cfg.decimation))
        due = (self.episode_length_buf % max(resample_steps, 1) == 0) & (self.episode_length_buf > 0)
        env_ids = due.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self._resample_commands(env_ids)

    def _resample_commands(self, env_ids: torch.Tensor):
        """Resample the per-env goal. The goal is always that tile's centre platform, which
                `terrain.env_origins` already resolves to, so there is no selection to make -- the
                difficult direction is always through the stairs. The velocity command itself is derived
                from this goal every step in _update_position_based_commands.
                """
        if len(env_ids) == 0:
            return
        self.goal_pos_w[env_ids] = self.terrain.env_origins[env_ids]

    def _update_position_based_commands(self):
        """v_x = clip(k_v * x_g, 0, v_max), w_z = clip(k_w * atan2(y_g, x_g), +-w_max), where
                (x_g, y_g) is the goal in the robot's yaw frame. Recomputed every step so the command
                tracks the live goal-relative pose. v_y is fixed at 0 (forward-facing camera only).
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

        arrived = torch.norm(goal_vec_w[:, :2], dim=1) <= r.command_target_dist_threshold_m
        self.commands[arrived] = 0.0

        if self.goal_marker is not None:
            self.goal_marker.visualize(translations=self.goal_pos_w)

    @staticmethod
    def _scanner_hit_heights(sensor: RayCaster) -> torch.Tensor:
        """World-Z of a height scanner's ray hits, misses zeroed. (num_envs, num_rays).

                Reading the real mesh keeps terrain height correct at every curriculum level, where a
                closed-form formula would need the per-level riser height and step count.
                """
        z = sensor.data.ray_hits_w[..., 2]
        return torch.where(torch.isfinite(z), z, torch.zeros_like(z))

    def _base_terrain_height(self) -> torch.Tensor:
        """Terrain height under the pelvis, (num_envs,).

                Reduced with max rather than squeeze so a wider ray pattern would still return the right
                shape. Max is also the conservative choice for the fall check: straddling a step edge is
                measured against the upper tread.
                """
        return self._scanner_hit_heights(self.base_height_scanner).max(dim=-1).values

    def _edge_distance_at(self, pos_w: torch.Tensor) -> torch.Tensor:
        """Planar distance from a world-space position to the nearest stair riser edge.

                pos_w is (num_envs, ..., 3); only XY is used. Riser lines sit at exact multiples of the
                tread depth out from the stair-region boundary, in the same Chebyshev metric the generator
                uses. Stays valid across curriculum levels because only riser HEIGHT varies with
                difficulty -- tread depth, platform width and tile size are fixed.
                """
        origin = self.terrain.env_origins
        extra_dims = pos_w.dim() - 2
        view_shape = (origin.shape[0],) + (1,) * extra_dims
        origin_x = origin[:, 0].view(view_shape)
        origin_y = origin[:, 1].view(view_shape)

        d = torch.maximum(torch.abs(pos_w[..., 0] - origin_x), torch.abs(pos_w[..., 1] - origin_y))
        d_prime = STAIRS_STAIR_REGION_HALF_EXTENT_M - d

        num_edge_lines = int(math.ceil(STAIRS_STAIR_REGION_HALF_EXTENT_M / STAIRS_TREAD_DEPTH_ACTUAL_M)) + 1
        k = torch.arange(num_edge_lines, device=pos_w.device, dtype=pos_w.dtype)
        edge_positions = k * STAIRS_TREAD_DEPTH_ACTUAL_M
        dist_to_each_edge = torch.abs(d_prime.unsqueeze(-1) - edge_positions)
        return dist_to_each_edge.min(dim=-1).values

    def _randomize_material_properties(self):
        """Randomize friction and restitution once at startup.

                Samples `event_material_num_buckets` distinct materials and assigns each body one, the
                same bucketing Isaac Lab's own event term uses to keep the PhysX material count bounded.
                The buffer shape is asserted rather than assumed so an API mismatch fails loudly here.
                """
        cfg = self.cfg
        view = self.robot.root_physx_view
        materials = view.get_material_properties().to(self.device)
        assert materials.dim() == 3 and materials.shape[-1] == 3, (
            f"unexpected material buffer shape {tuple(materials.shape)}; expected "
            "(num_envs, num_shapes, 3) of (static friction, dynamic friction, restitution)"
        )
        n_buckets = cfg.event_material_num_buckets
        buckets = torch.empty(n_buckets, 3, device=self.device)
        for col, (lo, hi) in enumerate(
            (cfg.event_static_friction_range, cfg.event_dynamic_friction_range, cfg.event_restitution_range)
        ):
            buckets[:, col] = sample_uniform(lo, hi, (n_buckets,), self.device)
        buckets[:, 1] = torch.minimum(buckets[:, 1], buckets[:, 0])

        choice = torch.randint(0, n_buckets, (materials.shape[0], materials.shape[1]), device=self.device)
        assigned = buckets[choice]
        view.set_material_properties(assigned.cpu(), torch.arange(materials.shape[0]))
        self._env_static_friction = assigned[:, :, 0].mean(dim=1)
        print(
            f"[g1_stairs] randomized materials over {n_buckets} buckets: "
            f"static friction {cfg.event_static_friction_range}, "
            f"dynamic {cfg.event_dynamic_friction_range}, restitution {cfg.event_restitution_range}"
        )

    def _log_terrain_geometry(self):
        """Print measured platform height per curriculum level, once, at startup.

                Cheap check that the generated stair geometry matches what the config intends: heights
                should rise monotonically across levels, and dividing by the level's step height gives the
                real step count.
                """
        origins_z = self.terrain.env_origins[:, 2]
        levels = self.terrain.terrain_levels
        print("[g1_stairs] measured terrain geometry (platform height above tile ground, per level):")
        for lvl in range(int(levels.max().item()) + 1):
            mask = levels == lvl
            if not bool(mask.any()):
                continue
            print(f"    level {lvl:>2}: platform z = {origins_z[mask].mean().item():.4f} m  "
                  f"({int(mask.sum().item())} envs)")

    def _update_terrain_curriculum(self, env_ids: torch.Tensor):
        """Promote or demote envs a terrain level on their episode-mean tracking score.

                Must run before the new spawn pose is computed: it rewrites `terrain.env_origins`, which
                both the spawn placement and the goal read from. Score is normalized by max_episode_length
                rather than actual length, so an episode cut short by a fall scores low and demotes.
                """
        # The trainer's first reset has no episode behind it; without this guard its all-zero
        # score would demote every env off its starting level before training begins.
        ran = self.episode_length_buf[env_ids] > 0
        if not bool(ran.any()):
            self._track_score_sum[env_ids] = 0.0
            return

        low, high = self.cfg.terrain_curriculum_lin_vel_threshold
        score = self._track_score_sum[env_ids] / self.max_episode_length
        move_up = (score > high) & ran
        move_down = (score < low) & ran & ~move_up
        self.terrain.update_env_origins(env_ids, move_up, move_down)
        self._track_score_sum[env_ids] = 0.0

    def _apply_action(self):
        target = self.robot.data.default_joint_pos.clone()
        target[:, self.action_dof_indexes] += self.cfg.action_scale * self.actions
        self.robot.set_joint_position_target(target)

    def _get_observations(self) -> dict:
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
        self.extras["amp_obs"] = self.amp_observation_buffer.view(-1, self.amp_observation_size)

        for i in reversed(range(self.cfg.policy_proprio_history_len - 1)):
            self.policy_proprio_buffer[:, i + 1] = self.policy_proprio_buffer[:, i]
        self.policy_proprio_buffer[:, 0] = amp_obs.clone()
        proprio_history_flat = self.policy_proprio_buffer.view(self.num_envs, -1)

        depth_now = self._process_depth_image().view(self.num_envs, -1)
        buf_len = self.cfg.depth_history_buffer_len
        # True circular buffer: the pointer walks backward and always marks the newest frame, so
        # the frame k steps ago is at (ptr + k) % len. Rolling this 155 MB buffer instead would
        # cost a full copy every control step.
        self._depth_ptr = (self._depth_ptr - 1) % buf_len
        self.depth_history_buffer[:, self._depth_ptr] = depth_now
        gather_idx = (self._depth_ptr + self._depth_history_offsets) % buf_len
        depth_obs = self.depth_history_buffer[:, gather_idx].reshape(self.num_envs, -1)

        policy_obs = torch.cat(
            (proprio_history_flat, self.commands, depth_obs, self._privileged_obs()), dim=-1
        )
        return {"policy": policy_obs}

    def _privileged_obs(self) -> torch.Tensor:
        """Critic-only features, appended to the shared observation.

                skrl hands one `states` tensor to both models, so these ride at the tail and the policy
                slices them off (see models.py). All of it is unmeasurable on a real robot: base height
                above terrain, each foot's height above terrain, sampled friction, terrain level.
                """
        root_pos = self.robot.data.root_pos_w
        base_clearance = (root_pos[:, 2] - self._base_terrain_height()).unsqueeze(-1)

        feet_pos_w = self.robot.data.body_pos_w[:, self.feet_body_indexes]
        right_clearance = feet_pos_w[:, 0, 2] - self._scanner_hit_heights(self.right_height_scanner).max(dim=-1).values
        left_clearance = feet_pos_w[:, 1, 2] - self._scanner_hit_heights(self.left_height_scanner).max(dim=-1).values

        level = self.terrain.terrain_levels.float().unsqueeze(-1) / max(float(STAIRS_NUM_LEVELS - 1), 1.0)
        return torch.cat(
            (
                base_clearance,
                right_clearance.unsqueeze(-1),
                left_clearance.unsqueeze(-1),
                self._env_static_friction.unsqueeze(-1),
                level,
            ),
            dim=-1,
        )

    def _process_depth_image(self) -> torch.Tensor:
        """Crop, blur and normalize the raw depth image to (N, 16, 16) in [0, 1]."""
        cfg = self.cfg
        raw = self.depth_camera.data.output["distance_to_image_plane"]
        if raw.dim() == 4:
            raw = raw.squeeze(-1)

        raw = torch.nan_to_num(raw, nan=cfg.DEPTH_RANGE_M[1], posinf=cfg.DEPTH_RANGE_M[1], neginf=cfg.DEPTH_RANGE_M[0])

        crop_x, crop_y, crop_w, crop_h = cfg.DEPTH_CROP_REGION
        cropped = raw[:, crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]

        blurred = F.conv2d(
            cropped.unsqueeze(1),
            self._depth_blur_kernel,
            padding=cfg.DEPTH_BLUR_KERNEL_SIZE // 2,
        ).squeeze(1)

        clamped = torch.clamp(blurred, cfg.DEPTH_RANGE_M[0], cfg.DEPTH_RANGE_M[1])
        normalized = (clamped - cfg.DEPTH_RANGE_M[0]) / (cfg.DEPTH_RANGE_M[1] - cfg.DEPTH_RANGE_M[0])
        return normalized

    def _get_rewards(self) -> torch.Tensor:
        self._update_position_based_commands()

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
            joint_acc_29,
            applied_torque_29[:, self.hip_knee_ankle_dof_indexes],
            joint_pos_29[:, self.ankle_dof_indexes],
            soft_limits_29[:, self.ankle_dof_indexes],
            joint_pos_29[:, self.hip_deviation_dof_indexes] - default_joint_pos_29[:, self.hip_deviation_dof_indexes],
            joint_pos_29[:, self.arm_deviation_dof_indexes] - default_joint_pos_29[:, self.arm_deviation_dof_indexes],
            joint_pos_29[:, self.torso_deviation_dof_indexes] - default_joint_pos_29[:, self.torso_deviation_dof_indexes],
        )

        feet_pos_w = self.robot.data.body_pos_w[:, self.feet_body_indexes]
        feet_vel_w = self.robot.data.body_lin_vel_w[:, self.feet_body_indexes]
        feet_contact_force = self.contact_sensor.data.net_forces_w[:, self.feet_contact_body_indexes]
        feet_in_contact = torch.norm(feet_contact_force, dim=-1) > 1.0

        edge_dist = self._edge_distance_at(feet_pos_w)

        # feet_body_names is [right, left], and feet_pos_w/feet_in_contact follow that order --
        # pairing a foot with the other foot's scanner would score silently wrong.
        right_col, left_col = 0, 1
        per_foot = []
        for col, scanner in ((right_col, self.right_height_scanner), (left_col, self.left_height_scanner)):
            hits = self._scanner_hit_heights(scanner)
            clearance = torch.clamp(
                feet_pos_w[:, col, 2].unsqueeze(-1) - hits - self.cfg.feet_height_margin_m,
                min=0.0, max=self.cfg.feet_height_clip_max_m,
            )
            per_foot.append(torch.sum(clearance * feet_in_contact[:, col].unsqueeze(-1).float(), dim=-1))
        rew_feet_height_error = self.cfg.rew_feet_height_error * (per_foot[0] + per_foot[1])

        edge_violation = torch.clamp(self.cfg.edge_safety_margin_m - edge_dist, min=0.0)
        feet_speed = torch.norm(feet_vel_w, dim=-1)
        rew_edge_penetration = self.cfg.rew_edge_penetration * torch.sum(
            edge_violation * (feet_speed + 1.0e-3) * feet_in_contact.float(), dim=1
        )

        rew_heading_error = self.cfg.rew_heading_error * torch.abs(self.commands[:, 2])

        feet_air_time = self.contact_sensor.data.current_air_time[:, self.feet_contact_body_indexes]
        feet_contact_time = self.contact_sensor.data.current_contact_time[:, self.feet_contact_body_indexes]
        feet_in_contact_now = feet_contact_time > 0.0
        in_mode_time = torch.where(feet_in_contact_now, feet_contact_time, feet_air_time)
        single_stance = torch.sum(feet_in_contact_now.int(), dim=1) == 1
        feet_air_time_value = torch.min(
            torch.where(single_stance.unsqueeze(-1), in_mode_time, torch.zeros_like(in_mode_time)), dim=1
        )[0]
        feet_air_time_cmd_active = torch.logical_or(
            torch.norm(self.commands[:, :2], dim=1) > self.cfg.feet_air_time_vel_threshold_mps,
            torch.abs(self.commands[:, 2]) > self.cfg.feet_air_time_vel_threshold_mps,
        )
        rew_feet_air_time = self.cfg.rew_feet_air_time * feet_air_time_value * feet_air_time_cmd_active.float()

        heading = self.robot.data.heading_w
        cos_h, sin_h = torch.cos(heading), torch.sin(heading)
        feet_lat = -sin_h.unsqueeze(-1) * feet_pos_w[..., 0] + cos_h.unsqueeze(-1) * feet_pos_w[..., 1]
        feet_distance_y = torch.abs(feet_lat[:, 0] - feet_lat[:, 1])
        rew_feet_close_xy = self.cfg.rew_feet_close_xy * (
            torch.exp(
                -torch.clamp(self.cfg.feet_close_xy_threshold_m - feet_distance_y, min=0.0)
                / self.cfg.feet_close_xy_std**2
            )
            - 1.0
        )

        feet_planar_speed = torch.norm(feet_vel_w[..., :2], dim=-1)
        rew_feet_slide = self.cfg.rew_feet_slide * torch.sum(feet_planar_speed * feet_in_contact.float(), dim=1)

        power_full = self.robot.data.applied_torque * self.robot.data.joint_vel
        for actuator in self.robot.actuators.values():
            power_full[:, actuator.joint_indices] = power_full[:, actuator.joint_indices] / actuator.stiffness
        power_hip_knee_ankle = power_full[:, self.hip_knee_ankle_full_dof_indexes]
        rew_energy = self.cfg.rew_energy * torch.sum(torch.square(power_hip_knee_ankle), dim=1)

        joint_effort_limits_29 = self.robot.data.joint_effort_limits[:, self.action_dof_indexes]
        torque_out_of_limits = (
            torch.abs(applied_torque_29) - joint_effort_limits_29 * self.cfg.torque_limit_ratio
        ).clip(min=0.0)
        rew_torque_limits = self.cfg.rew_torque_limits * torch.sum(torch.square(torque_out_of_limits), dim=1)

        undesired_contact_force_hist = self.body_contact_sensor.data.net_forces_w_history[
            :, :, self.undesired_contact_body_indexes
        ]
        undesired_contact = (
            torch.max(torch.norm(undesired_contact_force_hist, dim=-1), dim=1)[0]
            > self.cfg.undesired_contact_threshold_n
        )
        rew_undesired_contacts = self.cfg.rew_undesired_contacts * torch.sum(undesired_contact.float(), dim=1)

        joint_vel_29 = self.robot.data.joint_vel[:, self.action_dof_indexes]
        joint_vel_limits_29 = self.robot.data.joint_vel_limits[:, self.action_dof_indexes]
        vel_out_of_limits = (
            torch.abs(joint_vel_29) - joint_vel_limits_29 * self.cfg.dof_vel_limit_soft_ratio
        ).clip(min=0.0)
        rew_dof_vel_limits = self.cfg.rew_dof_vel_limits * torch.sum(vel_out_of_limits, dim=1)
        rew_dof_vel_l2 = self.cfg.rew_dof_vel_l2 * torch.sum(torch.square(joint_vel_29), dim=1)

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

        height_rel_to_goal = self.robot.data.root_pos_w[:, 2] - self.terrain.env_origins[:, 2]
        rew_max_height = self.cfg.rew_max_height * height_rel_to_goal

        rew_is_alive = self.cfg.rew_is_alive * (~self.reset_terminated).float()

        lin_vel_err_for_curriculum = torch.sum(
            torch.square(self.commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]), dim=1
        )
        self._track_score_sum += torch.exp(-lin_vel_err_for_curriculum / self.cfg.rew_track_sigma)

        total_reward = (
            total_reward + rew_feet_height_error + rew_edge_penetration + rew_dont_wait + rew_stand_still
            + rew_max_height
            + rew_heading_error + rew_feet_air_time + rew_feet_slide + rew_energy + rew_torque_limits
            + rew_undesired_contacts + rew_dof_vel_limits + rew_dof_vel_l2
            + rew_is_alive + rew_feet_close_xy
        )
        reward_log["rew_is_alive"] = rew_is_alive.mean()
        reward_log["rew_feet_close_xy"] = rew_feet_close_xy.mean()
        reward_log["diag_mean_feet_distance_y"] = feet_distance_y.mean()
        reward_log["rew_feet_height_error"] = rew_feet_height_error.mean()
        reward_log["rew_edge_penetration"] = rew_edge_penetration.mean()
        reward_log["rew_dont_wait"] = rew_dont_wait.mean()
        reward_log["rew_stand_still"] = rew_stand_still.mean()
        reward_log["rew_max_height"] = rew_max_height.mean()
        reward_log["rew_heading_error"] = rew_heading_error.mean()
        reward_log["rew_feet_air_time"] = rew_feet_air_time.mean()
        reward_log["rew_feet_slide"] = rew_feet_slide.mean()
        reward_log["rew_energy"] = rew_energy.mean()
        reward_log["rew_torque_limits"] = rew_torque_limits.mean()
        reward_log["rew_undesired_contacts"] = rew_undesired_contacts.mean()
        reward_log["rew_dof_vel_limits"] = rew_dof_vel_limits.mean()
        reward_log["rew_dof_vel_l2"] = rew_dof_vel_l2.mean()
        reward_log["diag_mean_height_rel_to_goal"] = height_rel_to_goal.mean()

        reward_log["diag_mean_raw_root_height"] = self.robot.data.root_pos_w[:, 2].mean()
        base_terrain_h = self._base_terrain_height()
        reward_log["diag_mean_terrain_height_at_root"] = base_terrain_h.mean()
        reward_log["diag_mean_terrain_level"] = self.terrain.terrain_levels.float().mean()
        reward_log["diag_mean_planar_dist_to_goal"] = torch.norm(
            self.goal_pos_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=-1
        ).mean()

        reward_log.update(self._term_cause)
        self.extras["log"] = reward_log
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if not self.cfg.early_termination:
            return torch.zeros_like(time_out), time_out

        base_terrain_h = self._base_terrain_height()
        root_height_above_terrain = (
            self.robot.data.body_pos_w[:, self.ref_body_index, 2] - base_terrain_h
        )
        died_height = root_height_above_terrain < self.cfg.termination_height

        tilt = torch.acos(torch.clamp(-self.robot.data.projected_gravity_b[:, 2], -1.0, 1.0))
        died_tilt = tilt > self.cfg.termination_bad_orientation_rad

        torso_forces = self.body_contact_sensor.data.net_forces_w_history[
            :, :, self.termination_contact_body_indexes
        ]
        died_contact = torch.max(
            torch.norm(torso_forces, dim=-1).flatten(start_dim=1), dim=1
        )[0] > self.cfg.termination_contact_threshold_n

        # Stashed for _get_rewards to log. Without a per-cause breakdown, a run that dies early
        # only tells you THAT it died -- these say which of the three conditions actually fired,
        # plus the raw quantities behind them, so a mis-scaled scanner or a bad sign convention is
        # distinguishable from the robot genuinely falling over.
        self._term_cause = {
            "diag_term_height": died_height.float().mean(),
            "diag_term_tilt": died_tilt.float().mean(),
            "diag_term_contact": died_contact.float().mean(),
            "diag_root_height_above_terrain": root_height_above_terrain.mean(),
            "diag_base_terrain_height": base_terrain_h.mean(),
            "diag_tilt_rad": tilt.mean(),
        }

        return died_height | died_tilt | died_contact, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        # Before super()._reset_idx (which clears episode_length_buf) and before the spawn pose
        # is computed: this rewrites terrain.env_origins, which the spawn and goal both read.
        self._update_terrain_curriculum(env_ids)
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

        n = len(env_ids)
        c = self.cfg
        root_state[:, 0] += sample_uniform(-c.event_reset_pos_range_m, c.event_reset_pos_range_m, (n,), self.device)
        root_state[:, 1] += sample_uniform(-c.event_reset_pos_range_m, c.event_reset_pos_range_m, (n,), self.device)
        yaw_noise = sample_uniform(-c.event_reset_yaw_range_rad, c.event_reset_yaw_range_rad, (n,), self.device)
        z_axis_noise = torch.zeros(n, 3, device=self.device)
        z_axis_noise[:, 2] = 1.0
        root_state[:, 3:7] = quat_mul(quat_from_angle_axis(yaw_noise, z_axis_noise), root_state[:, 3:7])
        root_state[:, 7:10] += sample_uniform(
            -c.event_reset_lin_vel_range, c.event_reset_lin_vel_range, (n, 3), self.device
        )
        root_state[:, 10:13] += sample_uniform(
            -c.event_reset_ang_vel_range, c.event_reset_ang_vel_range, (n, 3), self.device
        )
        joint_pos[:, self.action_dof_indexes] += sample_uniform(
            -c.event_reset_joint_pos_range_rad, c.event_reset_joint_pos_range_rad,
            (n, len(self.action_dof_indexes)), self.device,
        )
        joint_pos = torch.clamp(
            joint_pos,
            self.robot.data.soft_joint_pos_limits[env_ids, :, 0],
            self.robot.data.soft_joint_pos_limits[env_ids, :, 1],
        )

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._resample_commands(env_ids)
        self._update_position_based_commands()

        fresh_proprio = compute_obs(
            self.robot.data.joint_pos[env_ids][:, self.action_dof_indexes],
            self.robot.data.joint_vel[env_ids][:, self.action_dof_indexes],
            self.robot.data.body_pos_w[env_ids, self.ref_body_index],
            self.robot.data.body_quat_w[env_ids, self.ref_body_index],
            self.robot.data.body_lin_vel_w[env_ids, self.ref_body_index],
            self.robot.data.body_ang_vel_w[env_ids, self.ref_body_index],
            self.robot.data.body_pos_w[env_ids][:, self.key_body_indexes],
        )
        self.policy_proprio_buffer[env_ids] = fresh_proprio.unsqueeze(1).expand(
            -1, self.cfg.policy_proprio_history_len, -1
        )

        self.depth_history_buffer[env_ids] = (
            self._process_depth_image().view(self.num_envs, -1)[env_ids].unsqueeze(1)
            .expand(-1, self.cfg.depth_history_buffer_len, -1)
        )

    def _sample_border_spawn_positions(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Random spawn point on the tile's flat border ring, one of the pyramid's 4 faces per env.

                Computed from known tile dimensions rather than terrain.flat_patches. The drop from the
                centre platform to the ring is `env_origins[:, 2]`, which IS the platform height above
                that tile's ground by construction -- correct at every curriculum level without assuming a
                step count or riser height.
                """
        n = len(env_ids)
        face = torch.randint(0, 4, (n,), device=self.device)
        lateral = sample_uniform(-STAIRS_SPAWN_LATERAL_MARGIN_M, STAIRS_SPAWN_LATERAL_MARGIN_M, (n,), self.device)
        offset = torch.zeros(n, 3, device=self.device)
        offset[face == 0, 0] = STAIRS_SPAWN_RADIUS_M
        offset[face == 1, 0] = -STAIRS_SPAWN_RADIUS_M
        offset[face == 2, 1] = STAIRS_SPAWN_RADIUS_M
        offset[face == 3, 1] = -STAIRS_SPAWN_RADIUS_M
        cross_axis = torch.where(face < 2, torch.ones(n, device=self.device), torch.zeros(n, device=self.device))
        offset[:, 1] += lateral * cross_axis
        offset[:, 0] += lateral * (1.0 - cross_axis)
        offset[:, 2] = -self.terrain.env_origins[env_ids, 2]
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
        times = np.zeros(num_samples) if start else self._sample_amp_times(num_samples)
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
        root_state[:, 2] += 0.02

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
        dof_pos = self.robot.data.default_joint_pos[env_ids].clone()
        dof_vel = self.robot.data.default_joint_vel[env_ids].clone()
        dof_pos[:, self.action_dof_indexes] = dof_positions[:, self.motion_dof_indexes]
        dof_vel[:, self.action_dof_indexes] = dof_velocities[:, self.motion_dof_indexes]

        amp_observations = self.collect_reference_motions(num_samples, times)
        self.amp_observation_buffer[env_ids] = amp_observations.view(num_samples, self.cfg.num_amp_observations, -1)

        return root_state, dof_pos, dof_vel

    def _sample_amp_times(self, num_samples: int) -> np.ndarray:
        """Sample reference times that leave room for the AMP history window behind them.

                collect_reference_motions looks back (num_amp_observations - 1) * dt seconds.
                MotionLoader clips the frame index into range but not the interpolation blend, so a
                negative time extrapolates off the front of the clip rather than clamping. Sampling from
                [history_span, duration] keeps the whole window inside the clip.
                """
        history_span = (self.cfg.num_amp_observations - 1) * self._motion_loader.dt
        duration = self._motion_loader.duration
        assert duration > history_span, (
            f"reference motion is {duration:.2f} s but the AMP history window needs "
            f"{history_span:.2f} s; lower num_amp_observations or use a longer clip"
        )
        return history_span + (duration - history_span) * np.random.uniform(low=0.0, high=1.0, size=num_samples)

    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None) -> torch.Tensor:
        if current_times is None:
            current_times = self._sample_amp_times(num_samples)
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
    all_joint_acc: torch.Tensor,
    hip_knee_ankle_applied_torque: torch.Tensor,
    ankle_joint_pos: torch.Tensor,
    ankle_soft_joint_pos_limits: torch.Tensor,
    hip_deviation: torch.Tensor,
    arm_deviation: torch.Tensor,
    torso_deviation: torch.Tensor,
):
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - root_lin_vel_b[:, :2]), dim=1)
    rew_task_lin_vel = rew_lin_vel_xy * torch.exp(-lin_vel_error / rew_track_sigma)

    ang_vel_error = torch.square(commands[:, 2] - root_ang_vel_b[:, 2])
    rew_task_ang_vel = rew_ang_vel_z * torch.exp(-ang_vel_error / rew_track_sigma)

    rew_termination = rew_scale_termination * reset_terminated.float()
    rew_action_rate_l2 = rew_scale_action_rate_l2 * torch.sum(torch.square(actions - prev_actions), dim=1)

    out_of_limits = -(ankle_joint_pos - ankle_soft_joint_pos_limits[:, :, 0]).clip(max=0.0)
    out_of_limits += (ankle_joint_pos - ankle_soft_joint_pos_limits[:, :, 1]).clip(min=0.0)
    rew_joint_pos_limits = rew_scale_joint_pos_limits * torch.sum(out_of_limits, dim=1)

    rew_joint_acc_l2 = rew_scale_joint_acc_l2 * torch.sum(torch.square(all_joint_acc), dim=1)
    rew_dof_torques_l2 = rew_scale_dof_torques_l2 * torch.sum(torch.square(hip_knee_ankle_applied_torque), dim=1)
    rew_flat_orientation_l2 = rew_scale_flat_orientation_l2 * torch.sum(
        torch.square(projected_gravity_b[:, :2]), dim=1
    )
    rew_lin_vel_z_l2 = rew_scale_lin_vel_z_l2 * torch.square(root_lin_vel_b[:, 2])
    rew_ang_vel_xy_l2 = rew_scale_ang_vel_xy_l2 * torch.sum(torch.square(root_ang_vel_b[:, :2]), dim=1)
    rew_joint_deviation_hip = rew_scale_joint_deviation_hip * torch.sum(torch.square(hip_deviation), dim=1)
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
