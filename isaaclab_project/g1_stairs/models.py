"""Policy and value networks: a proprioception MLP plus a Conv2d depth encoder.

skrl's YAML model instantiator has no primitive for "split this observation into named ranges and
route them to different sub-networks", so these are real nn.Modules. The observation is laid out
as [proprio | depth | privileged]: the policy reads the first two, the value model reads all
three. That is how the critic gets privileged input while skrl hands both models the same tensor.

The discriminator is deliberately not here -- it stays a plain MLP over the AMP observation with
no depth input, built by skrl's own instantiator in train_amp_depth.py.

Written against skrl 1.4.3, whose Model/GaussianMixin/DeterministicMixin constructors are
positional and whose compute() takes inputs["states"]. Later skrl branches differ on both counts.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

POLICY_PROPRIO_HISTORY_LEN = 8
PROPRIO_DIM = POLICY_PROPRIO_HISTORY_LEN * (70 + 3 * 10) + 3
DEPTH_SIZE = 16
DEPTH_HISTORY_LEN = 8
DEPTH_DIM = DEPTH_HISTORY_LEN * DEPTH_SIZE * DEPTH_SIZE

PRIVILEGED_DIM = 5
OBS_DIM = PROPRIO_DIM + DEPTH_DIM + PRIVILEGED_DIM

_CONV_OUT_CHANNELS = 4
_CONV_KERNEL_SIZE = 3
_CONV_STRIDE = 1
_CONV_PADDING = 1
_MAXPOOL_KERNEL = 2
_ENCODER_HIDDEN_SIZES = [256, 256]
_ENCODER_OUTPUT_SIZE = 128


class DepthEncoder(nn.Module):
    """Conv2d depth branch shared, identically, by both DepthAmpPolicy and DepthAmpValue --
    same architecture InstinctLab uses for both its policy and critic encoders
    (`encoder_configs`/`critic_encoder_configs`), so mirroring that here as a
    single shared module used by both rather than two separately-initialized copies of the same
    shape.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=DEPTH_HISTORY_LEN,
            out_channels=_CONV_OUT_CHANNELS,
            kernel_size=_CONV_KERNEL_SIZE,
            stride=_CONV_STRIDE,
            padding=_CONV_PADDING,
        )
        self.pool = nn.MaxPool2d(_MAXPOOL_KERNEL)
        conv_out_spatial = DEPTH_SIZE // _MAXPOOL_KERNEL
        conv_out_dim = _CONV_OUT_CHANNELS * conv_out_spatial * conv_out_spatial

        mlp_layers = []
        in_dim = conv_out_dim
        for hidden in _ENCODER_HIDDEN_SIZES:
            mlp_layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
            in_dim = hidden
        mlp_layers.append(nn.Linear(in_dim, _ENCODER_OUTPUT_SIZE))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, depth_flat: torch.Tensor) -> torch.Tensor:
        """depth_flat: (N, DEPTH_DIM), already cropped/blurred/normalized to [0,1] by
        g1_stairs_env.py's _process_depth_image -- this module only does the Conv2d/MLP feature
        extraction, no image preprocessing. Laid out newest-frame-first, matching the order
        _get_observations gathers them in, so channel 0 is always the current view.
        """
        depth_img = depth_flat.view(-1, DEPTH_HISTORY_LEN, DEPTH_SIZE, DEPTH_SIZE)
        x = torch.relu(self.conv(depth_img))
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        return self.mlp(x)


class DepthAmpPolicy(GaussianMixin, Model):
    """Policy network: concat(raw proprio(104), depth-CNN feature(128)) -> MLP[256,128,128] ->
    action_space. The MLP head keeps this project's original policy capacity
    (skrl_g1_stairs_cfg.yaml's `[256,128,128]`) unchanged -- the depth branch is purely additive,
    not a replacement for it.
    """

    def __init__(
        self, observation_space, action_space, device,
        clip_actions=False, clip_log_std=True, min_log_std=-20.0, max_log_std=2.0,
        initial_log_std=0.0,
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction="sum")

        self.depth_encoder = DepthEncoder()
        head_in = PROPRIO_DIM + _ENCODER_OUTPUT_SIZE
        self.head = nn.Sequential(
            nn.Linear(head_in, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(),
            nn.Linear(128, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(initial_log_std * torch.ones(self.num_actions))

    def compute(self, inputs, role):
        obs = inputs["states"]
        proprio = obs[:, :PROPRIO_DIM]
        # Bounded slice: the privileged block sits after the depth image, and letting it into
        # the policy's encoder would defeat the point of it being critic-only.
        depth_flat = obs[:, PROPRIO_DIM : PROPRIO_DIM + DEPTH_DIM]
        feat = torch.cat([proprio, self.depth_encoder(depth_flat)], dim=-1)
        return self.head(feat), self.log_std_parameter, {}


class DepthAmpValue(DeterministicMixin, Model):
    """Value network: same architecture as DepthAmpPolicy (own DepthEncoder instance, not
    shared weights with the policy's -- matching InstinctLab's separate
    encoder_configs/critic_encoder_configs), head output size 1 instead of action_space.

    PRIVILEGED (changed 2026-09-06): unlike the policy, this model also consumes the
    PRIVILEGED_DIM features at the tail of the observation -- terrain clearances, sampled friction
    and terrain level, none of which a real robot could measure. Its first layer is correspondingly
    PRIVILEGED_DIM wider than the policy's. This mirrors InstinctLab, whose critic observation adds
    base_lin_vel and skips the actor's noise corruption (parkour_env_cfg.py:470,502).
    """

    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.depth_encoder = DepthEncoder()
        head_in = PROPRIO_DIM + _ENCODER_OUTPUT_SIZE + PRIVILEGED_DIM
        self.head = nn.Sequential(
            nn.Linear(head_in, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(),
            nn.Linear(128, 1),
        )

    def compute(self, inputs, role):
        obs = inputs["states"]
        proprio = obs[:, :PROPRIO_DIM]
        depth_flat = obs[:, PROPRIO_DIM : PROPRIO_DIM + DEPTH_DIM]
        privileged = obs[:, PROPRIO_DIM + DEPTH_DIM :]
        feat = torch.cat([proprio, self.depth_encoder(depth_flat), privileged], dim=-1)
        return self.head(feat), {}
