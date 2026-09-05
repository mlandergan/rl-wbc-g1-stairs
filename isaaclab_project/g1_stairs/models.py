"""Custom multi-modal (proprioception + depth-CNN) skrl models for the policy and value
networks -- see WORKING_NOTES.md's "Custom multi-modal skrl model" section for why this can't
be expressed via skrl's YAML model-instantiator shorthand (no primitive for "split this obs
tensor into two named sub-ranges and route to different sub-networks") and has to be a real
Python `nn.Module`-based model instead.

The discriminator is intentionally NOT defined here -- it stays exactly what it already was
(a plain MLP over amp_observation_space, no depth input; confirmed this matches InstinctLab's
own actual discriminator design too, see WORKING_NOTES.md), so it's built via skrl's own
`deterministic_model` instantiator utility function directly in train_amp_depth.py rather than
duplicated here.

VERIFIED (2026-09-05) against the actually-installed skrl 1.4.3 source, pulled directly off the
training VM's Docker image (models/torch/{base,gaussian,deterministic}.py): `Model.__init__`/
`GaussianMixin.__init__`/`DeterministicMixin.__init__` are positional (not keyword-only) in this
version, and `compute()`'s documented convention is exactly what's used below --
`inputs["states"]`, policy returns `(mean, log_std_parameter, {})`, value returns `(value, {})`.
(An earlier code-review pass flagged both of these as bugs by checking against skrl's unreleased
GitHub `main` branch, which does use keyword-only constructors and an `"observations"` key --
that branch is not what's installed. See wasabi_amp.py's module docstring for the fuller version-
mismatch story.)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

# Matches G1StairsEnvCfg's observation_space layout exactly (g1_stairs_env_cfg.py) --
# proprio(71 task-obs + 30 key-body + 3 command) ++ depth(16x16 flattened), in that order,
# because that's the order _get_observations concatenates them in.
PROPRIO_DIM = 71 + 3 * 10 + 3  # 104
DEPTH_SIZE = 16  # G1StairsEnvCfg.DEPTH_FINAL_SIZE
DEPTH_DIM = DEPTH_SIZE * DEPTH_SIZE  # 256

# InstinctLab's own DepthEncoderConv2dCfg shape (WORKING_NOTES.md) -- channels=[4], kernel=3,
# stride=1, padding=1, single conv layer + maxpool, then an MLP head down to a 128-dim feature.
_CONV_OUT_CHANNELS = 4
_CONV_KERNEL_SIZE = 3
_CONV_STRIDE = 1
_CONV_PADDING = 1
_MAXPOOL_KERNEL = 2  # halves the spatial size once: 16x16 -> 8x8
_ENCODER_HIDDEN_SIZES = [256, 256]
_ENCODER_OUTPUT_SIZE = 128


class DepthEncoder(nn.Module):
    """Conv2d depth branch shared, identically, by both DepthAmpPolicy and DepthAmpValue --
    same architecture InstinctLab uses for both its policy and critic encoders
    (`encoder_configs`/`critic_encoder_configs`, WORKING_NOTES.md), so mirroring that here as a
    single shared module used by both rather than two separately-initialized copies of the same
    shape.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=_CONV_OUT_CHANNELS,
            kernel_size=_CONV_KERNEL_SIZE,
            stride=_CONV_STRIDE,
            padding=_CONV_PADDING,
        )
        self.pool = nn.MaxPool2d(_MAXPOOL_KERNEL)
        conv_out_spatial = DEPTH_SIZE // _MAXPOOL_KERNEL  # 8
        conv_out_dim = _CONV_OUT_CHANNELS * conv_out_spatial * conv_out_spatial  # 4*8*8 = 256

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
        extraction, no image preprocessing.
        """
        depth_img = depth_flat.view(-1, 1, DEPTH_SIZE, DEPTH_SIZE)
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
        proprio, depth_flat = obs[:, :PROPRIO_DIM], obs[:, PROPRIO_DIM:]
        feat = torch.cat([proprio, self.depth_encoder(depth_flat)], dim=-1)
        return self.head(feat), self.log_std_parameter, {}


class DepthAmpValue(DeterministicMixin, Model):
    """Value network: same architecture as DepthAmpPolicy (own DepthEncoder instance, not
    shared weights with the policy's -- matching InstinctLab's separate
    encoder_configs/critic_encoder_configs), head output size 1 instead of action_space.
    """

    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.depth_encoder = DepthEncoder()
        head_in = PROPRIO_DIM + _ENCODER_OUTPUT_SIZE
        self.head = nn.Sequential(
            nn.Linear(head_in, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(),
            nn.Linear(128, 1),
        )

    def compute(self, inputs, role):
        obs = inputs["states"]
        proprio, depth_flat = obs[:, :PROPRIO_DIM], obs[:, PROPRIO_DIM:]
        feat = torch.cat([proprio, self.depth_encoder(depth_flat)], dim=-1)
        return self.head(feat), {}
