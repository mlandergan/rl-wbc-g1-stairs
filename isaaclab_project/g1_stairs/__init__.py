"""Vision-based AMP stair climbing for the Unitree G1.

Forked from `rl-wbc-g1-amp`'s `g1_amp` package (velocity-tracking task reward + AMP style
reward, direct-workflow env adapted from linden713/humanoid_amp's G1AmpEnv, BSD-3-Clause).
This fork adds a procedural pyramid-stairs terrain with a difficulty curriculum, and replaces
flat uniform velocity command sampling with a position-based, direction-biased command toward the
tile's centre platform — see g1_stairs_env_cfg.py / g1_stairs_env.py.

Registers `Isaac-G1-AMP-Stairs-Direct-v0` (and its `-Play` variant).
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-G1-AMP-Stairs-Direct-v0",
    entry_point=f"{__name__}.g1_stairs_env:G1StairsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_stairs_env_cfg:G1StairsEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_g1_stairs_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_g1_stairs_cfg.yaml",
    },
)

gym.register(
    id="Isaac-G1-AMP-Stairs-Direct-Play-v0",
    entry_point=f"{__name__}.g1_stairs_env:G1StairsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_stairs_env_cfg:G1StairsEnvCfg_PLAY",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_g1_stairs_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_g1_stairs_cfg.yaml",
    },
)
