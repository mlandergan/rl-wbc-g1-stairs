"""Vision-based AMP stair climbing for the Unitree G1 (see ../../project_description.md).

Forked from `rl-wbc-g1-amp`'s `g1_amp` package (velocity-tracking task reward + AMP style
reward, direct-workflow env adapted from linden713/humanoid_amp's G1AmpEnv, BSD-3-Clause).
This fork adds a procedural 5-step American stairs terrain and replaces flat uniform velocity
command sampling with the "Hiking in the Wild" paper's position-based, direction-biased command
(goal = the stairs' center plateau) — see g1_stairs_env_cfg.py / g1_stairs_env.py.

Registers `Isaac-G1-AMP-Stairs-Direct-v0` (and its `-Play` variant). Bootstrapped with
rl-wbc-g1-amp's existing Strut Walking reference clip (G1_strut_walk.npz) as a placeholder AMP
motion until stairs-specific reference motion is sourced and converted (see
project_description.md's Dataset section — open task, not yet done).
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
