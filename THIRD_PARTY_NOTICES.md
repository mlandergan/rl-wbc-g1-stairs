# Third-Party Notices

This repository is forked from `rl-wbc-g1-amp`'s `g1_amp` task, whose core
environment (`G1AmpEnv`, motion-imitation reward structure) is itself adapted
from [`linden713/humanoid_amp`](https://github.com/linden713/humanoid_amp)'s
`G1AmpEnv` (BSD-3-Clause). See the citation at the top of
`isaaclab_project/g1_stairs/g1_stairs_env.py`. Modifications and additions
made in this repository are licensed under the same BSD-3-Clause terms — see
the root `LICENSE`.

`isaaclab_project/g1_stairs/motions/motion_loader.py` retains its original
upstream copyright and license notice:

```
Copyright (c) 2022-2025, The Isaac Lab Project Developers.
SPDX-License-Identifier: BSD-3-Clause
```

## Code adapted/copied directly (not just design reference)

- `isaaclab_project/g1_stairs/agents/wasabi_amp.py`: `WasabiAMP._update()` is
  `skrl.agents.torch.amp.amp.AMP._update()` (the actually-installed skrl 1.4.3
  version, not the newer GitHub `main` branch — see the file's own module
  docstring for why that distinction mattered here)
  ([`Toni-SM/skrl`](https://github.com/Toni-SM/skrl), MIT-licensed) copied
  essentially verbatim, with three discriminator-training formulas swapped
  for [WASABI](https://github.com/martius-lab/wasabi)'s own versions, ported
  from `instinct_rl/algorithms/wasabi.py`
  ([`project-instinct/instinct_rl`](https://github.com/project-instinct/instinct_rl),
  Modified MIT License, copyright Ziwen Zhuang — standard MIT terms plus a UI
  credit requirement only above 100M monthly users or $20M/month revenue,
  irrelevant at this project's scale). See `WORKING_NOTES.md` for exactly
  which three formulas differ and why.
- `scripts/train_amp_depth.py` is adapted from Isaac Lab's stock
  `scripts/reinforcement_learning/skrl/train.py`
  ([`isaac-sim/IsaacLab`](https://github.com/isaac-sim/IsaacLab),
  BSD-3-Clause) — env construction and Hydra config loading kept as-is,
  only the agent/model instantiation section was rewritten (see the script's
  own docstring for why).
- `isaaclab_project/g1_stairs/models.py`'s `DepthAmpPolicy`/`DepthAmpValue`
  are original code written for this project, subclassing `skrl`'s
  `GaussianMixin`/`DeterministicMixin`/`Model` (MIT-licensed) per its
  documented public extension pattern, not copied from any example.

The following runtime dependencies are installed separately and remain
subject to their own licenses:

- NVIDIA Isaac Sim
- NVIDIA Isaac Lab
- PyTorch
- skrl

## InstinctLab-derived reward formulas (design facts/formulas, not copied source text)

**Correction (2026-09-05):** the "Reference material — not vendored" section
below originally claimed no InstinctLab code is reused, only Isaac Lab's own
BSD-3-Clause primitives. That was true when written but is no longer
accurate — the terms below are InstinctLab's own *original* reward-function
designs (not thin wrappers around Isaac Lab's stock `isaaclab.envs.mdp`
library), so their real formulas/weights/thresholds have been ported
following the same rule as everywhere else in this file: design facts and
numeric weights are not copyrightable the way source-code text is, so each
term below is this project's own clean-room implementation (matching this
codebase's own style/conventions, in `g1_stairs_env.py`/`g1_stairs_env_cfg.py`
directly, not InstinctLab's classes), citing InstinctLab's real
file/line/weight as the source of the *design decision*, never copy-pasted
InstinctLab source text. All terms below are pinned to InstinctLab commit
`ba28d3d2655b15a19b729476a630937a19610a3b` (see the "Reference material"
section for how to obtain that exact commit).

- `rew_dont_wait` / `rew_stand_still` (added 2026-08-13): ported from
  `source/instinctlab/instinctlab/tasks/parkour/mdp/rewards.py:38-52`
  (`stand_still`) and `:101-110` (`dont_wait`) — InstinctLab's own
  `instinctlab.tasks.parkour.mdp` module (not Isaac Lab's stock `mdp`).
- `rew_heading_error` (added 2026-09-05): ported from
  `source/instinctlab/instinctlab/tasks/parkour/mdp/rewards.py:94-98`
  (`heading_error`) — same module as above; weight `-1.0` is InstinctLab's
  `G1Rewards.heading_error` (`parkour_env_cfg.py:666`).
- `rew_feet_air_time` (added 2026-09-05): ported from
  `source/instinctlab/instinctlab/tasks/parkour/mdp/rewards.py:14-35`
  (`feet_air_time`) — same module; this is InstinctLab's own biped-specific
  override of the name, NOT Isaac Lab's stock `mdp.feet_air_time` (their
  local module's star-import order means `mdp.feet_air_time` in
  `parkour_env_cfg.py` resolves to this override, not the stock one). Weight
  `0.5`, `vel_threshold=0.15` from `G1Rewards.feet_air_time`
  (`parkour_env_cfg.py:679-687`).
- `rew_feet_slide` (added 2026-09-05): ported from
  `source/instinctlab/instinctlab/instinctlab/envs/mdp/rewards/regularizations.py:780-805`
  (`contact_slide`) — InstinctLab's own general `instinctlab.envs.mdp`
  library (shared across their tasks, still their own CC BY-NC code, not
  Isaac Lab's). Weight `-0.4`, `threshold=1.0` from `G1Rewards.feet_slide`
  (`parkour_env_cfg.py:688-696`).
- `rew_energy` (added 2026-09-05): ported from the same
  `instinctlab/envs/mdp/rewards/regularizations.py:33-48` (`motors_power_square`).
  Weight `-5e-5`, `normalize_by_stiffness=True` from `G1Rewards.energy`
  (`parkour_env_cfg.py:751-758`).
- `rew_torque_limits` (added 2026-09-05): ported from the same
  `instinctlab/envs/mdp/rewards/regularizations.py:760-777`
  (`applied_torque_limits_by_ratio`). Weight `-0.01`, `limit_ratio=0.8` from
  `G1Rewards.torque_limits` (`parkour_env_cfg.py:780-787`).
- `rew_joint_deviation_hip`'s formula (changed 2026-09-05 from an L1
  sum-of-abs to a sum-of-squares kernel, weight `-0.1` -> `-0.5`): ported
  from the same `instinctlab/envs/mdp/rewards/regularizations.py:243-251`
  (`joint_deviation_square`), matching InstinctLab's own override of this
  specific term (`G1Rewards.joint_deviation_hip`, `parkour_env_cfg.py:697-701`)
  — InstinctLab does NOT use this squared kernel for its other
  joint-deviation terms (only this one), so only `rew_joint_deviation_hip`
  was changed, not `rew_joint_deviation_arms`/`rew_joint_deviation_torso`
  (which have no InstinctLab equivalent at all — see
  `g1_stairs_env_cfg.py`'s own comment).

`rew_undesired_contacts` (added 2026-09-05) is **not** in this section — its
real function (`undesired_contacts`) is not defined anywhere in InstinctLab's
own source tree (confirmed by grepping the pinned submodule), meaning
`mdp.undesired_contacts` in `G1Rewards` resolves to Isaac Lab's own stock
`isaaclab.envs.mdp.rewards.undesired_contacts` (BSD-3-Clause) via the
`instinctlab.tasks.parkour.mdp` module's `from isaaclab.envs.mdp import *`
star-import — freely reimplementable with no CC BY-NC concern, same as
`rew_dof_vel_limits`/`rew_dof_vel_l2` below.

## Reference material — not vendored

[`project-instinct/InstinctLab`](https://github.com/project-instinct/InstinctLab)
(CC BY-NC 4.0, non-commercial) was used as a **design reference only** during
development. Two categories of borrowing from it exist in this repository,
per the section above and Isaac Lab's own BSD-3-Clause reward primitives
used elsewhere (`isaaclab.envs.mdp`, freely reusable — confirmed by checking
each `mdp.xxx` name's actual resolved module in InstinctLab's own
`tasks/parkour/mdp/__init__.py` star-import chain before treating it as
"stock"): (1) numeric config facts about InstinctLab's own published
research artifact (terrain dimensions, camera pose/FOV, episode length,
PhysX buffer sizes, reward weights that are themselves just InstinctLab's
chosen values for Isaac Lab's own stock reward functions) — these are design
facts about a public artifact, not code; and (2) InstinctLab's own *original*
reward-function formulas (listed above), each reimplemented as this
project's own clean code, not copy-pasted. No InstinctLab source file is
vendored or redistributed in this repository either way. Because
InstinctLab's license is non-commercial, this repository does not vendor or
redistribute a copy of it.
If you want the exact reference version used during design, clone it
yourself at commit `ba28d3d2655b15a19b729476a630937a19610a3b`:

```bash
git clone https://github.com/project-instinct/InstinctLab.git reference/InstinctLab
cd reference/InstinctLab && git checkout ba28d3d2655b15a19b729476a630937a19610a3b
```

(A pinned git submodule pointing at this same commit is also configured at
`reference/InstinctLab` — `git submodule update --init` fetches it directly
from InstinctLab's own repository; nothing from it is stored in this
repository's own history.)

This file records the review requirement; it is not a substitute for the
applicable upstream license texts or legal review.
