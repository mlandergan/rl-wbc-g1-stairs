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

## Reference material — not vendored

[`project-instinct/InstinctLab`](https://github.com/project-instinct/InstinctLab)
(CC BY-NC 4.0, non-commercial) was used as a **design reference only** during
development — confirmed by direct code comparison (see
`instinctlab_comparison.md`, kept local/not included in this repository) that
no InstinctLab code is reused in `isaaclab_project/g1_stairs/`; only Isaac
Lab's own BSD-3-Clause primitives are used. Because InstinctLab's license is
non-commercial, this repository does not vendor or redistribute a copy of it.
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
