# rl-wbc-g1-stairs

Vision-based AMP stair climbing for the Unitree G1. Part of the `rl-wbc-g1-*` series (see
[`rl-wbc-g1-baseline`](https://github.com/mlandergan/rl-wbc-g1-baseline) →
[`rl-wbc-g1-amp`](https://github.com/mlandergan/rl-wbc-g1-amp)). Full design spec:
[`project_description.md`](project_description.md).

## Origin and license

This repository forks `rl-wbc-g1-amp`'s `g1_amp` task. Its core environment (`G1AmpEnv`
inheritance, motion-imitation reward structure) is adapted from
[`linden713/humanoid_amp`](https://github.com/linden713/humanoid_amp) (BSD-3-Clause) — see the
citation at the top of `isaaclab_project/g1_stairs/g1_stairs_env.py`. Licensed under
BSD-3-Clause, matching upstream — see `LICENSE`. See `THIRD_PARTY_NOTICES.md` for full
attribution, including a design-reference-only (not vendored) comparison against
[`project-instinct/InstinctLab`](https://github.com/project-instinct/InstinctLab) (CC BY-NC 4.0).

## Status: work in progress, transparently

Six real training runs have happened on this code (`logs/skrl/g1_stairs/`), a real
reward-hacking bug was found and a fix was applied — but verification of that fix is partial and
uneven, and the two newest reward terms have never been run on a GPU at all. This section states
plainly what is and isn't confirmed, rather than a polished "done" claim.

| Task | Status |
|---|---|
| 1. Fork/scaffold from `rl-wbc-g1-amp` | done |
| 2. Procedural 5-step American stairs terrain | done, GPU-verified |
| 3. Position-based, direction-biased velocity command (paper's Eq. 8-9) | done |
| 4. Border-ring spawn placement + gym registration (`Isaac-G1-AMP-Stairs-Direct-v0`) | done |
| 5. AMP reference-motion source decision + conversion script | **not done** — still uses `rl-wbc-g1-amp`'s flat-ground Strut Walk clip as a placeholder |
| 6. Reward/termination design pass | partially done — see below |
| 7. Small-scale smoke test | done |
| 8. Full training run | six runs completed/attempted; only one ran to a full 50000 steps (see caveats below) |
| 9. Metrics + video collection | partial — see "What's actually confirmed" below |
| 10. Blog write-up | not started |

### What's actually confirmed

- **Terrain, goal markers, and the stairs contact sensor are GPU-verified.** Goal markers render
  exactly on each tile's center platform (2026-08-12); see `AMP_data/stairs_verify_results/` for
  the actual check videos/screenshots.
- **A real reward-hacking bug was found and diagnosed correctly.** A full 50000-step run
  (`logs/skrl/g1_stairs/2026-08-12_23-34-22_ppo_torch/`) converged to a policy that camps at the
  base of the stairs to avoid a termination penalty, rather than climbing — a genuine finding,
  not a training failure.
- **A fix was applied**: heading-relativized AMP observations (so the discriminator judges gait
  style independent of which of the 4 stairs faces the robot approaches from) plus
  `rew_dont_wait`/`rew_stand_still` anti-dawdling reward terms, ported from InstinctLab's real
  reward functions after a direct code comparison (kept local — see `instinctlab_comparison.md`
  is not included in this repository, but the citation and reasoning are in
  `g1_stairs_env.py`'s module docstring).

### What's genuinely uncertain — read before trusting "fixed"

- The run most associated with this fix
  (`logs/skrl/g1_stairs/2026-08-13_04-38-05_ppo_torch/`) only reached **13000 of a planned 20000
  steps**. The code's module docstring states this was "VERIFIED on the GPU VM (2026-08-13):
  checkpoint video at step 13000/20000 showed real goal-reaching" — but **that specific video is
  not present anywhere in this repository or its logs** (only a step-0, pre-training sanity clip
  exists in that run's `videos/play/`), so this claim could not be independently re-confirmed
  while preparing this repository. Take it as a claim made at the time, not something you can
  check yourself from what's here.
- **The codebase's own comments are not fully consistent about this.** The module docstring
  calls the anti-dawdling fix "VERIFIED on the GPU VM," but a separate inline comment on the
  exact same reward terms (`g1_stairs_env.py`, near `rew_dont_wait`/`rew_stand_still`) says
  "untested on GPU yet, kept easy to isolate/remove" — the same phrasing used for the terrain-aware
  rewards, which genuinely never ran. This may just be a stale comment copied across both
  additions rather than a real contradiction, but it could not be resolved with confidence, so
  both readings are noted here rather than picking one.
- A later run (`2026-08-13_05-25-59_ppo_torch`) was abandoned at only 2000 steps, for an unknown
  reason (crash, manual stop, or superseded by further edits — no record survives either way).
- **Two reward-code changes have never been run on a GPU at all**: the terrain-aware rewards
  (`rew_feet_height_error`, `rew_edge_penetration`, added 2026-08-12 — explicitly flagged in-code
  as "NOT yet run on the GPU VM") and the height-progress diagnostic (`rew_max_height`, added
  2026-08-14, default weight 0.0 so it has no training effect either way — added purely to watch
  climbing progress in TensorBoard once it does get run). Both are only `py_compile`-checked for
  syntax.
- **Two result videos in `AMP_data/` are not policy-climbing evidence, despite names/appearance
  suggesting otherwise** — see `AMP_data/README.md` for exactly what `climbing_highlight.mp4`
  and `final_checkpoint_50000.mp4` actually show (a pre-training reference-motion rendering and a
  flat-ground eval clip, respectively). Neither should be read as demonstrating a successful
  climb.

### Net assessment

The reward-hacking diagnosis is solid and the fix is a real, targeted response to it (not a
guess) — but a clean, complete, independently-reproducible confirmation that it works does not
currently exist in this repository. The honest next step, not yet done: one full training run
on the current code (including the two never-GPU-tested reward additions) to completion, with
its checkpoint videos actually kept and included.

## Implementation-risk notes (unrelated to the reward-hacking question above)

Several geometry/API assumptions were written before Isaac Sim was available to test locally,
and were later individually confirmed or left open on the GPU VM — see the inline comments at
each spot in `g1_stairs_env.py` / `g1_stairs_env_cfg.py` for specifics:

- Exact pyramid-stairs step count from `STAIRS_TERRAIN_CFG`'s tile-size derivation — GPU-verified
  (goal markers land exactly on the center platform, consistent with the intended geometry).
- `self.terrain.env_origins` behavior with an 8-tile grid shared across all envs — a
  rendering/video-only concern (per-env GPU collision filtering makes multiple envs sharing a
  tile physically fine), not independently re-verified beyond the training runs completing.
- `FlatPatchSamplingCfg`/`TerrainImporter` import paths — used at the paths matching
  `reference/InstinctLab`'s own usage of the same Isaac Lab classes; not separately verified.

## What's still a placeholder, not a bug

- **AMP reference motion** is still `rl-wbc-g1-amp`'s Strut Walking clip (`G1_strut_walk.npz`) —
  general locomotion style, not stairs-specific (Task 5, not done).
  `AMP_data/parkour_motion_without_run_retargetted.npz` (the reference paper's own combined
  walking dataset, confirmed 379.64s at 50fps against the paper's quoted `T = 379.62 s` for
  `D_walk`) is a real candidate source but has not been run through a `convert_gmr_to_npz.py`-style
  conversion into this env's actual `MotionLoader` schema.
- **Reward/termination weights** are inherited unchanged from `rl-wbc-g1-amp`'s flat-ground
  values except for the additions described above — not otherwise retuned for stairs.
- **`command_pos_k_v`/`command_pos_k_w`** (the position-based command's gain constants) are
  starting-point guesses — the paper doesn't give numeric values for these.

## Repo layout

Forked from `rl-wbc-g1-amp` — same `docker/` + `scripts/` GCP VM lifecycle (build, train, eval,
sync, teardown), same skrl-based AMP training loop, same `MotionLoader`/`.npz` motion convention.
See `project_description.md`'s "Repository / Fork Plan" section for exactly what's copied vs.
net-new.
