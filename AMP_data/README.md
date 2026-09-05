# AMP motion reference and result media

## Motion reference

`parkour_motion_without_run_retargetted.npz` + `parkour_motion_without_run.yaml` are the AMP
reference-motion clip used for imitation-style reward shaping (see `project_description.md`'s
"Adversarial Motion Priors" section). `render_parkour_npz_mp4.py` and
`visualize_parkour_npz_rerun.py` are the two tools used to preview it — the first renders it to
an MP4, the second replays it interactively via [rerun](https://rerun.io/).

## Terrain/setup verification (`stairs_verify_results/`)

Screenshots and short clips backing the verification claims in the main `README.md` and in
`g1_stairs_env.py`'s own comments: the procedural stairs terrain's goal markers
(`goal_marker_check.mp4`, `marker_frame_*.png`), the stairs contact sensor
(`contact_sensor_check.mp4`, `contact_check_frame.png`), and a zero-action baseline rollout on
the terrain (`stairs_zero_action-step-0.mp4`).

## Result clips and frames

**`climbing_highlight.mp4` is not policy behavior** — its file date (Aug 9) predates every
training run in `logs/` (earliest: Aug 12). It's a rendering of the raw AMP reference-motion
clip (`parkour_motion_without_run_retargetted.npz`) played back inside the actual stairs task
scene via `render_parkour_npz_mp4.py`, showing what the human-motion-capture reference the AMP
reward tries to imitate looks like — not a trained checkpoint climbing anything. Likewise
`final_checkpoint_50000.mp4` (Aug 12) is a real policy eval clip, but shows flat-ground running
with no stairs terrain visible — not usable as climbing evidence either. Neither of these two
files should be read as demonstrating successful stairs-climbing; see the main README's status
section for what is and isn't actually confirmed. The `track_frame_*.png`, `final_frame_*.png`,
`lateclip_*.png`, and `endcheck_*.png` stills are frame grabs taken from checkpoint eval videos
during development, kept as quick visual references rather than a curated gallery.
