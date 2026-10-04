# Robot4Art simulation + composition pipeline

Two pieces, independently testable:

1. **Semantic composition** (`composition.py`) -- turns a visitor's kiosk
   answers into a <=10-stroke artwork plan, colored with a small set of
   common pen colors (not custom-mixed paint). Pure Python/numpy/matplotlib,
   no simulator or GPU needed.
2. **Isaac Sim validation** (`franka_paint_sim.py`) -- drives a robot arm
   through a stroke plan on a virtual canvas and checks the executed
   end-effector trajectory actually traces the intended composition.
   Needs Isaac Sim (see below).

## Status

**Composition (`composition.py`): working.** Implements the "LLM-guided
procedural composition" path from the research proposal (Sec.
"Semantic-to-Artistic Composition", path 2): a small library of
parametric grammars (`arc_over_line`, `nested_curves`,
`radiating_strokes`) filled in with color/scale/placement derived from
the visitor's answers, each extended with small anchored accent dabs up
to a 10-stroke budget (`MAX_STROKES`). Colors come from a fixed set of
common pen colors (`STANDARD_PENS`) picked by nearest hue to the
visitor's answer, not generated freely -- the robot holds a small set of
interchangeable pens, not custom-mixed paint.

The visitor also picks a **style** (`STYLES`: Modernist Gesture, Impressionist
Bloom, Ink Wash Minimal), which picks the grammar directly -- this turned
out to look better and be more interactive than the earlier hash-derived
1-in-3 assignment. Separately, the visitor can pick a **brush/pen tool**
(`BRUSHES`: Fine Pen, Marker, Brush, Watercolor Dabs), independent of
style -- the robot's tool holder carries a few distinct tool types, not
just a few colors, so style (composition shape) and tool (rendering
width/taper/texture) compose freely; leaving the tool on "auto" falls
back to a sensible per-style default (`_DEFAULT_BRUSH_FOR_STYLE`) that
matches what shipped before the tool became independently selectable.
`rationale_for(answers, brief)` generates a short, templated explanation
of *why* the piece looks the way it does, tying the pen colors, grammar,
accents, and chosen tool back to the visitor's own answers -- the "why
created this way" explanation shown on the web demo as the piece is
painted, standing in for the same explanation an LLM would write.

The "which grammar, what parameters" choice the proposal originally
assigned to an LLM is, net of the visitor's style pick, a deterministic
function of the answers (`derive_brief`) rather than an actual model call
-- no model API key is available in this environment yet. Swapping in a
real model only requires changing `derive_brief` and `rationale_for`;
everything downstream (grammar rendering, Stroke output) is unaffected.
Run `python3 sim/preview_compositions.py` to render example outputs
(including deliberately non-default style/brush pairings, to show
they're independent choices) to `sim/output/compositions/examples.png`
and print their rationale text, without needing Isaac Sim at all. The other path in the
proposal (direct CLIPDraw-style stroke optimization against a
vision-language embedding) is not implemented.

**Isaac Sim (`franka_paint_sim.py`): working**, via Docker under this
account (see "Running the Isaac Sim validation" below), and wired
directly to `composition.py` (a fixed example `PromptAnswers`, standing
in for live kiosk input). Verified end-to-end with a Franka arm: all 10
strokes complete and the recorded end-effector trace visibly tracks the
intended composition (see `sim/output/paint_trace.png` after a run).

**Known limitation:** this only validates *motion* (the end effector
visits the right places at the right simulated times) -- there is no
ink/paint deposition model yet, so nothing is actually marked on the
virtual canvas. There is also a small systematic offset between intended
and executed traces, likely from RMPFlow's reactive convergence behavior.

**Video for the website (`render_paint_video.py`): working**, but not the
way it was first attempted. An Isaac Sim camera render was tried
extensively against this project's Docker setup (several camera-binding
approaches, explicit scene lighting, forcing the Replicator orchestrator
to step) -- every attempt returned an identical, scene-independent
placeholder frame regardless of camera position/orientation, strongly
suggesting the offscreen render product isn't actually wired to the stage
in this headless container configuration. Revisit if that gets resolved.
In the meantime, `render_paint_video.py` animates the *real* recorded
trace from `trace.json` with matplotlib + ffmpeg instead: a pen marker
traces the actual simulated end-effector path, drawing each stroke in its
color. Authentic data, just not a 3D render. Output feeds directly into
`docs/media/` for the GitHub Pages site.

**Not yet done:** Kinova and xArm adapters (the proposal's robot-agnostic
execution layer), a real model call in `derive_brief`, an actual Isaac
Sim camera render (see above), and real-time-budget validation against
the live-kiosk timing target (this run's *simulated* time was ~272s for
10 strokes, which is not directly comparable to real wall-clock
execution speed).

## Files

- `composition.py` -- visitor answers -> artistic brief -> <=10-stroke
  plan, colored from a fixed set of common pen colors. No simulator
  needed; see `preview_compositions.py`.
- `preview_compositions.py` -- renders example compositions to
  `output/compositions/examples.png`. No simulator needed.
- `docker_run.sh` -- runs a script inside this project's Isaac Sim 4.5.0
  Docker container, under this account's own cache/home directories.
  This is the supported way to run anything that needs Isaac Sim on this
  machine (see "Running the Isaac Sim validation" below for why).
- `find_isaac_sim.py` -- locates a *native* Isaac Sim install under this
  account's own home directory (`~/isaacsim` or `~/isaac-sim`), or at
  `$ROBOT4ART_ISAAC_SIM_PATH`. Not currently used (see below) -- kept for
  if this machine's OS is ever upgraded to Ubuntu 22.04+, where a native
  install would be viable and lighter-weight than Docker. Deliberately
  never looks under another user's home directory, even a readable one.
- `canvas.py` -- canvas geometry and the normalized-canvas-coords ->
  world-pose transform shared by any robot adapter.
- `stroke_plan.py` -- geometry primitives (`Stroke`, line/arc/bezier
  helpers) `composition.py` builds artwork plans from.
- `franka_paint_sim.py` -- the Isaac Sim standalone script that actually
  runs the simulation, using `composition.py`'s output.
- `render_paint_video.py` -- animates `output/trace.json` into an MP4
  (no Isaac Sim needed, just the trace file from a prior run). Used to
  produce `docs/media/paint_video.mp4` for the website.
- `output/` -- generated run artifacts (trace JSON, plots, video),
  gitignored.

## Running the composition preview (no simulator needed)

```bash
python3 sim/preview_compositions.py
```

## Running the Isaac Sim validation

This machine runs Ubuntu 20.04, but current Isaac Sim (pip wheel or
native standalone install) requires Ubuntu 22.04+ (glibc >= 2.34) --
that path was tried and fails outright on this OS. Docker sidesteps the
host glibc, so that's what this project uses instead, entirely under
this account (`~/docker/isaac-sim/` for caches, no other user's files
touched).

Image version matters too: the newest Isaac Sim (6.1.0) starts in a
container but its RTX renderer refuses to initialize on this machine's
driver (535.230.02 < the 550.90.07 it requires). Isaac Sim **4.5.0**
works cleanly on this driver, so `docker_run.sh` is pinned to that image.

```bash
sim/docker_run.sh franka_paint_sim.py --headless
```

First run on a given machine downloads/builds Isaac Sim's shader and
extension caches and takes several minutes; subsequent runs are fast
(under two minutes total) since `docker_run.sh` persists those caches
under `~/docker/isaac-sim/cache/`.

Set `ROBOT4ART_GPU` to pick a different GPU (default: 1) -- check
`nvidia-smi` first, since GPU 0 is often busy with other users' jobs on
this shared workstation.

Output: `sim/output/trace.json` (every simulated end-effector pose) and
`sim/output/paint_trace.png` (intended vs. executed overlay, the main
thing to eyeball after a run).

## Rendering the painting video (no simulator needed)

Reads `sim/output/trace.json` from a prior `franka_paint_sim.py` run:

```bash
python3 sim/render_paint_video.py
```

Writes `sim/output/paint_video.mp4`. To update the copies published on
the website, also copy the output into `docs/media/`:

```bash
cp sim/output/paint_video.mp4 docs/media/paint_video.mp4
ffmpeg -y -sseof -1 -i sim/output/paint_video.mp4 -frames:v 1 docs/media/paint_poster.png
```
