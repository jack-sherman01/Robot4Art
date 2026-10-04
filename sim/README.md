# Robot4Art simulation + composition pipeline

Three pieces, independently testable:

1. **Semantic composition** (`composition.py`) -- turns a visitor's kiosk
   answers into a <=10-stroke artwork plan, colored with a small set of
   common pen colors (not custom-mixed paint). Pure Python/numpy/matplotlib,
   no simulator or GPU needed.
2. **Isaac Sim validation** (`franka_paint_sim.py`) -- drives a robot arm
   through a stroke plan on a virtual canvas and checks the executed
   end-effector trajectory actually traces the intended composition.
   Needs Isaac Sim (see below).
3. **Isaac Sim video** (`franka_paint_video.py`) -- the same plan, but
   captures real rendered frames of the robot's full-body motion for the
   website. Also needs Isaac Sim.

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

**Real Isaac Sim video (`franka_paint_video.py`): working.** Earlier
attempts at an actual rendered camera view (a custom Camera sensor bound
to the active viewport's render product, explicit scene lighting,
forcing the Replicator orchestrator to step) all returned an identical,
scene-independent placeholder frame regardless of settings -- documented
in git history as a dead end. The fix was dropping the custom Camera
sensor entirely and using Kit's own viewport-capture utility
(`omni.kit.viewport.utility.capture_viewport_to_file`) against the
viewport the main render loop already drives, auto-framed with
`frame_viewport_prims` -- a different, simpler, standard code path, and
it worked on the first real attempt. `franka_paint_video.py` runs the
same 10-stroke plan as `franka_paint_sim.py` and periodically captures
real rendered frames of the robot's full-body motion (most physics steps
still run headless/fast; only every `--capture-every`-th step pays the
rendering + capture cost -- 363 frames over 16350 steps took ~90s wall
time). `render_target_painting.py` renders a static image of the same
composition for an apples-to-apples "what it was told to paint" vs.
"what it painted" comparison on the website.

The matplotlib/ffmpeg trace-animation approach (`render_paint_video.py`)
that stood in before this was fixed is kept as a simulator-free fallback
-- still useful for quickly visualizing a trace without waiting on
Isaac Sim/Docker.

**Known quirk:** `capture_viewport_to_file` writes frames to the bind
mount as root, unlike every other output in this pipeline (which comes
out owned by `heng`) -- something about Kit's internal renderer-capture
plugin doesn't go through the same path as a normal Python `open()`.
Clean up with a throwaway root container rather than `rm` directly:
`docker run --rm -v $(pwd)/sim:/workspace/sim alpine rm -rf /workspace/sim/output/frames`.

**Not yet done:** Kinova and xArm adapters (the proposal's robot-agnostic
execution layer), a real model call in `derive_brief`, and
real-time-budget validation against the live-kiosk timing target (this
run's *simulated* time was ~272s for 10 strokes, which is not directly
comparable to real wall-clock execution speed).

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
- `example_answers.py` -- the one fixed example `PromptAnswers` used
  everywhere a representative composition is needed (Isaac Sim
  validation, the robot video, the static reference image), so all three
  stay in sync.
- `franka_paint_sim.py` -- the Isaac Sim standalone script that runs the
  simulation and records the trace/trace-plot, using `composition.py`'s
  output. Headless/fast -- no rendering.
- `franka_paint_video.py` -- same plan, but periodically captures real
  rendered viewport frames of the robot's full-body motion. Slower
  (renders every `--capture-every`-th step); produces PNG frames for
  `ffmpeg` to assemble into `docs/media/isaac_robot_video.mp4`.
- `render_target_painting.py` -- renders a static PNG of
  `example_answers.EXAMPLE_ANSWERS`'s composition, for
  `docs/media/target_painting.png` (the "what it was told to paint" side
  of the website comparison).
- `render_paint_video.py` -- animates `output/trace.json` into an MP4
  with matplotlib (no Isaac Sim needed, just a trace file from a prior
  `franka_paint_sim.py` run). A simulator-free fallback, not currently
  published on the website.
- `output/` -- generated run artifacts (trace JSON, plots, frames,
  videos), gitignored.

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

## Recording the real Isaac Sim robot video

```bash
rm -rf sim/output/frames
sim/docker_run.sh franka_paint_video.py --headless
```

Captures frames to `sim/output/frames/` (owned by root -- see "Known
quirk" above), then encode on the host (ffmpeg isn't in the container):

```bash
ffmpeg -y -framerate 15 -i sim/output/frames/frame_%05d.png \
  -pix_fmt yuv420p -c:v libx264 -crf 20 sim/output/isaac_render_video.mp4
docker run --rm -v "$(pwd)/sim:/workspace/sim" alpine rm -rf /workspace/sim/output/frames
```

Then render the matching static reference image and update the website's copies:

```bash
python3 sim/render_target_painting.py
cp sim/output/isaac_render_video.mp4 docs/media/isaac_robot_video.mp4
cp sim/output/target_painting.png docs/media/target_painting.png
ffmpeg -y -sseof -1 -i sim/output/isaac_render_video.mp4 -frames:v 1 docs/media/isaac_robot_poster.png
```

`--capture-every` (default 45 physics steps) trades frame count for
wall-clock time -- lower it for a smoother video at the cost of a slower
capture run.

## Rendering the fallback trace-animation video (no simulator needed)

Reads `sim/output/trace.json` from a prior `franka_paint_sim.py` run;
not currently published on the website (superseded by the real Isaac Sim
video above), but still useful as a quick, simulator-free check:

```bash
python3 sim/render_paint_video.py
```

Writes `sim/output/paint_video.mp4`.
