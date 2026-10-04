# Robot4Art simulation + composition pipeline

Two pieces, independently testable:

1. **Semantic composition** (`composition.py`) -- turns a visitor's kiosk
   answers into a <=5-stroke artwork plan. Pure Python/numpy/matplotlib,
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
the visitor's answers. The "which grammar, what parameters" choice the
proposal assigns to an LLM is currently a deterministic function of the
answers (`derive_brief`) rather than an actual model call -- no model API
key is available in this environment yet. Swapping in a real model only
requires changing `derive_brief`; everything downstream (grammar
rendering, Stroke output) is unaffected. Run `python3 sim/preview_compositions.py`
to render example outputs to `sim/output/compositions/examples.png`
without needing Isaac Sim at all. The other path in the proposal (direct
CLIPDraw-style stroke optimization against a vision-language embedding)
is not implemented.

**Isaac Sim (`franka_paint_sim.py`): previously validated, currently
blocked.** It worked end-to-end with a Franka arm (see git history for
`sim/output/paint_trace.png`), but that run used another user's Isaac Sim
install on this shared machine, which this project no longer does (see
"Running it" below) -- **this account (`heng`) has no Isaac Sim install of
its own yet**, so this part can't run again until one exists here. The
script itself still uses the old fixed plan from `stroke_plan.py`;
wiring it up to `composition.py`'s output is a small follow-up once Isaac
Sim is available.

**Known limitation (from the earlier validated run):** this only
validates *motion* (the end effector visits the right places at the right
simulated times) -- there is no ink/paint deposition model yet, so
nothing is actually marked on the virtual canvas. There was also a small
systematic offset between intended and executed traces, likely from
RMPFlow's reactive convergence behavior.

**Not yet done:** Kinova and xArm adapters (the proposal's robot-agnostic
execution layer), a real model call in `derive_brief`, wiring
`composition.py`'s output into `franka_paint_sim.py`, and real-time-budget
validation against the ~2 minute target.

## Files

- `composition.py` -- visitor answers -> artistic brief -> <=5-stroke
  plan. No simulator needed; see `preview_compositions.py`.
- `preview_compositions.py` -- renders example compositions to
  `output/compositions/examples.png`. No simulator needed.
- `find_isaac_sim.py` -- locates a usable Isaac Sim install under *this*
  account's own home directory (`~/isaacsim` or `~/isaac-sim`), or at
  `$ROBOT4ART_ISAAC_SIM_PATH` if set. Deliberately never looks under
  another user's home directory, even a readable one.
- `canvas.py` -- canvas geometry and the normalized-canvas-coords ->
  world-pose transform shared by any robot adapter.
- `stroke_plan.py` -- geometry primitives (`Stroke`, line/arc/bezier
  helpers) plus a fixed example 5-stroke plan, used directly by
  `franka_paint_sim.py` and as a building block for `composition.py`.
- `franka_paint_sim.py` -- the Isaac Sim standalone script that actually
  runs the simulation.
- `output/` -- generated run artifacts (trace JSON, plots), gitignored.

## Running the composition preview (no simulator needed)

```bash
python3 sim/preview_compositions.py
```

## Running the Isaac Sim validation

Requires an Isaac Sim standalone install under this account, at `~/isaacsim`
or `~/isaac-sim` (or pointed to via `$ROBOT4ART_ISAAC_SIM_PATH`). Isaac Sim
is a multi-GB download gated behind an NVIDIA account; install it under
`heng`'s own home directory per [NVIDIA's workstation install
instructions](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html)
before running this.

```bash
ISAAC=$(python3 sim/find_isaac_sim.py)
CUDA_VISIBLE_DEVICES=1 "$ISAAC/python.sh" sim/franka_paint_sim.py --headless
```

`CUDA_VISIBLE_DEVICES` is worth setting explicitly on a shared GPU
workstation -- check `nvidia-smi` first and pick an idle GPU. Drop
`--headless` (pass `--show` instead) to open the Isaac Sim GUI window if
running with a display attached.

Output: `sim/output/trace.json` (every simulated end-effector pose) and
`sim/output/paint_trace.png` (intended vs. executed overlay, the main
thing to eyeball after a run).
