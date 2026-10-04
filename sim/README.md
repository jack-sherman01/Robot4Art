# Robot4Art simulation (Phase 0)

Validates the painting pipeline in NVIDIA Isaac Sim before any physical
robot or VLM integration: drive a robot arm through a 5-stroke artwork
plan on a virtual canvas and check that the executed end-effector
trajectory actually traces the intended composition.

## Status

Working end-to-end with a Franka arm (`sim/franka_paint_sim.py`), headless.
See `sim/output/paint_trace.png` after a run for the intended-vs-executed
overlay plot.

**This account (`heng`) has no Isaac Sim install of its own yet.** This
project only ever uses an Isaac Sim install that lives under this
account's own home directory -- it does not search, read, or run anything
under another user's home directory on this shared machine, even one that
happens to be readable. See "Running it" below for installing one here.

**Known limitation:** this validates *motion* (the end effector visits the
right places at the right simulated times) -- there is no ink/paint
deposition model yet, so nothing is actually marked on the virtual canvas.
There's also a small systematic offset between intended and executed
traces visible in the plot, likely from RMPFlow's reactive convergence
behavior; tightening `--position-tolerance` or increasing waypoint density
per stroke should reduce it further.

**Not yet done:** Kinova and xArm adapters (the proposal's robot-agnostic
execution layer), the VLM-based stroke generation step (currently a fixed
placeholder plan in `stroke_plan.py`), and real-time-budget validation
against the ~2 minute target (this run's *simulated* time was ~164s, which
is not directly comparable to real-world wall-clock execution speed).

## Files

- `find_isaac_sim.py` -- locates a usable Isaac Sim install under *this*
  account's own home directory (`~/isaacsim` or `~/isaac-sim`), or at
  `$ROBOT4ART_ISAAC_SIM_PATH` if set. Deliberately never looks under
  another user's home directory, even a readable one.
- `canvas.py` -- canvas geometry and the normalized-canvas-coords ->
  world-pose transform shared by any robot adapter.
- `stroke_plan.py` -- the 5-stroke artwork plan. Currently a fixed
  placeholder standing in for the future VLM-generated composition
  described in the research proposal.
- `franka_paint_sim.py` -- the Isaac Sim standalone script that actually
  runs the simulation.
- `output/` -- generated run artifacts (trace JSON + plot PNG), gitignored.

## Running it

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
