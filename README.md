# Robot4Art

An interactive human-robot co-creation system for live minimalist painting.

A visitor answers a few short, personal prompts — favorite color, favorite
city, a current dream or aspiration — and picks a robot platform. The
answers are composed into an original abstract artwork constrained to
roughly five brush strokes, which a robot arm then paints live in about two
minutes. Built for public demonstration at [CoRL](https://www.corl.org/)
(Conference on Robot Learning).

**[Try the composition step →](https://jack-sherman01.github.io/Robot4Art/)**
(runs entirely in your browser; see [How it works](#how-it-works) below for
what it does and doesn't cover)

Robot4Art is a collaboration between the Safe AI Lab at Carnegie Mellon
University, 破壳机器人 (Poke Robotics), and 洛可可创新设计集团 (LKK Design
Group).

## How it works

1. **Kiosk input.** A visitor answers a few short, personal prompts and
   picks a robot platform (Franka Emika Panda, Kinova Gen3, or UFACTORY
   xArm).
2. **Semantic composition.** The answers are composed into an artistic
   brief — a palette and a choice of parametric stroke "grammar" — and
   rendered as an ordered list of ≤5 vector strokes. ([`sim/composition.py`](sim/composition.py),
   also what the [GitHub Pages demo](https://jack-sherman01.github.io/Robot4Art/)
   runs client-side in JavaScript.)
3. **Stroke planning.** Each stroke is a smooth planar curve with a
   pen-up/pen-down state — the shared representation between the
   generative step and the robot execution step.
4. **Robot-agnostic execution.** The stroke plan is retargeted to the
   selected robot's task frame and physically executed.

Full technical detail — research questions, related work, evaluation plan,
timeline — lives in a bilingual (EN/ZH) research proposal that isn't
published in this repository.

## Status

| Piece | Status |
| --- | --- |
| Semantic composition (`sim/composition.py`) | Working. Deterministic grammar-based composer; the LLM call the design calls for is stubbed with a deterministic function (no model API access yet) — same answers always produce the same artwork. |
| Web demo (`docs/`) | Working. JavaScript port of the composition step, published via GitHub Pages. No backend. |
| Isaac Sim validation (`sim/franka_paint_sim.py`) | Working, via Docker (see [`sim/README.md`](sim/README.md)). Verified end-to-end with a simulated Franka arm: all 5 strokes execute and the recorded trajectory tracks the intended composition. Motion only — no ink/paint deposition model yet. |
| Kinova / xArm adapters | Not started. |
| Physical hardware | Not started. |
| Real VLM for composition | Not started (no model API access in the current dev environment). |

## Repository layout

```
sim/            Composition module + Isaac Sim simulation pipeline (Python)
docs/           Static web demo of the composition step, served via GitHub Pages
private/        Research proposal and other internal drafts (gitignored, not in this repo's history)
```

See [`sim/README.md`](sim/README.md) for how to run the composition preview
and the Isaac Sim validation.

## License

MIT — see [LICENSE](LICENSE).
