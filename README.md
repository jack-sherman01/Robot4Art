# Robot4Art

An interactive human-robot co-creation system for live minimalist painting.

A visitor answers a few short, personal prompts — favorite color, favorite
city, a current dream or aspiration — and picks a robot platform. A real
Claude model directly composes the piece: every stroke's own shape, pen
color, and weight, up to 20 strokes drawn with a small set of common pen
colors (not custom-mixed paint). A robot arm then paints it live in a
couple of minutes, alongside the model's own unedited explanation of why
the piece looks the way it does. Built for public demonstration at
[CoRL](https://www.corl.org/) (Conference on Robot Learning).

**[Try it live →](https://jack-sherman01.github.io/Robot4Art/)** — generate
a painting in your browser (a real model call via a small backend — see
[`worker/`](worker/)), then see the same composition painted by a
simulated Franka arm in NVIDIA Isaac Sim, side by side: what the model
composed vs. the real recorded video of the robot's full-body motion
painting it. See [How it works](#how-it-works) below for what the demo
does and doesn't cover.

Robot4Art is a collaboration between the Safe AI Lab at Carnegie Mellon
University, 破壳机器人 (Poke Robotics), and 洛可可创新设计集团 (LKK Design
Group).

## How it works

1. **Kiosk input.** A visitor answers a few short, personal prompts and
   picks a robot platform (Franka Emika Panda, Kinova Gen3, or UFACTORY
   xArm).
2. **Semantic composition.** A real Claude model directly authors the
   piece — every stroke's own 2-6 control points, pen color, and weight,
   up to 20 strokes from a small set of common pen colors, plus one
   brush/tool it also chooses — not picking from a template. The
   [GitHub Pages demo](https://jack-sherman01.github.io/Robot4Art/) calls
   a small Cloudflare Worker backend that holds the model API key
   server-side ([`worker/`](worker/)); if that's unreachable, a fast
   deterministic local preview ([`sim/composition.py`](sim/composition.py),
   ported to client-side JavaScript) takes over instead, clearly labeled
   as such. Either way the model's (or the preview's) own explanation is
   revealed on screen as the piece is painted.
3. **Stroke planning.** The model's handful of control points per stroke
   are smoothed into a dense curve (Catmull-Rom spline) with a
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
| Deterministic composition (`sim/composition.py`) | Working. Picks between a handful of hand-written stroke grammars — the local-preview fallback for the web demo, and still useful as a fast, free, simulator-free check of the geometry/rendering pipeline. |
| Real LLM composition (`sim/llm_composer.py`, `worker/`) | Working, two ways to call it. Batch: the headless `claude` CLI (no separate API key needed), used to generate the cached Isaac Sim example painting. Live: a Cloudflare Worker (`worker/`) that holds a real Anthropic API key server-side and powers the public web demo's "Paint it" button — rate-limited per-IP and with a global daily cap since it spends real money per request. Both ask the model to directly author every stroke's shape, color, and weight, not pick from a template. |
| Web demo (`docs/`) | Working. Calls the live Worker for a real composition; falls back to the deterministic local preview (clearly labeled) if that's unreachable. Also shows a real recorded Isaac Sim video of a Franka arm painting a real-Claude-authored piece, side by side with a static render of what the model painted. Published via GitHub Pages; the Worker is the only non-static piece. |
| Isaac Sim validation + video (`sim/franka_paint_sim.py`, `sim/franka_paint_video.py`) | Working, via Docker (see [`sim/README.md`](sim/README.md)). Verified end-to-end with a simulated Franka arm: all strokes execute and the recorded trajectory tracks the intended composition. The video now also renders painted stroke marks on the canvas (one mesh per stroke) instead of staying blank — though the live viewport capture has an intermittent freeze bug under investigation that can leave later strokes under-rendered in the recorded video. |
| Kinova / xArm adapters | Not started. |
| Physical hardware | Not started. |

## Repository layout

```
sim/            Composition module + Isaac Sim simulation pipeline (Python)
docs/           Static web demo, served via GitHub Pages
worker/         Cloudflare Worker backend for the live web demo's real model calls
private/        Research proposal and other internal drafts (gitignored, not in this repo's history)
```

See [`sim/README.md`](sim/README.md) for how to run the composition preview
and the Isaac Sim validation.

## License

MIT — see [LICENSE](LICENSE).
