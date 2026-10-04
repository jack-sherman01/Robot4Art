/**
 * Robot4Art kiosk demo.
 *
 * A JavaScript port of sim/composition.py's grammar-based composer, so the
 * same deterministic "visitor answers -> <=10-stroke artwork" step can run
 * entirely client-side on GitHub Pages. Mirrors the Python module's
 * structure (common pen colors, text-seeded RNG, three stroke grammars,
 * tapered brush rendering) but uses its own small seeded PRNG rather than
 * reproducing numpy's PCG64 bit for bit -- the point is a consistent,
 * explainable demo, not byte-for-byte parity with the Python backend.
 */

(() => {
  "use strict";

  const MAX_STROKES = 10;

  // ---- common pen colors (mirrors composition.py's STANDARD_PENS) ---------
  // A realistic, physically-stockable set of pen/marker colors -- the
  // robot holds a small set of interchangeable pens, not custom-mixed
  // paint, so every composition's colors come from this fixed list.

  const NAMED_HUES = [
    ["red", 0.00], ["orange", 0.08], ["amber", 0.11], ["yellow", 0.15], ["gold", 0.13],
    ["lime", 0.22], ["green", 0.33], ["olive", 0.19], ["teal", 0.50], ["cyan", 0.52],
    ["sky", 0.56], ["blue", 0.60], ["navy", 0.62], ["indigo", 0.68], ["purple", 0.75],
    ["violet", 0.78], ["lavender", 0.72], ["magenta", 0.83], ["pink", 0.90],
    ["rose", 0.95], ["brown", 0.07], ["black", 0.60], ["white", 0.60], ["gray", 0.60], ["grey", 0.60],
  ];

  const STANDARD_PENS = {
    black: "#232323",
    red: "#C0392B",
    orange: "#D2691E",
    yellow: "#D4A017",
    green: "#2E7D4F",
    teal: "#1F7A72",
    blue: "#2255A4",
    purple: "#6B3FA0",
    pink: "#C0527A",
    brown: "#6F4E2E",
  };

  const PEN_HUES = {
    red: 0.00, orange: 0.07, yellow: 0.14, green: 0.36, teal: 0.49,
    blue: 0.61, purple: 0.76, pink: 0.92, brown: 0.08,
  };

  // Visitor-facing style choices, each tied to one grammar + rendering
  // treatment (mirrors composition.py's STYLES). Letting the visitor pick
  // the style directly is both more satisfying to interact with and more
  // reliable for aesthetic quality than a hash-derived 1-in-3 assignment.
  const STYLES = {
    modernist: { label: "Modernist Gesture", grammar: "arc_over_line" },
    impressionist: { label: "Impressionist Bloom", grammar: "radiating_strokes" },
    ink_wash: { label: "Ink Wash Minimal", grammar: "nested_curves" },
  };
  const DEFAULT_STYLE = "modernist";

  // A short, templated explanation of *why* the piece looks the way it
  // does, connecting the artwork back to the visitor's own answers --
  // stands in for an LLM-written rationale, same reason deriveBrief is
  // deterministic rather than a real model call.
  const RATIONALE_TEMPLATES = {
    arc_over_line:
      'A single {primary} gesture rises across the canvas and closes with a {secondary} ' +
      'arc — a confident line for a dream like “{dream}.” The fine black marks scattered ' +
      "near it carry {moodArticle} {mood} energy, and the two pens were picked to echo " +
      "“{color}” and the feel of {city}.",
    nested_curves:
      "Layers of {primary} and {secondary} curves nest inside one another, each a little " +
      'larger than the last — like ripples spreading outward from “{dream}.” A small ' +
      "flourish signs off the outermost curve, and the fine black accents nearby are " +
      "{moodArticle} {mood} touch, in colors drawn from “{color}” and {city}.",
    radiating_strokes:
      "Strokes in {primary} and {secondary} radiate outward from a single point, like " +
      'petals opening — a burst of energy for “{dream}.” The small black marks at their ' +
      "tips add {moodArticle} {mood} rhythm, and the palette traces back to “{color}” " +
      "and a touch of {city}.",
  };

  function article(word) {
    return "aeiou".includes((word[0] || "").toLowerCase()) ? "an" : "a";
  }

  function rationaleFor(answers, brief) {
    const template = RATIONALE_TEMPLATES[brief.grammar];
    const mood = (answers.mood || "").trim() || "calm";
    return template
      .replaceAll("{primary}", brief.penNames[0])
      .replaceAll("{secondary}", brief.penNames[1])
      .replaceAll("{dream}", answers.dream.trim())
      .replaceAll("{mood}", mood)
      .replaceAll("{moodArticle}", article(mood))
      .replaceAll("{color}", answers.color.trim())
      .replaceAll("{city}", answers.city.trim());
  }

  function hashStr(s) {
    let h = 0x811c9dc5;
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
    return h >>> 0;
  }

  function textSeed(...parts) {
    const joined = parts.map((p) => p.trim().toLowerCase()).join("␟");
    return hashStr(joined);
  }

  function hueFromText(text) {
    const t = text.trim().toLowerCase();
    for (const [name, hue] of NAMED_HUES) {
      if (t.includes(name)) return hue;
    }
    return (hashStr(t) % 360) / 360;
  }

  function hueDistance(a, b) {
    const d = Math.abs(a - b) % 1;
    return Math.min(d, 1 - d);
  }

  function pickPens(hue) {
    const names = Object.keys(PEN_HUES);
    let primary = names[0];
    let bestDist = Infinity;
    for (const n of names) {
      const d = hueDistance(PEN_HUES[n], hue);
      if (d < bestDist) { bestDist = d; primary = n; }
    }
    const remaining = names.filter((n) => n !== primary);
    const target = (PEN_HUES[primary] + 1 / 3) % 1;
    let secondary = remaining[0];
    bestDist = Infinity;
    for (const n of remaining) {
      const d = hueDistance(PEN_HUES[n], target);
      if (d < bestDist) { bestDist = d; secondary = n; }
    }
    return [primary, secondary];
  }

  // ---- seeded RNG ----------------------------------------------------------

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0;
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function uniform(rng, lo, hi) {
    return lo + rng() * (hi - lo);
  }
  function uniformInt(rng, lo, hiInclusive) {
    return Math.floor(uniform(rng, lo, hiInclusive + 1));
  }

  // ---- geometry primitives (mirrors stroke_plan.py) -------------------------

  function line(p0, p1, n = 10) {
    const pts = [];
    for (let i = 0; i < n; i++) {
      const t = i / (n - 1);
      pts.push([p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t]);
    }
    return pts;
  }

  function arc(center, radius, startDeg, endDeg, n = 16) {
    const pts = [];
    for (let i = 0; i < n; i++) {
      const t = startDeg + ((endDeg - startDeg) * i) / (n - 1);
      const rad = (t * Math.PI) / 180;
      pts.push([center[0] + radius * Math.cos(rad), center[1] + radius * Math.sin(rad)]);
    }
    return pts;
  }

  function quadraticBezier(p0, p1, p2, n = 16) {
    const pts = [];
    for (let i = 0; i < n; i++) {
      const t = i / (n - 1);
      const mt = 1 - t;
      pts.push([
        mt * mt * p0[0] + 2 * mt * t * p1[0] + t * t * p2[0],
        mt * mt * p0[1] + 2 * mt * t * p1[1] + t * t * p2[1],
      ]);
    }
    return pts;
  }

  function clip01(points, margin = 0.08) {
    return points.map(([x, y]) => [
      Math.min(1 - margin, Math.max(margin, x)),
      Math.min(1 - margin, Math.max(margin, y)),
    ]);
  }

  function rotatePoints(points, center, deg) {
    const rad = (deg * Math.PI) / 180;
    const c = Math.cos(rad), s = Math.sin(rad);
    return points.map(([x, y]) => {
      const dx = x - center[0], dy = y - center[1];
      return [center[0] + dx * c - dy * s, center[1] + dx * s + dy * c];
    });
  }

  function normalize(v) {
    const n = Math.hypot(v[0], v[1]) || 1;
    return [v[0] / n, v[1] / n];
  }

  // ---- brief derivation (mirrors derive_brief) -------------------------------

  function deriveBrief(answers) {
    const seed = textSeed(answers.color, answers.city, answers.dream, answers.mood);
    const rng = mulberry32(seed);

    const hue = hueFromText(answers.color);
    const [primaryPen, secondaryPen] = pickPens(hue);
    const palette = [STANDARD_PENS[primaryPen], STANDARD_PENS[secondaryPen], STANDARD_PENS.black];

    const style = STYLES[answers.style] ? answers.style : DEFAULT_STYLE;
    const grammar = STYLES[style].grammar;
    const scale = uniform(rng, 0.9, 1.15);
    const rotationDeg = uniform(rng, -15, 15);
    const center = [0.5 + uniform(rng, -0.09, 0.09), 0.5 + uniform(rng, -0.07, 0.09)];

    return { seed, grammar, style, penNames: [primaryPen, secondaryPen], palette, scale, rotationDeg, center };
  }

  // ---- accent dabs (mirrors _accent_dabs) ------------------------------------

  function accentDabs(rng, anchors, color, n, scale, namePrefix = "accent") {
    const strokes = [];
    for (let i = 0; i < n; i++) {
      const anchor = anchors[uniformInt(rng, 0, anchors.length - 1)];
      const jitter = [uniform(rng, -0.02, 0.02) * scale, uniform(rng, -0.02, 0.02) * scale];
      const origin = [anchor[0] + jitter[0], anchor[1] + jitter[1]];
      const angleDeg = uniform(rng, 0, 360);
      const radius = scale * uniform(rng, 0.018, 0.032);
      const sweep = uniform(rng, 55, 95) * (rng() > 0.5 ? 1 : -1);
      strokes.push({
        name: `${namePrefix}_${i}`,
        color,
        width: uniform(rng, 0.2, 0.3),
        points: clip01(arc(origin, radius, angleDeg, angleDeg + sweep, 6)),
      });
    }
    return strokes;
  }

  // ---- grammars (mirror composition.py's _grammar_* functions) --------------

  function grammarArcOverLine(rng, palette, scale, c) {
    const half = 0.34 * scale;
    const p0 = [c[0] - half, c[1] - half * uniform(rng, 0.75, 1.0)];
    const p1 = [c[0] + half * uniform(rng, 0.9, 1.05), c[1] + half * uniform(rng, 0.85, 1.05)];
    const mid = [
      (p0[0] + p1[0]) / 2 + uniform(rng, -0.03, 0.03) * scale,
      (p0[1] + p1[1]) / 2 + uniform(rng, 0.02, 0.07) * scale,
    ];
    const arcCenter = [p1[0] - 0.01, p1[1] - 0.13 * scale];

    const strokes = [
      { name: "rising_gesture", color: palette[0], width: 1.0, points: clip01(quadraticBezier(p0, mid, p1, 18)) },
      { name: "closing_arc", color: palette[1], width: 0.72, points: clip01(arc(arcCenter, 0.16 * scale, -30, 200, 20)) },
    ];

    const offset = [uniform(rng, -0.02, 0.02) * scale, uniform(rng, 0.05, 0.09) * scale];
    const echoP0 = [p0[0] + offset[0], p0[1] + offset[1]];
    const echoP1 = [p1[0] + offset[0] * 0.6, p1[1] + offset[1] * 0.6];
    const echoMid = [(echoP0[0] + echoP1[0]) / 2, (echoP0[1] + echoP1[1]) / 2 + uniform(rng, 0.02, 0.05) * scale];
    strokes.push({ name: "echo_gesture", color: palette[0], width: 0.45, points: clip01(quadraticBezier(echoP0, echoMid, echoP1, 16)) });

    const accentOrigin = [c[0] + uniform(rng, -0.28, -0.12) * scale, c[1] + uniform(rng, -0.14, -0.02) * scale];
    const accentEnd = [accentOrigin[0] + uniform(rng, 0.12, 0.2) * scale, accentOrigin[1] + uniform(rng, -0.03, 0.03) * scale];
    strokes.push({ name: "horizon_accent", color: palette[2], width: 0.4, points: clip01(line(accentOrigin, accentEnd, 8)) });

    const anchors = [p0, p1, mid, arcCenter, echoP0, echoP1];
    strokes.push(...accentDabs(rng, anchors, palette[2], MAX_STROKES - strokes.length, scale));
    return strokes;
  }

  function grammarNestedCurves(rng, palette, scale, c) {
    const strokes = [];
    const colors = [palette[0], palette[1], palette[0], palette[1]];
    const widths = [1.0, 0.8, 0.62, 0.45];
    const anchors = [];
    let outerP2 = null;
    const nCurves = 4;
    for (let i = 0; i < nCurves; i++) {
      const r = (0.13 + 0.075 * i) * scale;
      const lean = uniform(rng, -0.15, 0.15);
      const p0 = [c[0] - r, c[1] + r * (0.25 + lean)];
      const p1 = [c[0] + lean * r * 0.4, c[1] - r * uniform(rng, 0.55, 0.85)];
      const p2 = [c[0] + r, c[1] + r * (0.3 - lean)];
      if (i === nCurves - 1) outerP2 = p2;
      anchors.push(p0, p2);
      strokes.push({ name: `nested_curve_${i}`, color: colors[i], width: widths[i], points: clip01(quadraticBezier(p0, p1, p2, 16)) });
    }

    const flourishDir = normalize([uniform(rng, 0.6, 1.0), uniform(rng, -0.1, 0.25)]);
    const flourishLen = 0.09 * scale;
    const dashOrigin = [outerP2[0] - flourishDir[0] * flourishLen * 0.2, outerP2[1] - flourishDir[1] * flourishLen * 0.2];
    const dashEnd = [outerP2[0] + flourishDir[0] * flourishLen, outerP2[1] + flourishDir[1] * flourishLen];
    strokes.push({ name: "flourish", color: palette[2], width: 0.42, points: clip01(line(dashOrigin, dashEnd, 8)) });

    strokes.push(...accentDabs(rng, anchors, palette[2], MAX_STROKES - strokes.length, scale));
    return strokes;
  }

  function grammarRadiatingStrokes(rng, palette, scale, c) {
    const strokes = [];
    const anchors = [];
    const nRays = 6;
    const colors = [];
    for (let i = 0; i < nRays; i++) colors.push(palette[i % 2]);
    const baseAngle = uniform(rng, 0, 360);
    const spread = 360 / nRays;
    for (let i = 0; i < nRays; i++) {
      const angle = ((baseAngle + i * spread + uniform(rng, -spread * 0.22, spread * 0.22)) * Math.PI) / 180;
      const length = (0.13 + 0.1 * uniform(rng, 0.4, 1.0)) * scale;
      const inner = (0.03 + 0.02 * uniform(rng, 0, 1)) * scale;
      const p0 = [c[0] + inner * Math.cos(angle), c[1] + inner * Math.sin(angle)];
      const p1 = [c[0] + length * Math.cos(angle), c[1] + length * Math.sin(angle)];
      const width = i % 2 === 0 ? 0.85 : 0.55;
      anchors.push(p1);
      strokes.push({ name: `ray_${i}`, color: colors[i], width, points: clip01(line(p0, p1, 10)) });
    }
    strokes.push({
      name: "center_arc",
      color: palette[2],
      width: 0.45,
      points: clip01(arc(c, 0.065 * scale, uniform(rng, 0, 60), uniform(rng, 220, 300), 14)),
    });
    strokes.push(...accentDabs(rng, anchors, palette[2], MAX_STROKES - strokes.length, scale));
    return strokes;
  }

  const GRAMMAR_FNS = {
    arc_over_line: grammarArcOverLine,
    nested_curves: grammarNestedCurves,
    radiating_strokes: grammarRadiatingStrokes,
  };

  function compose(answers) {
    const brief = deriveBrief(answers);
    const rng = mulberry32(brief.seed);
    let strokes = GRAMMAR_FNS[brief.grammar](rng, brief.palette, brief.scale, brief.center);
    if (Math.abs(brief.rotationDeg) > 1e-6) {
      strokes = strokes.map((s) => ({ ...s, points: clip01(rotatePoints(s.points, brief.center, brief.rotationDeg)) }));
    }
    return { strokes, brief };
  }

  // ---- painterly rendering ----------------------------------------------

  const SVG_NS = "http://www.w3.org/2000/svg";
  const canvasEl = document.getElementById("canvas");
  const placeholderEl = document.getElementById("canvas-placeholder");
  const briefEl = document.getElementById("brief");
  const briefGrammarEl = document.getElementById("brief-grammar");
  const briefStrokesEl = document.getElementById("brief-strokes");
  const briefRobotEl = document.getElementById("brief-robot");
  const briefPaletteEl = document.getElementById("brief-palette");

  const ROBOT_LABELS = {
    franka: "Franka Emika Panda",
    kinova: "Kinova Gen3",
    xarm: "UFACTORY xArm",
  };

  /** Build a tapered brush-stroke outline from a centerline (thin at both
   * ends, full width in the middle) -- mirrors preview_compositions.py's
   * tapered_polygon. Points are in the 0-100 SVG viewBox space already. */
  function taperedPolygonPath(points, maxWidth) {
    const n = points.length;
    if (n < 2) return "";
    const dirs = points.map((p, i) => {
      const a = points[Math.max(0, i - 1)];
      const b = points[Math.min(n - 1, i + 1)];
      return normalize([b[0] - a[0], b[1] - a[1]]);
    });
    const left = [];
    const right = [];
    for (let i = 0; i < n; i++) {
      const t = i / (n - 1);
      const taper = Math.pow(Math.sin(Math.PI * t), 0.7);
      const w = (maxWidth * (0.22 + 0.78 * taper)) / 2;
      const [dx, dy] = dirs[i];
      const perp = [-dy, dx];
      left.push([points[i][0] + perp[0] * w, points[i][1] + perp[1] * w]);
      right.push([points[i][0] - perp[0] * w, points[i][1] - perp[1] * w]);
    }
    const poly = left.concat(right.reverse());
    return poly.map(([x, y], i) => `${i === 0 ? "M" : "L"} ${x.toFixed(2)},${y.toFixed(2)}`).join(" ") + " Z";
  }

  function toScreenPoints(points) {
    // normalized [0,1]^2 -> 100x100 viewBox, y flipped so "up" feels up.
    return points.map(([x, y]) => [x * 100, (1 - y) * 100]);
  }

  /** Build the SVG element(s) for one stroke, in the rendering treatment
   * for the given style -- a real visual difference per style, not just
   * different geometry. Mirrors preview_compositions.py's render_stroke. */
  function buildStrokeElement(stroke, style, rng) {
    const screenPoints = toScreenPoints(stroke.points);

    if (style === "impressionist") {
      // A chain of overlapping dabs instead of one continuous shape --
      // impasto texture, closer to how Impressionist brushwork reads up
      // close, and visibly different from the other two styles.
      const g = document.createElementNS(SVG_NS, "g");
      const n = screenPoints.length;
      const step = Math.max(1, Math.floor(n / 7));
      const baseR = 2.8 * stroke.width;
      for (let i = 0; i < n; i += step) {
        const [x, y] = screenPoints[i];
        const r = baseR * uniform(rng, 0.75, 1.15);
        const dab = document.createElementNS(SVG_NS, "circle");
        dab.setAttribute("cx", x.toFixed(2));
        dab.setAttribute("cy", y.toFixed(2));
        dab.setAttribute("r", r.toFixed(2));
        dab.setAttribute("fill", stroke.color);
        dab.setAttribute("fill-opacity", "0.88");
        g.appendChild(dab);
      }
      return g;
    }

    const maxWidth = style === "modernist" ? 5.0 * stroke.width : 3.2 * stroke.width;
    const fillOpacity = style === "modernist" ? "0.96" : "0.82";
    const d = taperedPolygonPath(screenPoints, maxWidth);
    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("d", d);
    path.setAttribute("fill", stroke.color);
    path.setAttribute("fill-opacity", fillOpacity);
    path.setAttribute("stroke", "none");
    return path;
  }

  function renderPainting(strokes, brief, robotKey, answers) {
    canvasEl.innerHTML = "";
    placeholderEl.hidden = true;
    const rng = mulberry32(brief.seed ^ 0x9e3779b9);

    const strokeDuration = 230;
    strokes.forEach((stroke, i) => {
      const el = buildStrokeElement(stroke, brief.style, rng);
      el.style.opacity = "0";
      el.style.transformOrigin = "50% 50%";
      canvasEl.appendChild(el);
      el.animate(
        [
          { opacity: 0, transform: "scale(0.97)" },
          { opacity: 1, transform: "scale(1)" },
        ],
        { duration: 420, delay: i * strokeDuration, easing: "ease-out", fill: "forwards" }
      );
    });

    briefEl.hidden = false;
    briefGrammarEl.textContent = (STYLES[brief.style] || {}).label || brief.style;
    briefStrokesEl.textContent = `${strokes.length} of ${MAX_STROKES}`;
    briefRobotEl.textContent = ROBOT_LABELS[robotKey] || robotKey;
    briefPaletteEl.innerHTML = "";
    brief.palette.forEach((hex) => {
      const sw = document.createElement("span");
      sw.className = "swatch";
      sw.style.background = hex;
      sw.title = hex;
      briefPaletteEl.appendChild(sw);
    });

    revealRationale(rationaleFor(answers, brief), strokes.length * strokeDuration);
  }

  // ---- rationale reveal ("why this painting") ----------------------------

  const rationaleEl = document.getElementById("rationale");
  const rationalePlaceholderEl = document.getElementById("rationale-placeholder");

  /** Reveal the rationale sentence by sentence, timed to roughly track
   * the stroke animation -- "shown on screen as it's created," not just
   * dumped in at the end. Old spans are discarded via innerHTML = "" at
   * the start of each call, so a resubmission simply orphans any
   * still-animating spans from the previous run. */
  function revealRationale(text, totalStrokeMs) {
    rationalePlaceholderEl.hidden = true;
    rationaleEl.innerHTML = "";
    rationaleEl.hidden = false;

    const sentences = text.match(/[^.]+\.\s*/g) || [text];
    const perSentenceDelay = Math.max(350, totalStrokeMs / sentences.length);
    sentences.forEach((sentence, i) => {
      const span = document.createElement("span");
      span.textContent = sentence;
      span.style.opacity = "0";
      rationaleEl.appendChild(span);
      span.animate([{ opacity: 0 }, { opacity: 1 }], {
        duration: 500,
        delay: i * perSentenceDelay,
        easing: "ease-out",
        fill: "forwards",
      });
    });
  }

  // ---- wiring ------------------------------------------------------------

  const EXAMPLES = [
    { color: "teal", city: "Austin", dream: "to make music", mood: "playful", style: "impressionist" },
    { color: "sunset orange", city: "Paris", dream: "to open a bakery", mood: "cozy", style: "modernist" },
    { color: "deep purple", city: "Shanghai", dream: "to become a scientist", mood: "determined", style: "ink_wash" },
    { color: "forest green", city: "Tokyo", dream: "to write a novel", mood: "calm", style: "ink_wash" },
    { color: "blue", city: "Pittsburgh", dream: "to build robots that help people", mood: "curious", style: "modernist" },
  ];

  const form = document.getElementById("kiosk-form");

  function runFromForm() {
    const styleInput = form.querySelector('input[name="style"]:checked');
    const answers = {
      color: form.color.value || "blue",
      city: form.city.value || "a city",
      dream: form.dream.value || "a dream",
      mood: form.mood.value || "",
      style: styleInput ? styleInput.value : DEFAULT_STYLE,
    };
    const { strokes, brief } = compose(answers);
    renderPainting(strokes, brief, form.robot.value, answers);
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    runFromForm();
  });

  document.getElementById("surprise-me").addEventListener("click", () => {
    const ex = EXAMPLES[Math.floor(Math.random() * EXAMPLES.length)];
    form.color.value = ex.color;
    form.city.value = ex.city;
    form.dream.value = ex.dream;
    form.mood.value = ex.mood;
    const styleRadio = form.querySelector(`input[name="style"][value="${ex.style}"]`);
    if (styleRadio) styleRadio.checked = true;
    runFromForm();
  });
})();
