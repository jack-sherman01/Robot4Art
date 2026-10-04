/**
 * Robot4Art kiosk demo.
 *
 * A JavaScript port of sim/composition.py's grammar-based composer, so the
 * same deterministic "visitor answers -> <=5-stroke artwork" step can run
 * entirely client-side on GitHub Pages. Mirrors the Python module's
 * structure (named-hue lookup, text-seeded RNG, three stroke grammars) but
 * uses its own small seeded PRNG rather than reproducing numpy's PCG64 bit
 * for bit -- the point is a consistent, explainable demo, not byte-for-byte
 * parity with the Python backend.
 */

(() => {
  "use strict";

  // ---- color -------------------------------------------------------------

  const NAMED_HUES = [
    ["red", 0.00], ["orange", 0.08], ["amber", 0.11], ["yellow", 0.15], ["gold", 0.13],
    ["lime", 0.22], ["green", 0.33], ["teal", 0.50], ["cyan", 0.52], ["sky", 0.56],
    ["blue", 0.60], ["indigo", 0.68], ["purple", 0.75], ["violet", 0.78], ["magenta", 0.83],
    ["pink", 0.90], ["rose", 0.95], ["brown", 0.07],
  ];

  function hashStr(s) {
    // FNV-1a over UTF-16 code units; good enough for a stable demo seed.
    let h = 0x811c9dc5;
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
    return h >>> 0; // unsigned 32-bit
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

  function hsvToHex(h, s, v) {
    h = ((h % 1) + 1) % 1;
    s = Math.min(1, Math.max(0, s));
    v = Math.min(1, Math.max(0, v));
    const i = Math.floor(h * 6);
    const f = h * 6 - i;
    const p = v * (1 - s);
    const q = v * (1 - f * s);
    const t = v * (1 - (1 - f) * s);
    let r, g, b;
    switch (i % 6) {
      case 0: [r, g, b] = [v, t, p]; break;
      case 1: [r, g, b] = [q, v, p]; break;
      case 2: [r, g, b] = [p, v, t]; break;
      case 3: [r, g, b] = [p, q, v]; break;
      case 4: [r, g, b] = [t, p, v]; break;
      default: [r, g, b] = [v, p, q]; break;
    }
    const toHex = (x) => Math.round(x * 255).toString(16).padStart(2, "0").toUpperCase();
    return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
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

  function quadraticBezier(p0, p1, p2, n = 14) {
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

  // ---- brief derivation (mirrors derive_brief) -------------------------------

  const GRAMMARS = ["arc_over_line", "nested_curves", "radiating_strokes"];

  function deriveBrief(answers) {
    const seed = textSeed(answers.color, answers.city, answers.dream, answers.mood);
    const rng = mulberry32(seed);

    const baseHue = hueFromText(answers.color);
    const cityHueShift = (hashStr(answers.city.trim().toLowerCase()) % 1000) / 1000 * 0.12;
    const palette = [
      hsvToHex(baseHue, 0.65, 0.78),
      hsvToHex(baseHue + 0.08 + cityHueShift, 0.55, 0.60),
      "#1B1B1B",
    ];

    const grammar = GRAMMARS[seed % GRAMMARS.length];
    const scale = uniform(rng, 0.85, 1.15);
    const rotationDeg = uniform(rng, -20, 20);

    return { seed, grammar, palette, scale, rotationDeg };
  }

  // ---- grammars (mirror composition.py's _grammar_* functions) --------------

  function grammarArcOverLine(rng, palette, scale) {
    const c = [0.5, 0.5];
    const half = 0.32 * scale;
    const p0 = [c[0] - half, c[1] - half * uniform(rng, 0.8, 1.1)];
    const p1 = [c[0] + half, c[1] + half * uniform(rng, 0.8, 1.1)];
    const strokes = [
      { name: "rising_diagonal", color: palette[0], points: clip01(line(p0, p1)) },
      {
        name: "closing_arc",
        color: palette[0],
        points: clip01(arc([p1[0], p1[1] - 0.12 * scale], 0.14 * scale, -40, 220)),
      },
    ];
    const accentOrigin = [c[0] + uniform(rng, -0.3, -0.1) * scale, c[1] + uniform(rng, -0.1, 0.1) * scale];
    const accentEnd = [accentOrigin[0] + uniform(rng, 0.1, 0.22) * scale, accentOrigin[1] + uniform(rng, -0.05, 0.05) * scale];
    strokes.push({ name: "horizon_accent", color: palette[2], points: clip01(line(accentOrigin, accentEnd)) });
    const dashOrigin = [c[0] + uniform(rng, 0.05, 0.25) * scale, c[1] + uniform(rng, -0.25, -0.1) * scale];
    const dashEnd = [dashOrigin[0] + uniform(rng, 0.06, 0.12) * scale, dashOrigin[1] + uniform(rng, 0.02, 0.06) * scale];
    strokes.push({ name: "accent_dash", color: palette[2], points: clip01(line(dashOrigin, dashEnd)) });
    return strokes;
  }

  function grammarNestedCurves(rng, palette, scale) {
    const c = [0.5, 0.5];
    const strokes = [];
    for (let i = 0; i < 3; i++) {
      const r = (0.14 + 0.09 * i) * scale;
      const p0 = [c[0] - r, c[1] + r * 0.3];
      const p1 = [c[0], c[1] - r * uniform(rng, 0.5, 0.9)];
      const p2 = [c[0] + r, c[1] + r * 0.3];
      strokes.push({ name: `nested_curve_${i}`, color: palette[i % 2], points: clip01(quadraticBezier(p0, p1, p2)) });
    }
    const dashOrigin = [c[0] + uniform(rng, -0.3, 0.3) * scale, c[1] + uniform(rng, 0.2, 0.3) * scale];
    const dashEnd = [dashOrigin[0] + uniform(rng, 0.06, 0.14) * scale, dashOrigin[1] + uniform(rng, -0.03, 0.03) * scale];
    strokes.push({ name: "accent_dash", color: palette[2], points: clip01(line(dashOrigin, dashEnd)) });
    return strokes;
  }

  function grammarRadiatingStrokes(rng, palette, scale) {
    const c = [0.5, 0.5];
    const nRays = 4;
    const strokes = [];
    const baseAngle = uniform(rng, 0, 360);
    for (let i = 0; i < nRays; i++) {
      const angle = ((baseAngle + i * (360 / nRays) + uniform(rng, -10, 10)) * Math.PI) / 180;
      const length = (0.18 + 0.05 * (i % 2)) * scale;
      const p0 = [c[0] + 0.05 * scale * Math.cos(angle), c[1] + 0.05 * scale * Math.sin(angle)];
      const p1 = [c[0] + length * Math.cos(angle), c[1] + length * Math.sin(angle)];
      strokes.push({ name: `ray_${i}`, color: palette[i % 2], points: clip01(line(p0, p1)) });
    }
    strokes.push({ name: "center_arc", color: palette[2], points: clip01(arc(c, 0.07 * scale, 0, 300)) });
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
    let strokes = GRAMMAR_FNS[brief.grammar](rng, brief.palette, brief.scale);
    if (Math.abs(brief.rotationDeg) > 1e-6) {
      const center = [0.5, 0.5];
      strokes = strokes.map((s) => ({ ...s, points: clip01(rotatePoints(s.points, center, brief.rotationDeg)) }));
    }
    return { strokes, brief };
  }

  // ---- rendering -------------------------------------------------------------

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

  function pointsToPath(points) {
    return points
      .map(([x, y], i) => `${i === 0 ? "M" : "L"} ${(x * 100).toFixed(2)},${((1 - y) * 100).toFixed(2)}`)
      .join(" ");
  }

  function renderPainting(strokes, brief, robotKey) {
    canvasEl.innerHTML = "";
    placeholderEl.hidden = true;

    strokes.forEach((stroke, i) => {
      const path = document.createElementNS(SVG_NS, "path");
      path.setAttribute("d", pointsToPath(stroke.points));
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", stroke.color);
      path.setAttribute("stroke-width", "1.6");
      path.setAttribute("stroke-linecap", "round");
      path.setAttribute("stroke-linejoin", "round");
      canvasEl.appendChild(path);

      const len = path.getTotalLength();
      path.style.strokeDasharray = `${len}`;
      path.style.strokeDashoffset = `${len}`;
      path.animate(
        [{ strokeDashoffset: len }, { strokeDashoffset: 0 }],
        { duration: 650, delay: i * 420, easing: "ease-in-out", fill: "forwards" }
      );
    });

    briefEl.hidden = false;
    briefGrammarEl.textContent = brief.grammar.replace(/_/g, " ");
    briefStrokesEl.textContent = `${strokes.length} of 5`;
    briefRobotEl.textContent = ROBOT_LABELS[robotKey] || robotKey;
    briefPaletteEl.innerHTML = "";
    brief.palette.forEach((hex) => {
      const sw = document.createElement("span");
      sw.className = "swatch";
      sw.style.background = hex;
      sw.title = hex;
      briefPaletteEl.appendChild(sw);
    });
  }

  // ---- wiring ------------------------------------------------------------

  const EXAMPLES = [
    { color: "teal", city: "Austin", dream: "to make music", mood: "playful" },
    { color: "sunset orange", city: "Paris", dream: "to open a bakery", mood: "cozy" },
    { color: "deep purple", city: "Shanghai", dream: "to become a scientist", mood: "determined" },
    { color: "forest green", city: "Tokyo", dream: "to write a novel", mood: "calm" },
    { color: "blue", city: "Pittsburgh", dream: "to build robots that help people", mood: "curious" },
  ];

  const form = document.getElementById("kiosk-form");

  function runFromForm() {
    const answers = {
      color: form.color.value || "blue",
      city: form.city.value || "a city",
      dream: form.dream.value || "a dream",
      mood: form.mood.value || "",
    };
    const { strokes, brief } = compose(answers);
    renderPainting(strokes, brief, form.robot.value);
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
    runFromForm();
  });
})();
