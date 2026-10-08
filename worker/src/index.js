/**
 * Robot4Art composition backend: the only piece of this project that
 * isn't static. The GitHub Pages site has no server, so the real
 * stroke-by-stroke LLM composition used for the published Isaac Sim
 * example (sim/llm_composer.py) can't run from a visitor's browser --
 * it needs a real Anthropic API key, which must never reach client code.
 * This Worker is that boundary: it holds the key as a secret, builds the
 * same kind of direct-authorship prompt llm_composer.py uses (the model
 * places every stroke's own points/color/weight, not a template), calls
 * the Anthropic Messages API directly (no Claude Code CLI involved --
 * this runs on Cloudflare's edge, not inside an authenticated dev
 * session), validates the response, and returns clean stroke data for
 * docs/app.js to render with its existing SVG rendering code.
 *
 * Rate-limited per visitor IP and with a global daily cap (both via
 * Workers KV) since this spends real money per request and is reachable
 * by anyone who loads the public page.
 */

const MAX_STROKES = 20;
const CANVAS_MARGIN = 0.08;
const MODEL = "claude-sonnet-5";
// With extended thinking left on, this model burns the whole output
// budget on internal reasoning and never emits the final JSON (hit
// stop_reason "max_tokens" with zero text content in testing) --
// thinking is explicitly disabled below so the full budget goes to the
// actual answer.
const MAX_OUTPUT_TOKENS = 2000;

// Mirrors composition.py's STANDARD_PENS -- the robot holds this fixed
// set of interchangeable pens, not custom-mixed paint.
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

// Mirrors composition.py's BRUSHES (label only needed for the prompt).
const BRUSH_DESCRIPTIONS = {
  fine_pen: "thin, crisp, no taper",
  marker: "medium width, slight taper",
  brush: "thick, fully tapered gesture",
  watercolor: "loose overlapping dabs",
};

// Optional visitor-facing "style lean" -- a soft nudge in the prompt,
// not a template the model fills in (unlike composition.py's STYLES,
// which hard-codes a grammar function per style). The model still
// authors every stroke itself; this just colors its taste.
const STYLE_HINTS = {
  modernist: "a bold, confident, architectural feel -- sweeping gestures, strong negative space",
  impressionist: "a soft, blooming feel -- layered petal-like strokes, gentle overlapping color",
  ink_wash: "a quiet, minimal feel -- thin layered curves, lots of empty canvas, restraint over bravado",
};

// Per-IP cooldown and a global daily cap -- this calls a real paid API
// from a public page, so both exist to bound cost/abuse. Adjust freely;
// these are deliberately conservative defaults. 60 is Workers KV's own
// minimum expirationTtl -- a shorter cooldown would need a timestamp
// comparison instead of relying on KV's own expiry.
const PER_IP_COOLDOWN_SECONDS = 60;
const GLOBAL_DAILY_CAP = 150;

function corsHeaders(origin) {
  return {
    "Access-Control-Allow-Origin": origin || "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
  };
}

function jsonResponse(data, status, origin) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json", ...corsHeaders(origin) },
  });
}

function buildPrompt(answers) {
  const pens = Object.keys(STANDARD_PENS).join(", ");
  const brushes = Object.entries(BRUSH_DESCRIPTIONS)
    .map(([name, desc]) => `- ${name} (${desc})`)
    .join("\n");
  return `You are an abstract painter. A robot arm will physically paint exactly what you design here, stroke by stroke, so design a real composition yourself -- don't just pick from a template.

The visitor answered a few personal prompts:
- Favorite color: ${answers.color}
- Favorite city: ${answers.city}
- Dream/aspiration: ${answers.dream}
- Mood: ${answers.mood || "(not given)"}
${STYLE_HINTS[answers.style] ? `\nThe visitor leans toward ${STYLE_HINTS[answers.style]}. Let that inform your composition, but still design it yourself -- it's a lean, not a template to fill in.\n` : ""}
Canvas: a normalized square, x and y both in [0, 1], (0, 0) at the bottom-left. Keep all points within [${CANVAS_MARGIN}, ${1 - CANVAS_MARGIN}] of each axis.

Constraints (the robot physically has these, not a stylistic choice):
- At most ${MAX_STROKES} strokes total.
- Each stroke is a smooth path through 2 to 6 points you choose -- pick the points that define the shape you want; they'll be smoothed into a curve automatically, you don't need to output a dense path.
- Each stroke uses exactly one pen color, from: ${pens}.
- Pick exactly one brush/tool for the whole piece (it physically can't be swapped mid-painting):
${brushes}

Design an actual abstract composition -- real visual balance, intentional negative space, a clear focal gesture plus supporting strokes, varied stroke lengths and directions and weights -- not a generic diagram, not a symmetric pattern, not one shape repeated with rotations. Let the visitor's answers genuinely shape the composition (their mood, their dream, the feeling of their color and city), not just which colors you pick.

Respond with ONLY a single JSON object, no markdown fences, no other text:
{
  "brush": "<one of the brush names above>",
  "strokes": [
    {"color": "<one of the pen names above>", "points": [[x, y], [x, y], ...], "width": <0.3 to 1.0, relative weight of this stroke>},
    ...
  ],
  "rationale": "2-4 sentences, first person plural 'we', explaining the actual composition you designed and how it connects to the visitor's specific answers"
}`;
}

/** Parse the model's JSON response, tolerating a markdown code fence
 * around it (the prompt asks for none, but models sometimes add one
 * anyway) and falling back to grabbing the first {...} block. */
function extractJson(text) {
  let t = text.trim();
  const fenceMatch = t.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/);
  if (fenceMatch) t = fenceMatch[1].trim();
  try {
    return JSON.parse(t);
  } catch {
    // fall through
  }
  const braceMatch = t.match(/\{[\s\S]*\}/);
  if (braceMatch) {
    try {
      return JSON.parse(braceMatch[0]);
    } catch {
      // fall through
    }
  }
  return null;
}

function badRequest(msg) {
  const e = new Error(msg);
  e.status = 400;
  return e;
}

/** Validate and normalize one model response into clean stroke data --
 * mirrors sim/llm_composer.py's _parse_painting_response. Raises instead
 * of silently fixing anything beyond simple clamping, so a malformed
 * model response surfaces as a clear error rather than a wrong painting. */
function parsePaintingResponse(inner) {
  if (!BRUSH_DESCRIPTIONS[inner.brush]) {
    throw badRequest(`model chose an unknown brush: ${inner.brush}`);
  }
  const rawStrokes = inner.strokes;
  if (!Array.isArray(rawStrokes) || rawStrokes.length < 1 || rawStrokes.length > MAX_STROKES) {
    throw badRequest(`model returned an invalid stroke count`);
  }

  const strokes = rawStrokes.map((raw, i) => {
    const colorName = String(raw.color || "").trim().toLowerCase();
    if (!STANDARD_PENS[colorName]) {
      throw badRequest(`stroke ${i} has an unknown color: ${raw.color}`);
    }
    const rawPoints = raw.points;
    if (!Array.isArray(rawPoints) || rawPoints.length < 2 || rawPoints.length > 8) {
      throw badRequest(`stroke ${i} has an invalid point count`);
    }
    const points = rawPoints.map((p) => {
      const x = Math.min(1 - CANVAS_MARGIN, Math.max(CANVAS_MARGIN, Number(p[0])));
      const y = Math.min(1 - CANVAS_MARGIN, Math.max(CANVAS_MARGIN, Number(p[1])));
      if (Number.isNaN(x) || Number.isNaN(y)) throw badRequest(`stroke ${i} has malformed points`);
      return [x, y];
    });
    let width = Number(raw.width);
    if (Number.isNaN(width)) width = 0.7;
    width = Math.min(1.0, Math.max(0.3, width));
    return { name: `stroke_${i}`, color: STANDARD_PENS[colorName], points, width };
  });

  const rationale = String(inner.rationale || "").trim();
  if (!rationale) throw badRequest("model response had no rationale text");

  return { brush: inner.brush, strokes, rationale };
}

async function checkRateLimit(env, ip) {
  const ipKey = `cooldown:${ip}`;
  if (await env.RATE_LIMIT.get(ipKey)) {
    throw Object.assign(new Error("Please wait a bit before trying again."), { status: 429 });
  }
  const today = new Date().toISOString().slice(0, 10);
  const dayKey = `count:${today}`;
  const countStr = await env.RATE_LIMIT.get(dayKey);
  const count = countStr ? parseInt(countStr, 10) : 0;
  if (count >= GLOBAL_DAILY_CAP) {
    throw Object.assign(new Error("This demo has hit its daily limit -- please try again tomorrow."), { status: 429 });
  }
  await env.RATE_LIMIT.put(ipKey, "1", { expirationTtl: PER_IP_COOLDOWN_SECONDS });
  await env.RATE_LIMIT.put(dayKey, String(count + 1), { expirationTtl: 60 * 60 * 25 });
}

export default {
  async fetch(request, env) {
    const origin = request.headers.get("Origin");
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders(origin) });
    }
    if (request.method !== "POST") {
      return jsonResponse({ error: "Method not allowed" }, 405, origin);
    }

    try {
      const ip = request.headers.get("CF-Connecting-IP") || "unknown";
      await checkRateLimit(env, ip);

      let answers;
      try {
        answers = await request.json();
      } catch {
        throw badRequest("invalid JSON body");
      }
      if (!answers.color || !answers.city || !answers.dream) {
        throw badRequest("color, city, and dream are required");
      }

      const prompt = buildPrompt(answers);
      const apiResp = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: {
          "x-api-key": env.ANTHROPIC_API_KEY,
          "anthropic-version": "2023-06-01",
          "content-type": "application/json",
        },
        body: JSON.stringify({
          model: MODEL,
          max_tokens: MAX_OUTPUT_TOKENS,
          thinking: { type: "disabled" },
          messages: [{ role: "user", content: prompt }],
        }),
      });

      if (!apiResp.ok) {
        const errText = await apiResp.text();
        throw Object.assign(new Error(`model API error: ${errText.slice(0, 300)}`), { status: 502 });
      }
      const data = await apiResp.json();
      const textBlock = (data.content || []).find((b) => b.type === "text");
      const text = (textBlock && textBlock.text) || "";
      const inner = extractJson(text);
      if (!inner) {
        throw Object.assign(new Error("model did not return valid JSON"), { status: 502 });
      }

      const result = parsePaintingResponse(inner);
      return jsonResponse(result, 200, origin);
    } catch (err) {
      const status = err.status || 500;
      return jsonResponse({ error: err.message || "internal error" }, status, origin);
    }
  },
};
