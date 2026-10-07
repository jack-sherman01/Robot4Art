# Robot4Art composition backend

The only non-static piece of this project. `docs/` is pure GitHub Pages
with no server, so the real stroke-by-stroke LLM composition (the same
direct-authorship approach as `sim/llm_composer.py` -- the model places
every stroke's own points, color, and weight, not a template) needs
somewhere to hold a real Anthropic API key server-side. This Cloudflare
Worker is that boundary: `docs/app.js` calls it, it calls the Anthropic
API, the key never reaches the browser.

## Deploy

Requires a Cloudflare account (free tier is enough) and an Anthropic API
key, billed to you.

```bash
cd worker
npx wrangler login                              # opens a browser to authenticate

npx wrangler kv namespace create RATE_LIMIT      # prints an id --
# paste that id into wrangler.toml's `id = "REPLACE_WITH_KV_NAMESPACE_ID"`

npx wrangler secret put ANTHROPIC_API_KEY        # pastes the key without it
                                                  # touching shell history or this chat

npx wrangler deploy
```

The last command prints the Worker's URL
(`https://robot4art-composer.<your-subdomain>.workers.dev`). Put that
URL into `docs/app.js`'s `COMPOSER_API_URL` constant and redeploy the
GitHub Pages site.

## Cost / abuse protection

Rate-limited two ways via Workers KV (`src/index.js`):
- **Per-IP cooldown**: 25 seconds between requests from the same visitor.
- **Global daily cap**: 150 requests/day across all visitors, reset at
  midnight UTC.

Both are deliberately conservative starting points -- adjust
`PER_IP_COOLDOWN_SECONDS` / `GLOBAL_DAILY_CAP` in `src/index.js` and
redeploy. Each request costs roughly what a single `claude-sonnet-5`
call with a ~2000-token budget costs on your Anthropic account (check
Anthropic's current pricing) -- at the daily cap that's a bounded,
predictable worst case, not unlimited exposure.

## Local testing

```bash
npx wrangler dev
```

Then `curl -X POST http://localhost:8787 -H 'Content-Type: application/json' \
  -d '{"color":"teal","city":"Pittsburgh","dream":"to build robots that help people","mood":"curious"}'`
