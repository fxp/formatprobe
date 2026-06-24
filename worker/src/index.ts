export interface Env {
  DB: D1Database;
  INGEST_SECRET: string;
  GH_REPO: string;    // e.g. "yourname/formatprobe"
  GH_TOKEN: string;
}

interface CheckPayload {
  name: string;
  severity: "ok" | "warn" | "error" | "skip";
  message?: string;
  latency_ms?: number;
}

interface IngestBody {
  provider: string;   // "智谱 GLM"
  slug: string;       // "glm"
  model: string;      // "glm-4-flash"
  run_at: string;     // ISO 8601
  checks: CheckPayload[];
}

const CORS = {
  "Access-Control-Allow-Origin":  "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
} as const;

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { ...CORS, "Content-Type": "application/json" },
  });
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: CORS });

    const url = new URL(req.url);

    // ── POST /ingest ────────────────────────────────────────────────────────────
    if (req.method === "POST" && url.pathname === "/ingest") {
      if (req.headers.get("Authorization") !== `Bearer ${env.INGEST_SECRET}`) {
        return json({ error: "Unauthorized" }, 401);
      }

      let body: IngestBody;
      try {
        body = await req.json() as IngestBody;
      } catch {
        return json({ error: "Invalid JSON" }, 400);
      }

      const { provider, slug, model, run_at, checks } = body;
      if (!provider || !slug || !model || !run_at || !Array.isArray(checks)) {
        return json({ error: "Missing required fields" }, 400);
      }

      const stmt = env.DB.prepare(
        `INSERT INTO runs (id, provider, slug, model, check_name, severity, pass, latency_ms, message, run_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)`
      );

      await env.DB.batch(
        checks.map(c =>
          stmt.bind(
            crypto.randomUUID(),
            provider, slug, model,
            c.name, c.severity,
            c.severity === "ok" ? 1 : 0,
            c.latency_ms ?? null,
            c.message ?? null,
            run_at,
          )
        )
      );

      return json({ ok: true, inserted: checks.length });
    }

    // ── GET /api/leaderboard ─────────────────────────────────────────────────
    if (url.pathname === "/api/leaderboard") {
      const { results } = await env.DB.prepare(`
        WITH latest AS (
          SELECT slug, model, MAX(run_at) AS last_run_at
          FROM runs GROUP BY slug, model
        )
        SELECT
          r.provider, r.slug, r.model,
          l.last_run_at,
          COUNT(*)                                                               AS total,
          SUM(CASE WHEN r.severity = 'ok'    THEN 1 ELSE 0 END)                AS ok_count,
          SUM(CASE WHEN r.severity = 'warn'  THEN 1 ELSE 0 END)                AS warn_count,
          SUM(CASE WHEN r.severity = 'error' THEN 1 ELSE 0 END)                AS error_count,
          SUM(CASE WHEN r.severity = 'skip'  THEN 1 ELSE 0 END)                AS skip_count,
          ROUND(
            100.0 * SUM(CASE WHEN r.severity = 'ok' THEN 1 ELSE 0 END) /
            NULLIF(COUNT(*) - SUM(CASE WHEN r.severity = 'skip' THEN 1 ELSE 0 END), 0),
            1
          )                                                                      AS pass_rate
        FROM runs r
        JOIN latest l ON r.slug = l.slug AND r.model = l.model AND r.run_at = l.last_run_at
        GROUP BY r.slug, r.model
        ORDER BY pass_rate DESC
      `).all();

      return json(results);
    }

    // ── GET /api/provider/:slug/latest ───────────────────────────────────────
    const latestM = url.pathname.match(/^\/api\/provider\/([^/]+)\/latest$/);
    if (latestM) {
      const slug = latestM[1];
      const { results } = await env.DB.prepare(`
        WITH top AS (SELECT MAX(run_at) AS max_at FROM runs WHERE slug = ?1)
        SELECT check_name, severity, pass, latency_ms, message, run_at
        FROM runs
        WHERE slug = ?1 AND run_at = (SELECT max_at FROM top)
        ORDER BY check_name
      `).bind(slug).all();
      return json(results);
    }

    // ── GET /api/provider/:slug/history?days=30 ──────────────────────────────
    const histM = url.pathname.match(/^\/api\/provider\/([^/]+)\/history$/);
    if (histM) {
      const slug = histM[1];
      const days = Math.min(90, Math.max(1, parseInt(url.searchParams.get("days") ?? "30")));
      const { results } = await env.DB.prepare(`
        SELECT
          strftime('%Y-%m-%d', run_at)                                                AS date,
          COUNT(DISTINCT run_at)                                                       AS runs,
          ROUND(
            100.0 * SUM(CASE WHEN severity = 'ok' THEN 1 ELSE 0 END) /
            NULLIF(COUNT(*) - SUM(CASE WHEN severity = 'skip' THEN 1 ELSE 0 END), 0),
            1
          )                                                                            AS pass_rate
        FROM runs
        WHERE slug = ?1 AND run_at > datetime('now', ?2)
        GROUP BY date
        ORDER BY date
      `).bind(slug, `-${days} days`).all();
      return json(results);
    }

    // ── GET /api/provider/:slug/checks ───────────────────────────────────────
    const checksM = url.pathname.match(/^\/api\/provider\/([^/]+)\/checks$/);
    if (checksM) {
      const slug = checksM[1];
      const days = Math.min(30, Math.max(1, parseInt(url.searchParams.get("days") ?? "7")));
      const { results } = await env.DB.prepare(`
        SELECT
          check_name,
          COUNT(*)                                                                AS runs,
          ROUND(
            100.0 * SUM(CASE WHEN severity = 'ok' THEN 1 ELSE 0 END) /
            NULLIF(COUNT(*) - SUM(CASE WHEN severity = 'skip' THEN 1 ELSE 0 END), 0),
            1
          )                                                                      AS pass_rate,
          ROUND(AVG(latency_ms), 0)                                             AS avg_latency_ms
        FROM runs
        WHERE slug = ?1 AND run_at > datetime('now', ?2)
        GROUP BY check_name
        ORDER BY pass_rate, check_name
      `).bind(slug, `-${days} days`).all();
      return json(results);
    }

    return json({ error: "Not found" }, 404);
  },

  // Cron: triggers GitHub Actions probe workflow every 6 hours
  async scheduled(_event: ScheduledEvent, env: Env): Promise<void> {
    if (!env.GH_REPO || !env.GH_TOKEN) return;
    await fetch(
      `https://api.github.com/repos/${env.GH_REPO}/actions/workflows/probe.yml/dispatches`,
      {
        method: "POST",
        headers: {
          Authorization:  `Bearer ${env.GH_TOKEN}`,
          Accept:         "application/vnd.github.v3+json",
          "Content-Type": "application/json",
          "User-Agent":   "formatprobe-worker/1.0",
        },
        body: JSON.stringify({ ref: "main" }),
      }
    );
  },
};
