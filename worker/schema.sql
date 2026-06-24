-- FormatProbe D1 schema
-- Apply with: wrangler d1 execute formatprobe --file=schema.sql

CREATE TABLE IF NOT EXISTS runs (
  id         TEXT NOT NULL PRIMARY KEY,
  provider   TEXT NOT NULL,    -- "智谱 GLM"
  slug       TEXT NOT NULL,    -- "glm"
  model      TEXT NOT NULL,    -- "glm-4-flash"
  check_name TEXT NOT NULL,    -- "chat_basic"
  severity   TEXT NOT NULL CHECK(severity IN ('ok', 'warn', 'error', 'skip')),
  pass       INTEGER NOT NULL DEFAULT 0,
  latency_ms REAL,
  message    TEXT,
  run_at     TEXT NOT NULL     -- ISO 8601
);

CREATE INDEX IF NOT EXISTS idx_slug_run_at ON runs(slug, run_at DESC);
CREATE INDEX IF NOT EXISTS idx_run_at      ON runs(run_at DESC);
CREATE INDEX IF NOT EXISTS idx_slug_check  ON runs(slug, check_name, run_at DESC);
