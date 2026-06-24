#!/usr/bin/env bash
# 批量诊断多个端点，并汇总结果
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$DIR/llm_diagnose.py"
REPORT_DIR="$DIR/reports"
mkdir -p "$REPORT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "═══════════════════════════════════════════"
echo " LLM API 批量诊断  $(date)"
echo "═══════════════════════════════════════════"

PASS=0
FAIL=0

run_check() {
  local NAME="$1"
  local URL="$2"
  local MODEL="$3"
  local PROTO="$4"
  local KEY_VAR="${5:-LLM_API_KEY}"
  shift 5
  local EXTRA=("$@")

  local OUT="$REPORT_DIR/${TIMESTAMP}_${NAME// /_}.json"
  local KEY="${!KEY_VAR:-}"

  echo ""
  echo "▶ [$NAME]  $URL  ($MODEL)"
  if python3 "$SCRIPT" diagnose "$URL" \
      --model "$MODEL" \
      --protocol "$PROTO" \
      ${KEY:+--api-key "$KEY"} \
      --output "$OUT" \
      "${EXTRA[@]}"; then
    echo "  → PASS  报告: $OUT"
    ((PASS++)) || true
  else
    echo "  → FAIL  报告: $OUT"
    ((FAIL++)) || true
  fi
}

# ── 在这里添加你的端点 ───────────────────────────────────────────────────────
# run_check "名称" "URL" "模型" "协议" "API_KEY_ENV变量名"

# 示例:
# run_check "OpenAI"    "https://api.openai.com"    "gpt-4o-mini"               "openai"    "OPENAI_API_KEY"
# run_check "Anthropic" "https://api.anthropic.com" "claude-3-5-haiku-20241022" "anthropic" "ANTHROPIC_API_KEY"
# run_check "Ollama"    "http://localhost:11434"    "llama3"                    "openai"    ""

# 如果只有一个目标快速测试:
if [ "${1:-}" ]; then
  run_check "Target" "$1" "${2:-gpt-4o-mini}" "${3:-auto}" "${4:-LLM_API_KEY}"
else
  echo ""
  echo "用法: $0 <url> [model] [protocol] [KEY_ENV_VAR]"
  echo "或者：直接编辑本文件的 run_check 行，然后执行"
  exit 0
fi

echo ""
echo "═══════════════════════════════════════════"
echo " 汇总: PASS=$PASS  FAIL=$FAIL"
echo "═══════════════════════════════════════════"

[ "$FAIL" -eq 0 ] || exit 1
