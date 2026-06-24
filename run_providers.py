#!/usr/bin/env python3
"""
run_providers.py — 一键诊断五家国内 LLM API
支持: 智谱 GLM / Moonshot KIMI / Minimax / 通义千问 Qwen / DeepSeek

用法:
  python3 run_providers.py                          # 运行全部已配置 key 的厂商
  python3 run_providers.py --only glm kimi          # 只测指定厂商
  python3 run_providers.py --skip-formats           # 跳过格式生成检测
  python3 run_providers.py --parallel               # 并发运行（可能触发限流）
  python3 run_providers.py --output reports/        # 保存 JSON 报告到目录
  python3 run_providers.py --compare                # 仅显示最终对比矩阵（读已有报告）
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
import typer
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

# ── 从主模块导入 ───────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from llm_diagnose import (
    CheckResult,
    DiagReport,
    Protocol,
    Severity,
    SEVERITY_STYLE,
    make_client,
    detect_protocol,
    run_openai_checks,
    run_format_checks,
    render_report,
)

# ── 加载 .env 文件 ─────────────────────────────────────────────────────────────

def load_dotenv(path: Path = Path(__file__).parent / ".env") -> None:
    """极简 .env 加载器，无需 python-dotenv 依赖。"""
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and v and k not in os.environ:
                os.environ[k] = v


load_dotenv()

# ── 厂商配置 ──────────────────────────────────────────────────────────────────

@dataclass
class ProviderConfig:
    name: str              # 显示名称
    slug: str              # CLI 简称（用于 --only）
    base_url: str          # API Base URL
    default_model: str     # 默认测试模型
    api_key_env: str       # 环境变量名
    protocol: str = "openai"
    extra_headers: dict = field(default_factory=dict)
    notes: str = ""        # 厂商特有备注


PROVIDERS: list[ProviderConfig] = [
    ProviderConfig(
        name="智谱 GLM",
        slug="glm",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        default_model="glm-4-plus",           # 主流付费旗舰；免费可用 glm-4-flash
        api_key_env="GLM_API_KEY",
        notes="完全 OpenAI 兼容；glm-4-plus 为付费旗舰，格式遵循度优于 flash",
    ),
    ProviderConfig(
        name="Moonshot KIMI",
        slug="kimi",
        base_url="https://api.moonshot.cn/v1",
        default_model="kimi-k2",              # 2025 年发布的新旗舰；回退可用 moonshot-v1-8k
        api_key_env="MOONSHOT_API_KEY",
        notes="完全 OpenAI 兼容；kimi-k2 为当前主流旗舰模型",
    ),
    ProviderConfig(
        name="Minimax",
        slug="minimax",
        base_url="https://api.minimaxi.com/v1",
        default_model="MiniMax-Text-01",       # 当前主流，MiniMax-M1 为推理模型
        api_key_env="MINIMAX_API_KEY",
        notes="新版 OpenAI 兼容接口，响应含 base_resp/input_sensitive 扩展字段",
    ),
    ProviderConfig(
        name="通义千问 Qwen",
        slug="qwen",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model="qwen-plus",             # 主流均衡模型；qwen-turbo 更快但格式合规较弱
        api_key_env="DASHSCOPE_API_KEY",
        notes="DashScope OpenAI 兼容模式；qwen-plus 格式合规优于 qwen-turbo",
    ),
    ProviderConfig(
        name="DeepSeek",
        slug="deepseek",
        base_url="https://api.deepseek.com",
        default_model="deepseek-chat",         # deepseek-v3 别名，当前主流
        api_key_env="DEEPSEEK_API_KEY",
        notes="deepseek-chat 为 deepseek-v3 别名，完全 OpenAI 兼容",
    ),
    ProviderConfig(
        name="OpenRouter",
        slug="openrouter",
        base_url="https://openrouter.ai/api/v1",
        default_model="qwen/qwen3-235b-a22b:free",  # 免费高质量模型，格式能力强
        api_key_env="OPENROUTER_API_KEY",
        extra_headers={"HTTP-Referer": "https://llm-diagnose", "X-Title": "LLM API Diagnose"},
        notes="聚合路由，OpenAI 兼容；qwen3-235b 免费且格式遵循度高",
    ),
]

PROVIDER_MAP = {p.slug: p for p in PROVIDERS}

console = Console(force_terminal=True, no_color=not __import__("sys").stdout.isatty())
app = typer.Typer(
    name="run-providers",
    help="批量诊断国内五家 LLM API 并输出对比报告",
    rich_markup_mode="rich",
)


# ── 厂商专项补充检测 ──────────────────────────────────────────────────────────

async def check_minimax_extra_fields(
    base_url: str, client: httpx.AsyncClient, model: str
) -> CheckResult:
    """
    Minimax 响应体包含非标准扩展字段：
    base_resp.status_code / input_sensitive / output_sensitive
    检验这些字段的存在与合理值。
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 16,
        "stream": False,
    }
    t0 = time.monotonic()
    try:
        r = await client.post(url, json=payload)
        latency = (time.monotonic() - t0) * 1000
        if r.status_code != 200:
            return CheckResult(
                "minimax_ext_fields", Severity.warn,
                f"HTTP {r.status_code} — 无法检查扩展字段",
                r.text[:100], latency,
            )
        try:
            data = r.json()
        except Exception as e:
            return CheckResult("minimax_ext_fields", Severity.error, "响应非 JSON", str(e), latency)

        issues: list[str] = []
        notes: list[str] = []

        # base_resp
        base_resp = data.get("base_resp")
        if base_resp is None:
            notes.append("base_resp 字段不存在（新版可能已移除）")
        else:
            code = base_resp.get("status_code")
            msg = base_resp.get("status_msg", "")
            if code != 0:
                issues.append(f"base_resp.status_code={code} ({msg})")
            else:
                notes.append(f"base_resp.status_code=0 OK")

        # input_sensitive / output_sensitive
        for field_name in ("input_sensitive", "output_sensitive"):
            val = data.get(field_name)
            if val is None:
                notes.append(f"{field_name} 不存在")
            elif val is True:
                issues.append(f"{field_name}=true 内容被标记敏感")
            else:
                notes.append(f"{field_name}=false")

        severity = Severity.error if any("敏感" in i for i in issues) else (
            Severity.warn if issues else Severity.ok
        )
        detail = " | ".join(notes[:4])
        msg = "Minimax 扩展字段正常" if not issues else f"扩展字段异常: {issues[0]}"
        return CheckResult("minimax_ext_fields", severity, msg, detail, latency)

    except httpx.TimeoutException:
        return CheckResult("minimax_ext_fields", Severity.error, "超时", "")
    except Exception as e:
        return CheckResult("minimax_ext_fields", Severity.error, str(e), "")


async def check_glm_web_search(
    base_url: str, client: httpx.AsyncClient, model: str
) -> CheckResult:
    """
    GLM 支持 web_search 工具（联网搜索），检验接口是否正常触发。
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "今天北京的天气如何？"}],
        "tools": [{"type": "web_search", "web_search": {"enable": True}}],
        "max_tokens": 128,
        "stream": False,
    }
    t0 = time.monotonic()
    try:
        r = await client.post(url, json=payload)
        latency = (time.monotonic() - t0) * 1000
        if r.status_code == 200:
            try:
                data = r.json()
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                # Look for signs of web search being used
                has_search_signal = any(
                    kw in content for kw in ["天气", "℃", "°C", "晴", "阴", "雨", "云", "温度", "今天"]
                )
                return CheckResult(
                    "glm_web_search",
                    Severity.ok if content else Severity.warn,
                    "GLM web_search 工具响应正常" if content else "返回内容为空",
                    f"content={content[:80]}",
                    latency,
                )
            except Exception as e:
                return CheckResult("glm_web_search", Severity.error, "解析失败", str(e), latency)
        elif r.status_code in (400, 422):
            return CheckResult(
                "glm_web_search", Severity.warn,
                f"web_search 工具不支持该模型 (HTTP {r.status_code})",
                r.text[:100], latency,
            )
        else:
            return CheckResult("glm_web_search", Severity.warn, f"HTTP {r.status_code}", "", latency)
    except httpx.TimeoutException:
        return CheckResult("glm_web_search", Severity.error, "超时", "")
    except Exception as e:
        return CheckResult("glm_web_search", Severity.error, str(e), "")


async def check_kimi_long_context(
    base_url: str, client: httpx.AsyncClient, model: str
) -> CheckResult:
    """
    KIMI 以超长上下文著称，验证 128k context 基础能力（送入较长 prompt）。
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    # 约 2000 token 的重复文本，测试长文本处理
    long_text = ("The quick brown fox jumps over the lazy dog. " * 200).strip()
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Here is a repeated text:\n\n{long_text}\n\n"
                    "Count how many times the word 'fox' appears. "
                    "Reply with just the number."
                ),
            }
        ],
        "max_tokens": 16,
        "stream": False,
    }
    t0 = time.monotonic()
    try:
        r = await client.post(url, json=payload, timeout=httpx.Timeout(60, connect=10))
        latency = (time.monotonic() - t0) * 1000
        if r.status_code == 200:
            try:
                data = r.json()
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
                # "fox" appears 200 times
                digits = "".join(c for c in content if c.isdigit())
                answer = int(digits) if digits else None
                if answer == 200:
                    return CheckResult(
                        "kimi_long_ctx", Severity.ok,
                        f"长上下文计数准确 (fox=200)",
                        f"回答: {content!r} latency={latency:.0f}ms", latency,
                    )
                elif answer is not None:
                    return CheckResult(
                        "kimi_long_ctx", Severity.warn,
                        f"长上下文计数偏差: 回答={answer}，正确=200",
                        f"content={content!r}", latency,
                    )
                else:
                    return CheckResult(
                        "kimi_long_ctx", Severity.warn,
                        "长上下文响应非数字",
                        f"content={content[:60]!r}", latency,
                    )
            except Exception as e:
                return CheckResult("kimi_long_ctx", Severity.error, "解析失败", str(e), latency)
        elif r.status_code == 400:
            return CheckResult(
                "kimi_long_ctx", Severity.warn,
                f"长上下文请求被拒绝 (HTTP 400)",
                r.text[:120], latency,
            )
        else:
            return CheckResult("kimi_long_ctx", Severity.warn, f"HTTP {r.status_code}", "", latency)
    except httpx.TimeoutException:
        return CheckResult("kimi_long_ctx", Severity.error, "长上下文请求超时 (>60s)", "")
    except Exception as e:
        return CheckResult("kimi_long_ctx", Severity.error, str(e), "")


async def check_deepseek_reasoning(
    base_url: str, client: httpx.AsyncClient
) -> CheckResult:
    """
    DeepSeek 支持 deepseek-reasoner（R1），检验 reasoning_content 字段。
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": "deepseek-reasoner",
        "messages": [{"role": "user", "content": "What is 17 × 23? Show reasoning."}],
        "max_tokens": 256,
        "stream": False,
    }
    t0 = time.monotonic()
    try:
        r = await client.post(url, json=payload, timeout=httpx.Timeout(60, connect=10))
        latency = (time.monotonic() - t0) * 1000
        if r.status_code == 200:
            try:
                data = r.json()
                message = data.get("choices", [{}])[0].get("message", {})
                reasoning = message.get("reasoning_content", "")
                content = message.get("content", "")
                if reasoning:
                    return CheckResult(
                        "deepseek_reasoner", Severity.ok,
                        "reasoning_content 字段存在",
                        f"reasoning={reasoning[:60]!r} answer={content[:40]!r}",
                        latency,
                    )
                else:
                    return CheckResult(
                        "deepseek_reasoner", Severity.warn,
                        "reasoning_content 为空（模型可能不支持）",
                        f"content={content[:60]!r}", latency,
                    )
            except Exception as e:
                return CheckResult("deepseek_reasoner", Severity.error, "解析失败", str(e), latency)
        elif r.status_code in (400, 404):
            return CheckResult(
                "deepseek_reasoner", Severity.warn,
                f"deepseek-reasoner 模型不可用 (HTTP {r.status_code})",
                r.text[:100], latency,
            )
        else:
            return CheckResult("deepseek_reasoner", Severity.warn, f"HTTP {r.status_code}", "", latency)
    except httpx.TimeoutException:
        return CheckResult("deepseek_reasoner", Severity.error, "超时 (>60s)", "")
    except Exception as e:
        return CheckResult("deepseek_reasoner", Severity.error, str(e), "")


async def check_qwen_vl_stub(
    base_url: str, client: httpx.AsyncClient, model: str
) -> CheckResult:
    """
    通义千问支持视觉理解 (qwen-vl-plus)，这里仅检验接口是否存在、
    多模态 messages 格式是否被接受（不传真实图片）。
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": "qwen-vl-plus",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image briefly."},
                    {"type": "image_url", "image_url": {"url": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"}},
                ],
            }
        ],
        "max_tokens": 64,
        "stream": False,
    }
    t0 = time.monotonic()
    try:
        r = await client.post(url, json=payload, timeout=httpx.Timeout(30, connect=10))
        latency = (time.monotonic() - t0) * 1000
        if r.status_code == 200:
            try:
                data = r.json()
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                return CheckResult(
                    "qwen_vl", Severity.ok,
                    "Qwen-VL 多模态接口可用",
                    f"content={content[:80]!r}", latency,
                )
            except Exception as e:
                return CheckResult("qwen_vl", Severity.error, "解析失败", str(e), latency)
        elif r.status_code in (400, 404, 422):
            try:
                err = r.json().get("error", {})
                msg = err.get("message", r.text[:80])
            except Exception:
                msg = r.text[:80]
            return CheckResult(
                "qwen_vl", Severity.warn,
                f"qwen-vl-plus 不可用 (HTTP {r.status_code})",
                msg, latency,
            )
        else:
            return CheckResult("qwen_vl", Severity.warn, f"HTTP {r.status_code}", r.text[:80], latency)
    except httpx.TimeoutException:
        return CheckResult("qwen_vl", Severity.error, "超时", "")
    except Exception as e:
        return CheckResult("qwen_vl", Severity.error, str(e), "")


# ── 厂商专项检测分发 ──────────────────────────────────────────────────────────

async def run_provider_specific(
    slug: str,
    base_url: str,
    client: httpx.AsyncClient,
    model: str,
) -> list[CheckResult]:
    """针对每家厂商补充专属检测项。"""
    if slug == "minimax":
        return [await check_minimax_extra_fields(base_url, client, model)]
    elif slug == "glm":
        return [await check_glm_web_search(base_url, client, model)]
    elif slug == "kimi":
        return [await check_kimi_long_context(base_url, client, model)]
    elif slug == "deepseek":
        return [await check_deepseek_reasoning(base_url, client)]
    elif slug == "qwen":
        return [await check_qwen_vl_stub(base_url, client, model)]
    return []


# ── 单个厂商的完整诊断 ────────────────────────────────────────────────────────

async def diagnose_provider(
    cfg: ProviderConfig,
    model_override: Optional[str] = None,
    skip_formats: bool = False,
    skip_specific: bool = True,
    check_delay: float = 0.0,
    format_only: bool = False,
) -> DiagReport:
    model = model_override or cfg.default_model
    api_key = os.environ.get(cfg.api_key_env)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = DiagReport(
        target_url=cfg.base_url,
        protocol=cfg.protocol,
        model=model,
        timestamp=timestamp,
    )

    if not api_key:
        report.checks.append(CheckResult(
            "auth", Severity.error,
            f"未设置 {cfg.api_key_env}",
            "请在 .env 中配置 API Key",
        ))
        return report

    async with make_client(api_key, cfg.extra_headers) as client:
        # 1. 核心协议检测 + 格式检测
        core = await run_openai_checks(
            cfg.base_url, api_key, model, cfg.extra_headers,
            skip_formats=skip_formats,
            check_delay=check_delay,
            format_only=format_only,
        )
        report.checks.extend(core)

        # 2. 厂商专属检测（默认跳过）
        if not skip_specific:
            specific = await run_provider_specific(cfg.slug, cfg.base_url, client, model)
            report.checks.extend(specific)

    return report


# ── 对比矩阵渲染 ──────────────────────────────────────────────────────────────

# 在对比矩阵中展示的检测项（按顺序）
MATRIX_CHECKS = [
    # 格式合规（主要检测项）
    "chat_basic",
    "json_mode",
    "tool_calling",
    "fmt_json",
    "fmt_jsonl",
    "fmt_svg",
    "fmt_svg_chart",
    "fmt_svg_path",
    "fmt_svg_defs",
    "fmt_xml",
    "fmt_csv",
    "fmt_html",
    "fmt_markdown",
    "fmt_yaml",
    "fmt_toml",
    "fmt_sql",
    # 全量协议（--format-only 时不运行）
    "models_list",
    "chat_streaming",
    "error_format",
    "system_prompt",
    "long_output",
    "latency_p95",
    # 厂商专属（--with-specific 时运行）
    "minimax_ext_fields",
    "glm_web_search",
    "kimi_long_ctx",
    "deepseek_reasoner",
    "qwen_vl",
]

MATRIX_LABELS = {
    "chat_basic":           "基础对话",
    "json_mode":            "JSON 模式",
    "tool_calling":         "工具调用",
    "fmt_json":             "生成 JSON",
    "fmt_jsonl":            "生成 JSONL",
    "fmt_svg":              "SVG 基础",
    "fmt_svg_chart":        "SVG 柱状图",
    "fmt_svg_path":         "SVG Path曲线",
    "fmt_svg_defs":         "SVG 渐变Defs",
    "fmt_xml":              "生成 XML",
    "fmt_csv":              "生成 CSV",
    "fmt_html":             "生成 HTML",
    "fmt_markdown":         "生成 Markdown",
    "fmt_yaml":             "生成 YAML",
    "fmt_toml":             "生成 TOML",
    "fmt_sql":              "生成 SQL",
    "models_list":          "模型列表",
    "chat_streaming":       "流式 SSE",
    "error_format":         "错误格式",
    "system_prompt":        "System Prompt",
    "long_output":          "长输出",
    "latency_p95":          "延迟 p95",
    "minimax_ext_fields":   "Minimax 扩展字段",
    "glm_web_search":       "GLM 联网搜索",
    "kimi_long_ctx":        "KIMI 长上下文",
    "deepseek_reasoner":    "DeepSeek 推理链",
    "qwen_vl":              "Qwen 视觉理解",
}

SECTION_BREAKS = {
    "chat_basic":        "── 格式合规 ──",
    "models_list":       "── 协议检测 ──",
    "minimax_ext_fields":"── 厂商专属 ──",
}

SEV_CELL = {
    Severity.ok:    "✅",
    Severity.warn:  "⚠️",
    Severity.error: "❌",
    Severity.skip:  "⏭️",
}


def render_comparison_matrix(
    reports: dict[str, DiagReport],
    providers: list[ProviderConfig],
) -> None:
    """渲染横向对比矩阵表格。"""
    active = [p for p in providers if p.slug in reports]
    if not active:
        console.print("[red]没有可用的报告数据[/red]")
        return

    table = Table(
        title="[bold cyan]LLM API 综合对比矩阵[/bold cyan]",
        box=box.ROUNDED,
        highlight=True,
        expand=True,
    )
    table.add_column("检测项", style="bold", min_width=18, no_wrap=True)
    for p in active:
        table.add_column(p.name, justify="center", min_width=12)

    # Build check lookup: slug -> {check_name -> CheckResult}
    lookup: dict[str, dict[str, CheckResult]] = {}
    for slug, report in reports.items():
        lookup[slug] = {c.name: c for c in report.checks}

    section_printed: set[str] = set()

    for check_name in MATRIX_CHECKS:
        label = MATRIX_LABELS.get(check_name, check_name)

        # 检查是否有任何厂商包含该检测项
        any_present = any(
            check_name in lookup.get(p.slug, {})
            for p in active
        )
        if not any_present:
            continue

        # 分区标题行
        if check_name in SECTION_BREAKS and check_name not in section_printed:
            section_title = SECTION_BREAKS[check_name]
            section_printed.add(check_name)
            table.add_row(
                f"[dim]{section_title}[/dim]",
                *["" for _ in active],
                style="dim",
            )

        cells: list[str] = []
        for p in active:
            result = lookup.get(p.slug, {}).get(check_name)
            if result is None:
                cells.append("[dim]—[/dim]")
            else:
                icon = SEV_CELL[result.severity]
                # For latency, show the number
                if check_name == "latency_p95" and result.latency_ms:
                    cells.append(f"{icon}\n[dim]{result.latency_ms:.0f}ms[/dim]")
                else:
                    cells.append(icon)

        table.add_row(label, *cells)

    # ── Token 消耗汇总行（追加在矩阵底部） ────────────────────────────────────
    table.add_row(
        "[dim]── Token 统计 ──[/dim]",
        *["" for _ in active],
        style="dim",
    )

    def _fmt_tokens(n: int) -> str:
        """格式化 token 数，超过 1000 用 k 表示。"""
        return f"{n/1000:.1f}k" if n >= 1000 else str(n)

    for tok_label, tok_key in [
        ("Prompt Tokens",     "prompt_tokens"),
        ("Completion Tokens", "completion_tokens"),
        ("Total Tokens",      "total_tokens"),
    ]:
        tok_cells: list[str] = []
        for p in active:
            report = reports.get(p.slug)
            if not report:
                tok_cells.append("[dim]—[/dim]")
                continue
            ts = report.token_summary()
            val = ts.get(tok_key, 0)
            color = "cyan" if tok_key == "total_tokens" else "dim"
            tok_cells.append(f"[{color}]{_fmt_tokens(val)}[/{color}]")
        table.add_row(tok_label, *tok_cells)

    console.print()
    console.print(table)

    # ── 汇总统计表 ────────────────────────────────────────────────────────────
    summary_table = Table(box=box.SIMPLE, expand=False, show_header=True)
    summary_table.add_column("厂商", style="bold")
    summary_table.add_column("模型", style="cyan")
    summary_table.add_column("✅ OK", justify="right", style="green")
    summary_table.add_column("⚠️  WARN", justify="right", style="yellow")
    summary_table.add_column("❌ ERROR", justify="right", style="red")
    summary_table.add_column("Prompt T", justify="right", style="dim")
    summary_table.add_column("Completion T", justify="right", style="dim")
    summary_table.add_column("Total T", justify="right", style="cyan bold")
    summary_table.add_column("总耗时", justify="right")
    summary_table.add_column("评定")

    for p in active:
        report = reports.get(p.slug)
        if not report:
            continue
        s = report.summary()
        ts = report.token_summary()
        passed = report.passed()
        verdict = "[green]通过[/green]" if passed else "[red]未通过[/red]"
        total_ms = sum(c.latency_ms for c in report.checks if c.latency_ms)
        summary_table.add_row(
            p.name,
            report.model,
            str(s["ok"]),
            str(s["warn"]),
            str(s["error"]),
            _fmt_tokens(ts["prompt_tokens"]),
            _fmt_tokens(ts["completion_tokens"]),
            _fmt_tokens(ts["total_tokens"]),
            f"{total_ms / 1000:.1f}s",
            verdict,
        )

    console.print()
    console.print(
        Panel(summary_table, title="[bold]汇总统计（含 Token 消耗）[/bold]", border_style="cyan", expand=False)
    )


def render_provider_banner(cfg: ProviderConfig, api_key_present: bool) -> None:
    key_status = "[green]✓ Key 已配置[/green]" if api_key_present else f"[red]✗ 未找到 {cfg.api_key_env}[/red]"
    console.print(
        Panel(
            f"[bold white]{cfg.name}[/bold white]  {key_status}\n"
            f"[dim]URL: {cfg.base_url}[/dim]\n"
            f"[dim]注: {cfg.notes}[/dim]",
            border_style="blue",
            expand=False,
        )
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

@app.command()
def main(
    only: list[str] = typer.Option(
        [], "--only", "-o",
        help="只测指定厂商 (slug)，可多次: --only glm --only kimi",
    ),
    skip: list[str] = typer.Option(
        [], "--skip", "-s",
        help="跳过指定厂商 (slug)",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="覆盖所有厂商的模型名",
    ),
    skip_formats: bool = typer.Option(
        False, "--skip-formats",
        help="跳过结构化格式生成检测（JSON/SVG/XML 等）",
    ),
    format_only: bool = typer.Option(
        False, "--format-only",
        help="只运行格式合规检测（chat sanity + json_mode + tool_calling + fmt_*），跳过协议类检测",
    ),
    check_delay: float = typer.Option(
        0.0, "--check-delay",
        help="每次检测之间的等待秒数（对限速严格的模型如 glm-5.2 建议 65）",
    ),
    with_specific: bool = typer.Option(
        False, "--with-specific",
        help="同时运行厂商专属检测（GLM联网/KIMI长上下文/Minimax扩展字段/Qwen视觉/DeepSeek推理链）",
    ),
    parallel: bool = typer.Option(
        False, "--parallel",
        help="并发运行所有厂商（注意可能触发限流）",
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output", "-O",
        help="保存每个厂商的 JSON 报告到目录",
    ),
    json_output: bool = typer.Option(
        False, "--json",
        help="输出完整 JSON 结果（stdout）",
    ),
    push_url: Optional[str] = typer.Option(
        None, "--push-url",
        help="FormatProbe Worker 的 /ingest 地址，设置后自动上传结果",
        envvar="FORMATPROBE_PUSH_URL",
    ),
    push_secret: Optional[str] = typer.Option(
        None, "--push-secret",
        help="FormatProbe Worker 的 INGEST_SECRET",
        envvar="FORMATPROBE_PUSH_SECRET",
    ),
    no_detail: bool = typer.Option(
        False, "--no-detail",
        help="不显示每个厂商的逐项结果，只显示最终对比矩阵",
    ),
    env_file: str = typer.Option(
        ".env", "--env",
        help=".env 文件路径",
    ),
):
    """
    批量诊断国内五家 LLM API：智谱 GLM / KIMI / Minimax / Qwen / DeepSeek

    首次使用请先配置 API Key：

      cp .env.template .env && vim .env

    常用示例：

      python3 run_providers.py                          # 全部厂商（跳过厂商专属）
      python3 run_providers.py --only glm kimi          # 仅 GLM 和 KIMI
      python3 run_providers.py --skip-formats           # 跳过格式测试（快速模式）
      python3 run_providers.py --with-specific          # 含厂商专属检测
      python3 run_providers.py --output reports/        # 保存报告
    """
    # 重新加载指定 .env
    load_dotenv(Path(env_file))

    # 筛选厂商
    target_providers: list[ProviderConfig]
    if only:
        unknown = [s for s in only if s not in PROVIDER_MAP]
        if unknown:
            console.print(f"[red]未知厂商 slug: {unknown}，可用: {list(PROVIDER_MAP.keys())}[/red]")
            raise typer.Exit(1)
        target_providers = [PROVIDER_MAP[s] for s in only]
    else:
        target_providers = [p for p in PROVIDERS if p.slug not in skip]

    if not target_providers:
        console.print("[yellow]没有选中任何厂商[/yellow]")
        raise typer.Exit(0)

    # 创建输出目录
    out_dir: Path | None = None
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    console.print()
    mode_label = "仅格式合规" if format_only else ("跳过格式" if skip_formats else "全量检测")
    console.print(Panel(
        f"[bold cyan]LLM API 批量诊断[/bold cyan]\n"
        f"厂商: {' / '.join(p.name for p in target_providers)}\n"
        f"模式: {mode_label} | "
        f"厂商专属: {'开启' if with_specific else '关闭'} | "
        f"并发: {'是' if parallel else '否'} | "
        + (f"间隔: {check_delay:.0f}s | " if check_delay > 0 else "")
        + f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        border_style="cyan",
        expand=False,
    ))

    reports: dict[str, DiagReport] = {}

    async def run_one(cfg: ProviderConfig) -> tuple[str, DiagReport]:
        key_present = bool(os.environ.get(cfg.api_key_env))
        if not no_detail:
            render_provider_banner(cfg, key_present)
        report = await diagnose_provider(
            cfg, model,
            skip_formats=skip_formats,
            skip_specific=not with_specific,
            check_delay=check_delay,
            format_only=format_only,
        )
        if not no_detail:
            render_report(report)
        if push_url and push_secret:
            await _push_report(cfg, report, push_url, push_secret)
        return cfg.slug, report

    async def run_all_serial() -> None:
        for cfg in target_providers:
            slug, report = await run_one(cfg)
            reports[slug] = report
            if out_dir:
                _save_report(report, cfg, out_dir)

    async def run_all_parallel() -> None:
        tasks = [run_one(cfg) for cfg in target_providers]
        for coro in asyncio.as_completed(tasks):
            slug, report = await coro
            reports[slug] = report
            if out_dir:
                _save_report(report, cfg := next(p for p in target_providers if p.slug == slug), out_dir)

    t_start = time.monotonic()
    if parallel:
        asyncio.run(run_all_parallel())
    else:
        asyncio.run(run_all_serial())
    elapsed = time.monotonic() - t_start

    console.print(f"\n[dim]全部检测完成，总耗时 {elapsed:.1f}s[/dim]")

    # 对比矩阵
    render_comparison_matrix(reports, target_providers)

    # JSON 输出
    if json_output:
        all_data = {
            slug: _report_to_dict(report)
            for slug, report in reports.items()
        }
        console.print_json(json.dumps(all_data, ensure_ascii=False, indent=2))

    # 退出码：任何厂商 ERROR 则非零
    if any(not r.passed() for r in reports.values()):
        raise typer.Exit(1)


def _report_to_dict(report: DiagReport) -> dict:
    return {
        "target_url": report.target_url,
        "protocol": report.protocol,
        "model": report.model,
        "timestamp": report.timestamp,
        "summary": report.summary(),
        "token_summary": report.token_summary(),
        "passed": report.passed(),
        "checks": [
            {
                "name": c.name,
                "severity": c.severity.value,
                "message": c.message,
                "detail": c.detail,
                "latency_ms": round(c.latency_ms, 1),
                "extra": c.extra,
            }
            for c in report.checks
        ],
    }


async def _push_report(cfg: ProviderConfig, report: DiagReport, push_url: str, secret: str) -> None:
    """POST results to FormatProbe Worker /ingest."""
    payload = {
        "provider": cfg.name,
        "slug": cfg.slug,
        "model": report.model,
        "run_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checks": [
            {
                "name": c.name,
                "severity": c.severity.value,
                "message": c.message,
                "detail": c.detail[:500] if c.detail else None,
                "latency_ms": round(c.latency_ms, 1) if c.latency_ms else None,
            }
            for c in report.checks
        ],
    }
    url = push_url.rstrip("/") + "/ingest"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(
                url, json=payload,
                headers={"Authorization": f"Bearer {secret}"},
            )
        if r.status_code == 200:
            console.print(f"[dim]  ↑ 已推送 {len(report.checks)} 项结果到 FormatProbe[/dim]")
        else:
            console.print(f"[yellow]  ↑ 推送失败 HTTP {r.status_code}: {r.text[:80]}[/yellow]")
    except Exception as e:
        console.print(f"[yellow]  ↑ 推送异常: {e}[/yellow]")


def _save_report(report: DiagReport, cfg: ProviderConfig, out_dir: Path) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = out_dir / f"{ts}_{cfg.slug}_{report.model.replace('/', '-')}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(_report_to_dict(report), f, ensure_ascii=False, indent=2)
    console.print(f"[dim]  报告已保存: {fname}[/dim]")


if __name__ == "__main__":
    app()
