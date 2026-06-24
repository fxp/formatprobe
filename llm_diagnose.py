#!/usr/bin/env python3
"""
LLM API Health Diagnostics Tool
Supports: OpenAI-compatible, Anthropic, and custom APIs
"""
from __future__ import annotations

import asyncio
import json
import time
import sys
import os
import re
import csv
import io
import html as html_lib
import html.parser as _html_parser
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator
from datetime import datetime

# Optional deps — degrade gracefully
try:
    import yaml as _yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import tomllib as _tomllib          # Python 3.11+
    HAS_TOML = True
except ImportError:
    try:
        import tomli as _tomllib        # pip install tomli
        HAS_TOML = True
    except ImportError:
        HAS_TOML = False

import httpx
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
from rich.text import Text
from rich.live import Live
from rich.layout import Layout

app = typer.Typer(
    name="llm-diagnose",
    help="LLM API Health Diagnostics — checks protocol compliance, latency, streaming, JSON validity, and more.",
    rich_markup_mode="rich",
)
import sys as _sys
console = Console(
    force_terminal=True,
    no_color=not _sys.stdout.isatty(),   # plain text when not in a real terminal
)


# ── Enums & Dataclasses ────────────────────────────────────────────────────────

class Protocol(str, Enum):
    openai = "openai"
    anthropic = "anthropic"
    auto = "auto"


class Severity(str, Enum):
    ok = "ok"
    warn = "warn"
    error = "error"
    skip = "skip"


@dataclass
class CheckResult:
    name: str
    severity: Severity
    message: str
    detail: str = ""
    latency_ms: float = 0.0
    extra: dict = field(default_factory=dict)


@dataclass
class DiagReport:
    target_url: str
    protocol: str
    model: str
    timestamp: str
    checks: list[CheckResult] = field(default_factory=list)

    def summary(self) -> dict[str, int]:
        counts = {s.value: 0 for s in Severity}
        for c in self.checks:
            counts[c.severity.value] += 1
        return counts

    def passed(self) -> bool:
        return all(c.severity != Severity.error for c in self.checks)

    def token_summary(self) -> dict[str, int]:
        """Aggregate token usage across all checks that recorded it."""
        total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for c in self.checks:
            u = c.extra.get("usage") or {}
            for k in total:
                total[k] += int(u.get(k) or 0)
        return total


# ── Token usage helpers ────────────────────────────────────────────────────────

def _extract_usage(data: dict) -> dict:
    """Extract {prompt_tokens, completion_tokens, total_tokens} from a response dict.

    Handles two layouts:
    - Standard (OpenAI / Qwen / Minimax): top-level ``data["usage"]``
    - KIMI / Moonshot streaming: usage nested in ``data["choices"][0]["usage"]``
    """
    # 1. Top-level usage (most providers)
    u = data.get("usage")
    # 2. KIMI streams usage inside choices[0].usage
    if not u:
        choices = data.get("choices") or []
        if choices and isinstance(choices[0], dict):
            u = choices[0].get("usage")
    if not u or not isinstance(u, dict):
        return {}
    return {
        "prompt_tokens":     int(u.get("prompt_tokens")     or 0),
        "completion_tokens": int(u.get("completion_tokens") or 0),
        "total_tokens":      int(u.get("total_tokens")      or 0),
    }


def _supports_stream_options(base_url: str) -> bool:
    """Return True if the provider accepts stream_options:{include_usage:true}.

    Minimax's OpenAI-compatible API does not expose this parameter and may
    reject or silently misbehave when it is included.
    """
    return "minimaxi.com" not in base_url and "minimax.com" not in base_url


def _get_content(msg: dict) -> str:
    """Extract text content from a chat message dict.

    Thinking/reasoning models (GLM-5.x, Kimi-K2.x, DeepSeek-R1 …) place the
    visible answer in ``content`` and the internal chain-of-thought in
    ``reasoning_content``.  When ``content`` is empty we fall back to
    ``reasoning_content`` so protocol compliance checks still work.
    """
    content = msg.get("content") or ""
    if not content:
        content = msg.get("reasoning_content") or ""
    return content


def _api_url(base_url: str, path: str) -> str:
    """
    Construct a full API URL without double-versioning.

    Many providers include the version in the base URL already
    (e.g. `.../v1`, `.../v4`).  If the base already ends with `/vN`,
    we strip the leading `/v1` from `path` before joining, so:

        base=".../v4"  + path="/v1/chat/completions"
        → ".../v4/chat/completions"          ✓  (GLM)

        base=".../v1"  + path="/v1/chat/completions"
        → ".../v1/chat/completions"          ✓  (KIMI, Minimax, Qwen)

        base="...com"  + path="/v1/chat/completions"
        → "...com/v1/chat/completions"       ✓  (DeepSeek, OpenAI)
    """
    base = base_url.rstrip("/")
    if re.search(r"/v\d+$", base):
        # base already versioned — drop /v1 prefix from path if present
        clean = re.sub(r"^/v\d+", "", path.lstrip("/") if not path.startswith("/v") else path)
        clean = clean.lstrip("/")
        return f"{base}/{clean}"
    return f"{base}/{path.lstrip('/')}"


# ── HTTP helpers ───────────────────────────────────────────────────────────────

TIMEOUT_CONNECT = 10.0
TIMEOUT_READ = 60.0


def make_client(api_key: str | None, extra_headers: dict) -> httpx.AsyncClient:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    headers.update(extra_headers)
    return httpx.AsyncClient(
        headers=headers,
        timeout=httpx.Timeout(TIMEOUT_READ, connect=TIMEOUT_CONNECT),
        follow_redirects=True,
    )


def make_anthropic_client(api_key: str | None, extra_headers: dict) -> httpx.AsyncClient:
    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    if api_key:
        headers["x-api-key"] = api_key
    headers.update(extra_headers)
    return httpx.AsyncClient(
        headers=headers,
        timeout=httpx.Timeout(TIMEOUT_READ, connect=TIMEOUT_CONNECT),
        follow_redirects=True,
    )


# ── Protocol detection ─────────────────────────────────────────────────────────

async def detect_protocol(base_url: str, api_key: str | None) -> str:
    """Best-effort protocol detection by probing known endpoints."""
    async with make_client(api_key, {}) as client:
        for path in ["/v1/models", "/models"]:
            try:
                r = await client.get(base_url.rstrip("/") + path, timeout=8)
                if r.status_code < 500:
                    try:
                        data = r.json()
                        if "data" in data or "models" in data or "object" in data:
                            return "openai"
                    except Exception:
                        pass
            except Exception:
                pass

    async with make_anthropic_client(api_key, {}) as client:
        try:
            r = await client.get(_api_url(base_url, "/v1/models"), timeout=8)
            if r.status_code < 500:
                try:
                    data = r.json()
                    if "data" in data:
                        return "anthropic"
                except Exception:
                    pass
        except Exception:
            pass

    return "openai"  # default fallback


# ── OpenAI checks ──────────────────────────────────────────────────────────────

async def check_openai_models(base_url: str, client: httpx.AsyncClient) -> CheckResult:
    """GET /v1/models — list models endpoint."""
    url = _api_url(base_url, "/v1/models")
    t0 = time.monotonic()
    try:
        r = await client.get(url)
        latency = (time.monotonic() - t0) * 1000
        if r.status_code == 200:
            try:
                data = r.json()
                if "data" in data and isinstance(data["data"], list):
                    model_ids = [m.get("id", "?") for m in data["data"][:5]]
                    return CheckResult(
                        "models_list", Severity.ok,
                        f"返回 {len(data['data'])} 个模型",
                        f"前5: {model_ids}",
                        latency,
                    )
                else:
                    return CheckResult(
                        "models_list", Severity.warn,
                        "响应缺少 'data' 字段",
                        f"keys: {list(data.keys())}",
                        latency,
                    )
            except Exception as e:
                return CheckResult("models_list", Severity.error, "响应非 JSON", str(e), latency)
        elif r.status_code == 404:
            return CheckResult("models_list", Severity.warn, "404 — /v1/models 不存在", "", latency)
        elif r.status_code == 401:
            return CheckResult("models_list", Severity.error, "401 认证失败", r.text[:200], latency)
        else:
            return CheckResult("models_list", Severity.warn, f"HTTP {r.status_code}", r.text[:200], latency)
    except httpx.ConnectError as e:
        return CheckResult("models_list", Severity.error, "连接失败", str(e))
    except httpx.TimeoutException:
        return CheckResult("models_list", Severity.error, f"超时 (>{TIMEOUT_CONNECT}s)", "")


async def check_openai_chat(
    base_url: str, client: httpx.AsyncClient, model: str
) -> CheckResult:
    """POST /v1/chat/completions — basic non-streaming chat."""
    url = _api_url(base_url, "/v1/chat/completions")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: PONG"}],
        "max_tokens": 512,
        "stream": False,
    }
    t0 = time.monotonic()
    try:
        r = await client.post(url, json=payload)
        latency = (time.monotonic() - t0) * 1000
        if r.status_code == 200:
            try:
                data = r.json()
            except Exception as e:
                return CheckResult("chat_basic", Severity.error, "响应非 JSON", str(e), latency)

            issues = []
            required = ["id", "object", "created", "model", "choices", "usage"]
            missing = [k for k in required if k not in data]
            if missing:
                issues.append(f"缺少字段: {missing}")

            if "choices" in data:
                choices = data["choices"]
                if not choices:
                    issues.append("choices 为空列表")
                else:
                    c = choices[0]
                    if "message" not in c:
                        issues.append("choices[0] 缺少 message")
                    if c.get("finish_reason") not in ("stop", "length", "content_filter", "tool_calls", None):
                        issues.append(f"未知 finish_reason: {c.get('finish_reason')}")
                    content = _get_content(c.get("message", {}))
                    if not content:
                        issues.append("content 为空")

            if "usage" in data:
                u = data["usage"]
                for f_ in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                    if f_ not in u:
                        issues.append(f"usage 缺少 {f_}")

            if issues:
                return CheckResult(
                    "chat_basic", Severity.warn,
                    "协议合规警告",
                    "; ".join(issues),
                    latency,
                    {"content": _get_content(data.get("choices", [{}])[0].get("message", {}))[:100], "usage": _extract_usage(data)},
                )
            content = _get_content(data["choices"][0]["message"])
            return CheckResult(
                "chat_basic", Severity.ok,
                "非流式 chat 响应正常",
                f"content: {content[:80]}",
                latency,
                {"usage": _extract_usage(data)},
            )
        elif r.status_code == 401:
            return CheckResult("chat_basic", Severity.error, "401 认证失败", r.text[:200], latency)
        elif r.status_code == 422:
            return CheckResult("chat_basic", Severity.error, "422 参数错误", r.text[:300], latency)
        else:
            return CheckResult("chat_basic", Severity.error, f"HTTP {r.status_code}", r.text[:200], latency)
    except httpx.TimeoutException:
        return CheckResult("chat_basic", Severity.error, f"请求超时 (>{TIMEOUT_READ}s)", "")
    except Exception as e:
        return CheckResult("chat_basic", Severity.error, f"请求异常: {type(e).__name__}", str(e))


async def check_openai_streaming(
    base_url: str, client: httpx.AsyncClient, model: str
) -> CheckResult:
    """POST /v1/chat/completions stream=true — SSE streaming compliance."""
    url = _api_url(base_url, "/v1/chat/completions")
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": "Count 1 to 5, one per line."}],
        "max_tokens": 512,
        "stream": True,
    }
    if _supports_stream_options(base_url):
        payload["stream_options"] = {"include_usage": True}

    chunks: list[dict] = []
    content_parts: list[str] = []
    parse_errors: list[str] = []
    first_chunk_latency: float = 0.0
    total_latency: float = 0.0
    silence_gaps: list[float] = []  # gaps > 5s between chunks
    last_chunk_time: float = 0.0
    stream_usage: dict = {}

    t0 = time.monotonic()
    try:
        async with client.stream("POST", url, json=payload) as r:
            if r.status_code != 200:
                text = await r.aread()
                return CheckResult(
                    "chat_streaming", Severity.error,
                    f"流式请求失败 HTTP {r.status_code}",
                    text.decode()[:200],
                    (time.monotonic() - t0) * 1000,
                )

            async for line in r.aiter_lines():
                now = time.monotonic()
                if last_chunk_time and (now - last_chunk_time) > 5.0:
                    silence_gaps.append(round((now - last_chunk_time) * 1000))
                last_chunk_time = now

                if not first_chunk_latency:
                    first_chunk_latency = (now - t0) * 1000

                if not line:
                    continue
                if not line.startswith("data:"):
                    if line.startswith(":"):
                        continue  # SSE comment / heartbeat
                    parse_errors.append(f"非data行: {line[:60]}")
                    continue

                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    # Usage-only chunk: choices=[] + usage field (stream_options / Qwen)
                    # Must be handled before delta extraction to avoid [][0] IndexError
                    if not chunk.get("choices") and chunk.get("usage"):
                        stream_usage = _extract_usage(chunk)
                    else:
                        chunks.append(chunk)
                        choices = chunk.get("choices") or []
                        if choices:
                            d = choices[0].get("delta", {})
                            # Reasoning models put thinking in reasoning_content; fall back to it
                            delta = d.get("content") or d.get("reasoning_content") or ""
                        else:
                            delta = ""
                        if delta:
                            content_parts.append(delta)
                        # KIMI: usage is embedded inside choices[0].usage on the final chunk
                        if choices and not stream_usage:
                            u = _extract_usage(chunk)
                            if u.get("total_tokens"):
                                stream_usage = u
                except json.JSONDecodeError as e:
                    parse_errors.append(f"JSON解析失败: {data_str[:60]} — {e}")

        total_latency = (time.monotonic() - t0) * 1000
    except httpx.TimeoutException:
        return CheckResult(
            "chat_streaming", Severity.error,
            f"流式请求超时 (>{TIMEOUT_READ}s)",
            f"已收到 {len(chunks)} chunks",
            (time.monotonic() - t0) * 1000,
        )
    except Exception as e:
        return CheckResult(
            "chat_streaming", Severity.error,
            f"流式请求异常: {type(e).__name__}", str(e),
            (time.monotonic() - t0) * 1000,
        )

    full_content = "".join(content_parts)
    issues = []
    if parse_errors:
        issues.append(f"SSE解析错误 x{len(parse_errors)}: {parse_errors[0]}")
    if not chunks:
        issues.append("没有收到任何 chunk")
    if not full_content:
        issues.append("拼接内容为空")
    if silence_gaps:
        issues.append(f"流中断 {len(silence_gaps)} 次，最大间隔 {max(silence_gaps)}ms")

    # Check first chunk structure
    if chunks:
        fc = chunks[0]
        for fld in ["id", "object", "created", "model", "choices"]:
            if fld not in fc:
                issues.append(f"首个chunk缺少字段: {fld}")
        if fc.get("object") != "chat.completion.chunk":
            issues.append(f"object 应为 chat.completion.chunk，实际: {fc.get('object')}")

    severity = Severity.error if not chunks else (Severity.warn if issues else Severity.ok)
    detail_parts = [f"chunks={len(chunks)}", f"首包={first_chunk_latency:.0f}ms", f"总耗时={total_latency:.0f}ms"]
    if silence_gaps:
        detail_parts.append(f"停顿间隔={silence_gaps}")
    if full_content:
        detail_parts.append(f"content='{full_content[:60]}'")

    return CheckResult(
        "chat_streaming",
        severity,
        "流式输出正常" if not issues else f"流式问题: {issues[0]}",
        " | ".join(detail_parts),
        total_latency,
        {"parse_errors": parse_errors, "silence_gaps": silence_gaps, "usage": stream_usage},
    )


async def check_openai_json_mode(
    base_url: str, client: httpx.AsyncClient, model: str
) -> CheckResult:
    """Test response_format: json_object compliance."""
    url = _api_url(base_url, "/v1/chat/completions")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": 'Return a JSON object with keys "status" (value "ok") and "count" (value 42). Only output JSON.',
            }
        ],
        "max_tokens": 1024,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    t0 = time.monotonic()
    try:
        r = await client.post(url, json=payload)
        latency = (time.monotonic() - t0) * 1000
        if r.status_code == 200:
            try:
                data = r.json()
                content = _get_content(data.get("choices", [{}])[0].get("message", {}))
                try:
                    parsed = json.loads(content)
                    return CheckResult(
                        "json_mode", Severity.ok,
                        "JSON mode 有效",
                        f"content={content[:80]}",
                        latency,
                        {"usage": _extract_usage(data)},
                    )
                except json.JSONDecodeError:
                    return CheckResult(
                        "json_mode", Severity.error,
                        "JSON mode 返回非 JSON",
                        f"content={content[:100]}",
                        latency,
                    )
            except Exception as e:
                return CheckResult("json_mode", Severity.error, "响应解析失败", str(e), latency)
        elif r.status_code in (400, 422):
            # Model may not support json_object
            return CheckResult(
                "json_mode", Severity.warn,
                f"JSON mode 不支持 (HTTP {r.status_code})",
                r.text[:200],
                latency,
            )
        else:
            return CheckResult("json_mode", Severity.warn, f"HTTP {r.status_code}", r.text[:100], latency)
    except httpx.TimeoutException:
        return CheckResult("json_mode", Severity.error, "超时", "")
    except Exception as e:
        return CheckResult("json_mode", Severity.error, str(e), "")


async def check_openai_error_format(
    base_url: str, client: httpx.AsyncClient, model: str
) -> CheckResult:
    """Verify error responses follow OpenAI error schema."""
    url = _api_url(base_url, "/v1/chat/completions")
    payload = {
        "model": "__invalid_model_xyz_9999__",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
    }
    t0 = time.monotonic()
    try:
        r = await client.post(url, json=payload)
        latency = (time.monotonic() - t0) * 1000
        if r.status_code == 200:
            return CheckResult(
                "error_format", Severity.warn,
                "无效 model 仍返回 200",
                "预期应返回 4xx",
                latency,
            )
        try:
            data = r.json()
            if "error" in data:
                err = data["error"]
                has_message = "message" in err
                has_type = "type" in err
                if has_message and has_type:
                    return CheckResult(
                        "error_format", Severity.ok,
                        "错误格式符合 OpenAI 规范",
                        f"type={err.get('type')} msg={err.get('message','')[:60]}",
                        latency,
                    )
                else:
                    missing = [f for f in ["message", "type"] if f not in err]
                    return CheckResult(
                        "error_format", Severity.warn,
                        "error 对象缺少标准字段",
                        f"缺少: {missing}",
                        latency,
                    )
            else:
                return CheckResult(
                    "error_format", Severity.warn,
                    "错误响应缺少 'error' 字段",
                    f"keys: {list(data.keys())}",
                    latency,
                )
        except Exception:
            return CheckResult(
                "error_format", Severity.warn,
                "错误响应非 JSON",
                r.text[:100],
                latency,
            )
    except httpx.TimeoutException:
        return CheckResult("error_format", Severity.error, "超时", "")
    except Exception as e:
        return CheckResult("error_format", Severity.error, str(e), "")


async def check_openai_latency_p95(
    base_url: str, client: httpx.AsyncClient, model: str, n: int = 5
) -> CheckResult:
    """Run N quick completions, report p50/p95 TTFB."""
    url = _api_url(base_url, "/v1/chat/completions")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say OK"}],
        "max_tokens": 4,
        "stream": True,
    }
    ttfbs: list[float] = []

    for _ in range(n):
        t0 = time.monotonic()
        try:
            async with client.stream("POST", url, json=payload) as r:
                if r.status_code != 200:
                    break
                async for line in r.aiter_lines():
                    if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]"):
                        ttfbs.append((time.monotonic() - t0) * 1000)
                        break
        except Exception:
            break
        await asyncio.sleep(0.2)

    if not ttfbs:
        return CheckResult("latency_p95", Severity.warn, "无法完成延迟测量", "")

    ttfbs.sort()
    p50 = ttfbs[len(ttfbs) // 2]
    p95 = ttfbs[int(len(ttfbs) * 0.95)] if len(ttfbs) >= 5 else ttfbs[-1]
    avg = sum(ttfbs) / len(ttfbs)

    severity = Severity.ok
    if p95 > 10000:
        severity = Severity.error
    elif p95 > 4000:
        severity = Severity.warn

    return CheckResult(
        "latency_p95",
        severity,
        f"TTFB avg={avg:.0f}ms p50={p50:.0f}ms p95={p95:.0f}ms",
        f"样本数={len(ttfbs)} 原始={[round(t) for t in ttfbs]}",
        avg,
    )


async def check_openai_tool_calling(
    base_url: str, client: httpx.AsyncClient, model: str
) -> CheckResult:
    """Check function/tool calling support."""
    url = _api_url(base_url, "/v1/chat/completions")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "What is the weather in Beijing?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"}
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "max_tokens": 512,
        "stream": False,
    }
    t0 = time.monotonic()
    try:
        r = await client.post(url, json=payload)
        latency = (time.monotonic() - t0) * 1000
        if r.status_code == 200:
            try:
                data = r.json()
                choice = data.get("choices", [{}])[0]
                finish = choice.get("finish_reason")
                tool_calls = choice.get("message", {}).get("tool_calls")
                if finish == "tool_calls" and tool_calls:
                    tc = tool_calls[0]
                    fn_name = tc.get("function", {}).get("name", "")
                    args_str = tc.get("function", {}).get("arguments", "{}")
                    try:
                        args = json.loads(args_str)
                    except Exception:
                        return CheckResult(
                            "tool_calling", Severity.warn,
                            "tool_calls arguments 非有效 JSON",
                            args_str[:100],
                            latency,
                        )
                    return CheckResult(
                        "tool_calling", Severity.ok,
                        f"Tool calling 正常: {fn_name}({args})",
                        "",
                        latency,
                        {"usage": _extract_usage(data)},
                    )
                elif finish == "stop":
                    content = choice.get("message", {}).get("content", "")
                    return CheckResult(
                        "tool_calling", Severity.warn,
                        "模型未触发 tool_call，直接回复文本",
                        f"content: {content[:80]}",
                        latency,
                        {"usage": _extract_usage(data)},
                    )
                else:
                    return CheckResult(
                        "tool_calling", Severity.warn,
                        f"finish_reason={finish}，tool_calls={bool(tool_calls)}",
                        "",
                        latency,
                    )
            except Exception as e:
                return CheckResult("tool_calling", Severity.error, "响应解析失败", str(e), latency)
        elif r.status_code in (400, 422):
            return CheckResult(
                "tool_calling", Severity.warn,
                f"Tool calling 不支持 (HTTP {r.status_code})",
                r.text[:200],
                latency,
            )
        else:
            return CheckResult("tool_calling", Severity.warn, f"HTTP {r.status_code}", r.text[:100], latency)
    except httpx.TimeoutException:
        return CheckResult("tool_calling", Severity.error, "超时", "")
    except Exception as e:
        return CheckResult("tool_calling", Severity.error, str(e), "")


async def check_openai_long_output(
    base_url: str, client: httpx.AsyncClient, model: str, min_tokens: int = 400
) -> CheckResult:
    """Test long output doesn't prematurely truncate mid-stream."""
    url = _api_url(base_url, "/v1/chat/completions")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a detailed 500-word essay on the history of the internet. "
                    "Do not stop until you have written at least 500 words."
                ),
            }
        ],
        "max_tokens": 2000,
        "stream": True,
    }
    if _supports_stream_options(base_url):
        payload["stream_options"] = {"include_usage": True}
    chunks: list[str] = []
    t0 = time.monotonic()
    last_chunk_time: float = 0.0
    silence_gaps: list[float] = []
    premature_stop = False
    stream_usage: dict = {}

    try:
        async with client.stream("POST", url, json=payload, timeout=httpx.Timeout(120, connect=TIMEOUT_CONNECT)) as r:
            if r.status_code != 200:
                body = await r.aread()
                return CheckResult(
                    "long_output", Severity.warn,
                    f"HTTP {r.status_code}",
                    body.decode()[:200],
                    (time.monotonic() - t0) * 1000,
                )
            finish_reason = None
            async for line in r.aiter_lines():
                now = time.monotonic()
                if last_chunk_time and (now - last_chunk_time) > 8.0:
                    silence_gaps.append(round((now - last_chunk_time) * 1000))
                last_chunk_time = now
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    if not chunk.get("choices") and chunk.get("usage"):
                        stream_usage = _extract_usage(chunk)
                    else:
                        choices = chunk.get("choices") or []
                        choice = choices[0] if choices else {}
                        d = choice.get("delta", {})
                        delta = d.get("content") or d.get("reasoning_content") or ""
                        if delta:
                            chunks.append(delta)
                        fr = choice.get("finish_reason")
                        if fr:
                            finish_reason = fr
                        # KIMI: usage embedded in choices[0].usage on final chunk
                        if choices and not stream_usage:
                            u = _extract_usage(chunk)
                            if u.get("total_tokens"):
                                stream_usage = u
                except Exception:
                    pass
    except httpx.TimeoutException:
        return CheckResult("long_output", Severity.error, "长输出超时 (>120s)", f"已收 {len(chunks)} chunks")
    except Exception as e:
        return CheckResult("long_output", Severity.error, str(e), "")

    full = "".join(chunks)
    word_count = len(full.split())
    total_latency = (time.monotonic() - t0) * 1000

    issues = []
    if word_count < min_tokens // 2:
        issues.append(f"输出过短: {word_count} 词 (期望 ≥{min_tokens // 2})")
    if silence_gaps:
        issues.append(f"输出中断 {len(silence_gaps)} 次，最大 {max(silence_gaps)}ms")
    if finish_reason == "length":
        issues.append("因 max_tokens 截断")

    severity = Severity.error if word_count < 20 else (Severity.warn if issues else Severity.ok)
    return CheckResult(
        "long_output",
        severity,
        f"长输出完成: {word_count} 词 finish={finish_reason}" if not issues else f"长输出问题: {issues[0]}",
        f"total={total_latency:.0f}ms silence_gaps={silence_gaps}",
        total_latency,
        {"usage": stream_usage},
    )


async def check_openai_system_prompt(
    base_url: str, client: httpx.AsyncClient, model: str
) -> CheckResult:
    """Verify system prompt is honored."""
    url = _api_url(base_url, "/v1/chat/completions")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You only speak Pig Latin. Never use English."},
            {"role": "user", "content": "Say hello."},
        ],
        "max_tokens": 512,
        "stream": False,
    }
    t0 = time.monotonic()
    try:
        r = await client.post(url, json=payload)
        latency = (time.monotonic() - t0) * 1000
        if r.status_code == 200:
            try:
                data = r.json()
                content = _get_content(data.get("choices", [{}])[0].get("message", {}))
                # Very rough check: if model echoes "hello" verbatim it likely ignored the system prompt
                if content.lower().strip() in ("hello", "hello!", "hello."):
                    return CheckResult(
                        "system_prompt", Severity.warn,
                        "System prompt 可能被忽略",
                        f"content: {content}",
                        latency,
                        {"usage": _extract_usage(data)},
                    )
                return CheckResult(
                    "system_prompt", Severity.ok,
                    "System prompt 生效",
                    f"content: {content[:80]}",
                    latency,
                    {"usage": _extract_usage(data)},
                )
            except Exception as e:
                return CheckResult("system_prompt", Severity.error, "解析失败", str(e), latency)
        else:
            return CheckResult("system_prompt", Severity.warn, f"HTTP {r.status_code}", "", latency)
    except httpx.TimeoutException:
        return CheckResult("system_prompt", Severity.error, "超时", "")
    except Exception as e:
        return CheckResult("system_prompt", Severity.error, str(e), "")


# ── Structured Format Generation Checks ───────────────────────────────────────
#
#  Each check asks the model to generate a specific structured format,
#  then validates it with the appropriate parser / rules.
#  A shared helper handles the HTTP round-trip + code-fence stripping.
# ──────────────────────────────────────────────────────────────────────────────

async def _ask_format(
    base_url: str,
    client: httpx.AsyncClient,
    model: str,
    prompt: str,
    max_tokens: int = 2048,
) -> tuple[str, float, str | None, dict]:
    """
    Send a chat request and return (raw_content, latency_ms, error_or_None, usage_dict).
    Strips markdown code fences automatically.
    """
    url = _api_url(base_url, "/v1/chat/completions")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    t0 = time.monotonic()
    try:
        r = await client.post(url, json=payload)
        latency = (time.monotonic() - t0) * 1000
        if r.status_code != 200:
            return "", latency, f"HTTP {r.status_code}: {r.text[:120]}", {}
        data = r.json()
        content: str = _get_content(data.get("choices", [{}])[0].get("message", {}))
        # Strip markdown code fences (```lang\n...\n```)
        content = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", content.strip())
        content = re.sub(r"\n?```$", "", content.strip())
        return content.strip(), latency, None, _extract_usage(data)
    except httpx.TimeoutException:
        return "", (time.monotonic() - t0) * 1000, f"超时 (>{TIMEOUT_READ}s)", {}
    except Exception as e:
        return "", (time.monotonic() - t0) * 1000, f"{type(e).__name__}: {e}", {}


# ── JSON ───────────────────────────────────────────────────────────────────────

async def check_format_json(
    base_url: str, client: httpx.AsyncClient, model: str
) -> CheckResult:
    """Generate a nested JSON object and validate structure + types."""
    prompt = (
        "Output ONLY valid JSON (no markdown fences, no explanation). "
        "The JSON must contain:\n"
        '  - "name": a string\n'
        '  - "age": an integer\n'
        '  - "scores": an array of at least 3 numbers\n'
        '  - "address": an object with "city" (string) and "zip" (string)\n'
        '  - "active": a boolean\n'
        "Example: {\"name\":\"Alice\",\"age\":30,...}"
    )
    content, latency, err, _usage = await _ask_format(base_url, client, model, prompt, 256)
    if err:
        return CheckResult("fmt_json", Severity.error, err, "", latency)
    if not content:
        return CheckResult("fmt_json", Severity.error, "返回空内容", "", latency)

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        snippet = content[:120].replace("\n", "↵")
        return CheckResult(
            "fmt_json", Severity.error,
            f"JSON 解析失败: {e}",
            f"内容片段: {snippet}", latency,
        )

    issues: list[str] = []
    if not isinstance(data, dict):
        issues.append(f"顶层应为 object，实际: {type(data).__name__}")
    else:
        if not isinstance(data.get("name"), str):
            issues.append("name 应为 string")
        if not isinstance(data.get("age"), int) or isinstance(data.get("age"), bool):
            issues.append("age 应为 integer")
        scores = data.get("scores")
        if not isinstance(scores, list) or len(scores) < 3:
            issues.append(f"scores 应为 ≥3 元素的 array，实际: {scores!r}")
        elif not all(isinstance(s, (int, float)) for s in scores):
            issues.append("scores 元素应全为数字")
        addr = data.get("address")
        if not isinstance(addr, dict):
            issues.append("address 应为 object")
        else:
            if not isinstance(addr.get("city"), str):
                issues.append("address.city 应为 string")
            if not isinstance(addr.get("zip"), str):
                issues.append("address.zip 应为 string")
        if not isinstance(data.get("active"), bool):
            issues.append("active 应为 boolean")

    if issues:
        return CheckResult(
            "fmt_json", Severity.warn,
            f"JSON 结构问题 ({len(issues)}项)",
            "; ".join(issues), latency,
        )
    return CheckResult(
        "fmt_json", Severity.ok,
        "JSON 生成与验证通过",
        f"keys={list(data.keys())} scores={data.get('scores')}", latency,
        {"usage": _usage},
    )


# ── JSON Lines (JSONL) ─────────────────────────────────────────────────────────

async def check_format_jsonl(
    base_url: str, client: httpx.AsyncClient, model: str
) -> CheckResult:
    """Generate JSON Lines (newline-delimited JSON) and validate each row."""
    prompt = (
        "Output exactly 5 lines of JSON Lines (JSONL) format — one valid JSON object per line, no blank lines, "
        "no markdown fences. Each object must have: "
        '"id" (integer), "product" (string), "price" (number), "in_stock" (boolean). '
        "No extra text before or after."
    )
    content, latency, err, _usage = await _ask_format(base_url, client, model, prompt, 300)
    if err:
        return CheckResult("fmt_jsonl", Severity.error, err, "", latency)
    if not content:
        return CheckResult("fmt_jsonl", Severity.error, "返回空内容", "", latency)

    lines = [l.strip() for l in content.splitlines() if l.strip()]
    parse_errors: list[str] = []
    field_errors: list[str] = []

    for i, line in enumerate(lines, 1):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            parse_errors.append(f"行{i}: {e}")
            continue
        if not isinstance(obj.get("id"), int) or isinstance(obj.get("id"), bool):
            field_errors.append(f"行{i}.id 非整数")
        if not isinstance(obj.get("product"), str):
            field_errors.append(f"行{i}.product 非字符串")
        if not isinstance(obj.get("price"), (int, float)) or isinstance(obj.get("price"), bool):
            field_errors.append(f"行{i}.price 非数字")
        if not isinstance(obj.get("in_stock"), bool):
            field_errors.append(f"行{i}.in_stock 非布尔")

    issues = parse_errors + field_errors
    expected_lines = 5
    if len(lines) != expected_lines:
        issues.insert(0, f"行数={len(lines)}，期望={expected_lines}")

    severity = Severity.error if parse_errors else (Severity.warn if issues else Severity.ok)
    return CheckResult(
        "fmt_jsonl", severity,
        f"JSONL 通过 ({len(lines)} 行)" if not issues else f"JSONL 问题: {issues[0]}",
        "; ".join(issues[:3]) if issues else f"行数={len(lines)} 全部可解析",
        latency,
        {"usage": _usage},
    )


# ── SVG ───────────────────────────────────────────────────────────────────────

# SVG namespace
_SVG_NS = "http://www.w3.org/2000/svg"

def _check_svg_content(svg_text: str) -> list[str]:
    """Parse SVG and return list of issues."""
    issues: list[str] = []
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError as e:
        return [f"XML 解析失败: {e}"]

    # Root tag
    tag = root.tag
    if tag not in ("svg", f"{{{_SVG_NS}}}svg"):
        issues.append(f"根元素应为 <svg>，实际: {tag}")

    # viewBox or width/height
    has_viewbox = root.get("viewBox") is not None
    has_wh = root.get("width") is not None and root.get("height") is not None
    if not has_viewbox and not has_wh:
        issues.append("缺少 viewBox 或 width/height 属性")

    # Must have at least one drawable child (rect/circle/path/text/g/line/ellipse/polygon)
    ns_prefix = f"{{{_SVG_NS}}}" if f"{{{_SVG_NS}}}" in root.tag else ""
    drawable_tags = {
        f"{ns_prefix}{t}" for t in
        ("rect", "circle", "path", "text", "g", "line", "ellipse", "polygon",
         "polyline", "use", "image", "tspan")
    } | {"rect", "circle", "path", "text", "g", "line", "ellipse", "polygon",
         "polyline", "use", "image", "tspan"}

    def has_drawable(el: ET.Element) -> bool:
        for child in el:
            if child.tag in drawable_tags or has_drawable(child):
                return True
        return False

    if not has_drawable(root):
        issues.append("SVG 中没有可绘制元素 (rect/circle/path/text 等)")

    return issues


async def check_format_svg(
    base_url: str, client: httpx.AsyncClient, model: str
) -> CheckResult:
    """Ask model to generate a simple SVG and validate XML structure."""
    prompt = (
        "Output ONLY a valid SVG image (no markdown fences, no explanation). "
        "Draw a simple scene: a blue rectangle as background (full width/height), "
        "a yellow circle (sun), and a green rectangle (ground). "
        'Set viewBox="0 0 200 150" width="200" height="150". '
        "The output must start with <svg and end with </svg>."
    )
    content, latency, err, _usage = await _ask_format(base_url, client, model, prompt, 512)
    if err:
        return CheckResult("fmt_svg", Severity.error, err, "", latency)

    # Some models wrap in <!DOCTYPE> or xml decl — keep only from <svg onwards
    m = re.search(r"<svg[\s>]", content, re.IGNORECASE)
    if m:
        content = content[m.start():]
    # Close at </svg>
    m2 = re.search(r"</svg>", content, re.IGNORECASE)
    if m2:
        content = content[: m2.end()]

    if not content.strip().lower().startswith("<svg"):
        return CheckResult(
            "fmt_svg", Severity.error,
            "未找到 <svg> 根元素",
            f"内容开头: {content[:80]!r}", latency,
        )

    issues = _check_svg_content(content)
    if issues:
        return CheckResult(
            "fmt_svg", Severity.warn if len(issues) <= 1 else Severity.error,
            f"SVG 问题 ({len(issues)}项): {issues[0]}",
            "; ".join(issues[:3]), latency,
        )
    # Count elements
    try:
        root = ET.fromstring(content)
        elem_count = sum(1 for _ in root.iter()) - 1
    except Exception:
        elem_count = 0
    return CheckResult(
        "fmt_svg", Severity.ok,
        f"SVG 合法 ({elem_count} 子元素)",
        f"viewBox={ET.fromstring(content).get('viewBox')} width={ET.fromstring(content).get('width')}",
        latency,
        {"usage": _usage},
    )


# ── SVG 进阶测试 ──────────────────────────────────────────────────────────────

def _extract_svg(content: str) -> str | None:
    """Extract the outermost <svg>…</svg> block from model output."""
    m = re.search(r"<svg[\s>]", content, re.IGNORECASE)
    if not m:
        return None
    content = content[m.start():]
    m2 = re.search(r"</svg>", content, re.IGNORECASE)
    if m2:
        content = content[: m2.end()]
    return content


def _iter_local(root: ET.Element, *local_tags: str):
    """Yield all descendants whose local tag name (without namespace) is in local_tags."""
    tag_set = set(local_tags)
    for el in root.iter():
        local = el.tag.split("}")[-1] if "}" in el.tag else el.tag
        if local in tag_set:
            yield el


async def check_format_svg_chart(
    base_url: str, client: httpx.AsyncClient, model: str
) -> CheckResult:
    """SVG bar chart: ≥4 rect bars, ≥2 text labels, ≥1 axis line."""
    prompt = (
        "Output ONLY a valid SVG bar chart (no markdown fences, no explanation). "
        "Draw a vertical bar chart showing 4 months of sales: "
        "Jan=80, Feb=55, Mar=95, Apr=70. "
        "Requirements: "
        "(1) Each bar must be a <rect> element with a different height. "
        "(2) Add <text> labels below each bar showing the month abbreviation. "
        "(3) Include at least one <line> element for the X-axis or Y-axis. "
        "(4) Use viewBox=\"0 0 300 220\" width=\"300\" height=\"220\". "
        "Output must start with <svg and end with </svg>. No markdown."
    )
    content, latency, err, _usage = await _ask_format(base_url, client, model, prompt, 900)
    if err:
        return CheckResult("fmt_svg_chart", Severity.error, err, "", latency)

    svg = _extract_svg(content)
    if not svg:
        return CheckResult("fmt_svg_chart", Severity.error, "未找到 <svg>", content[:80], latency)

    try:
        root = ET.fromstring(svg)
    except ET.ParseError as e:
        return CheckResult("fmt_svg_chart", Severity.error, f"XML解析失败: {e}", svg[:80], latency)

    rects  = sum(1 for _ in _iter_local(root, "rect"))
    texts  = sum(1 for _ in _iter_local(root, "text", "tspan"))
    axes   = sum(1 for _ in _iter_local(root, "line", "polyline", "path"))

    issues = []
    if rects < 4:
        issues.append(f"bar rect 数量不足 (期望≥4，实际={rects})")
    if texts < 2:
        issues.append(f"text 标签不足 (期望≥2，实际={texts})")
    if axes < 1:
        issues.append("缺少坐标轴元素 (line/polyline/path)")

    summary = f"rects={rects} texts={texts} axes={axes}"
    if issues:
        sev = Severity.warn if len(issues) == 1 else Severity.error
        return CheckResult("fmt_svg_chart", sev, f"柱状图问题: {issues[0]}", summary, latency, {"usage": _usage})
    return CheckResult(
        "fmt_svg_chart", Severity.ok,
        f"SVG 柱状图合法 (bars={rects}, labels={texts})",
        summary, latency, {"usage": _usage},
    )


async def check_format_svg_path(
    base_url: str, client: httpx.AsyncClient, model: str
) -> CheckResult:
    """SVG complex path: ≥3 path elements with Bézier curve or arc commands."""
    prompt = (
        "Output ONLY a valid SVG image (no markdown fences, no explanation). "
        "Draw an abstract composition using ONLY <path> elements (no rect/circle/ellipse/polygon). "
        "Requirements: "
        "(1) At least 3 <path> elements, each with a distinct fill color. "
        "(2) Every path's 'd' attribute MUST contain at least one cubic Bézier command "
        "(C or c) or arc command (A or a). "
        "(3) Use viewBox=\"0 0 300 300\" width=\"300\" height=\"300\". "
        "Output must start with <svg and end with </svg>. No markdown."
    )
    content, latency, err, _usage = await _ask_format(base_url, client, model, prompt, 1100)
    if err:
        return CheckResult("fmt_svg_path", Severity.error, err, "", latency)

    svg = _extract_svg(content)
    if not svg:
        return CheckResult("fmt_svg_path", Severity.error, "未找到 <svg>", content[:80], latency)

    try:
        root = ET.fromstring(svg)
    except ET.ParseError as e:
        return CheckResult("fmt_svg_path", Severity.error, f"XML解析失败: {e}", svg[:80], latency)

    paths = list(_iter_local(root, "path"))
    curve_pat = re.compile(r"[CcQqAaSsTt]")
    paths_with_curves = [p for p in paths if curve_pat.search(p.get("d", ""))]
    fills = {p.get("fill") or p.get("style", "") for p in paths} - {None, ""}

    issues = []
    if len(paths) < 3:
        issues.append(f"path 数量不足 (期望≥3，实际={len(paths)})")
    if not paths_with_curves:
        issues.append("无 path 使用曲线/圆弧命令 (C/c/Q/q/A/a)")
    elif len(paths_with_curves) < min(len(paths), 2):
        issues.append(f"使用曲线命令的 path 过少 ({len(paths_with_curves)}/{len(paths)})")

    summary = f"paths={len(paths)} with_curves={len(paths_with_curves)} distinct_fills={len(fills)}"
    if issues:
        sev = Severity.warn if len(issues) == 1 else Severity.error
        return CheckResult("fmt_svg_path", sev, f"Path SVG 问题: {issues[0]}", summary, latency, {"usage": _usage})
    return CheckResult(
        "fmt_svg_path", Severity.ok,
        f"SVG Path 合法 (paths={len(paths)}, curves={len(paths_with_curves)})",
        summary, latency, {"usage": _usage},
    )


async def check_format_svg_defs(
    base_url: str, client: httpx.AsyncClient, model: str
) -> CheckResult:
    """SVG defs + gradient: <defs>, linearGradient with stops, url(#) references."""
    prompt = (
        "Output ONLY a valid SVG image (no markdown fences, no explanation). "
        "Draw a sunset landscape using SVG gradients. "
        "Requirements: "
        "(1) Include a <defs> section. "
        "(2) Define at least one <linearGradient> inside <defs>, with at least 2 <stop> elements. "
        "(3) At least 2 shapes must reference the gradient via fill='url(#id)'. "
        "(4) Use viewBox=\"0 0 400 260\" width=\"400\" height=\"260\". "
        "Output must start with <svg and end with </svg>. No markdown."
    )
    content, latency, err, _usage = await _ask_format(base_url, client, model, prompt, 1100)
    if err:
        return CheckResult("fmt_svg_defs", Severity.error, err, "", latency)

    svg = _extract_svg(content)
    if not svg:
        return CheckResult("fmt_svg_defs", Severity.error, "未找到 <svg>", content[:80], latency)

    try:
        root = ET.fromstring(svg)
    except ET.ParseError as e:
        return CheckResult("fmt_svg_defs", Severity.error, f"XML解析失败: {e}", svg[:80], latency)

    has_defs      = any(True for _ in _iter_local(root, "defs"))
    gradients     = list(_iter_local(root, "linearGradient", "radialGradient"))
    stops         = list(_iter_local(root, "stop"))
    # Count url(#...) references anywhere in the serialised SVG
    svg_text      = ET.tostring(root, encoding="unicode")
    url_refs      = re.findall(r"url\(#[^)]+\)", svg_text)
    gradient_ids  = [g.get("id", "?") for g in gradients[:4]]

    issues = []
    if not has_defs:
        issues.append("缺少 <defs> 元素")
    if not gradients:
        issues.append("缺少 gradient 定义 (linearGradient/radialGradient)")
    if len(stops) < 2:
        issues.append(f"stop 数量不足 (期望≥2，实际={len(stops)})")
    if len(url_refs) < 2:
        issues.append(f"url(#...) 引用不足 (期望≥2，实际={len(url_refs)})")

    summary = (
        f"defs={'✓' if has_defs else '✗'} "
        f"gradients={len(gradients)}({','.join(gradient_ids)}) "
        f"stops={len(stops)} url_refs={len(url_refs)}"
    )
    if issues:
        sev = Severity.warn if len(issues) == 1 else Severity.error
        return CheckResult("fmt_svg_defs", sev, f"SVG defs 问题: {issues[0]}", summary, latency, {"usage": _usage})
    return CheckResult(
        "fmt_svg_defs", Severity.ok,
        f"SVG Defs/Gradient 合法 (gradients={len(gradients)}, url_refs={len(url_refs)})",
        summary, latency, {"usage": _usage},
    )


# ── XML ───────────────────────────────────────────────────────────────────────

async def check_format_xml(
    base_url: str, client: httpx.AsyncClient, model: str
) -> CheckResult:
    """Generate a well-formed XML document with namespace and validate it."""
    prompt = (
        "Output ONLY valid XML (no markdown fences, no explanation). "
        "Generate a <library> document with exactly 3 <book> children. "
        "Each <book> must have attributes: id (integer string) and lang (e.g. 'en'). "
        "Each <book> must contain: <title>, <author>, <year>, <isbn> elements. "
        "Start with <?xml version=\"1.0\" encoding=\"UTF-8\"?>. "
        "Output nothing else."
    )
    content, latency, err, _usage = await _ask_format(base_url, client, model, prompt, 512)
    if err:
        return CheckResult("fmt_xml", Severity.error, err, "", latency)
    if not content:
        return CheckResult("fmt_xml", Severity.error, "返回空内容", "", latency)

    # Strip any leading/trailing non-XML
    if "<?xml" in content:
        content = content[content.index("<?xml"):]
    elif "<library" in content:
        content = content[content.index("<library"):]

    try:
        root = ET.fromstring(content)
    except ET.ParseError as e:
        return CheckResult(
            "fmt_xml", Severity.error,
            f"XML 解析失败: {e}",
            content[:120].replace("\n", "↵"), latency,
        )

    issues: list[str] = []
    # Strip namespace from tag for comparison
    root_tag = re.sub(r"\{[^}]+\}", "", root.tag)
    if root_tag != "library":
        issues.append(f"根元素应为 <library>，实际: {root_tag}")

    books = root.findall("book") or root.findall(f"{{{root.tag.split('}')[0][1:]}}}book" if "{" in root.tag else "book")
    # Fallback: find any child regardless of namespace
    all_children = list(root)
    book_children = [c for c in all_children if re.sub(r"\{[^}]+\}", "", c.tag) == "book"]

    if len(book_children) != 3:
        issues.append(f"<book> 数量应为 3，实际: {len(book_children)}")

    required_children = {"title", "author", "year", "isbn"}
    for i, book in enumerate(book_children):
        child_tags = {re.sub(r"\{[^}]+\}", "", c.tag) for c in book}
        missing = required_children - child_tags
        if missing:
            issues.append(f"book[{i}] 缺少: {missing}")
        if not book.get("id"):
            issues.append(f"book[{i}] 缺少 id 属性")

    if issues:
        return CheckResult(
            "fmt_xml", Severity.warn,
            f"XML 结构问题 ({len(issues)}项)",
            "; ".join(issues[:3]), latency,
        )
    total_el = sum(1 for _ in root.iter())
    return CheckResult(
        "fmt_xml", Severity.ok,
        f"XML 合法 ({total_el} 元素, {len(book_children)} books)",
        "", latency,
        {"usage": _usage},
    )


# ── CSV ───────────────────────────────────────────────────────────────────────

async def check_format_csv(
    base_url: str, client: httpx.AsyncClient, model: str
) -> CheckResult:
    """Generate CSV with header + 5 data rows and validate consistency."""
    prompt = (
        "Output ONLY valid CSV data (no markdown fences, no explanation, no extra text). "
        "Include a header row and exactly 5 data rows. "
        "Columns: id,name,email,department,salary\n"
        "Rules: id is integer, email contains @, salary is a number. "
        "Use comma as delimiter. No quotes unless the field contains a comma."
    )
    content, latency, err, _usage = await _ask_format(base_url, client, model, prompt, 400)
    if err:
        return CheckResult("fmt_csv", Severity.error, err, "", latency)
    if not content:
        return CheckResult("fmt_csv", Severity.error, "返回空内容", "", latency)

    issues: list[str] = []
    try:
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)
    except csv.Error as e:
        return CheckResult("fmt_csv", Severity.error, f"CSV 解析失败: {e}", content[:80], latency)

    non_empty = [r for r in rows if any(cell.strip() for cell in r)]
    if len(non_empty) < 2:
        return CheckResult(
            "fmt_csv", Severity.error,
            f"CSV 行数不足: {len(non_empty)}（含header）",
            "", latency,
        )

    header = [h.strip().lower() for h in non_empty[0]]
    expected_cols = ["id", "name", "email", "department", "salary"]
    missing_cols = [c for c in expected_cols if c not in header]
    if missing_cols:
        issues.append(f"缺少列: {missing_cols}")

    data_rows = non_empty[1:]
    if len(data_rows) != 5:
        issues.append(f"数据行数={len(data_rows)}，期望=5")

    col_count = len(header)
    for i, row in enumerate(data_rows, 1):
        if len(row) != col_count:
            issues.append(f"行{i} 列数={len(row)} ≠ header列数={col_count}")
        # Validate email and salary if columns present
        if "email" in header:
            idx = header.index("email")
            if idx < len(row) and "@" not in row[idx]:
                issues.append(f"行{i} email 格式异常: {row[idx]!r}")
        if "salary" in header:
            idx = header.index("salary")
            if idx < len(row):
                try:
                    float(row[idx].replace(",", ""))
                except ValueError:
                    issues.append(f"行{i} salary 非数字: {row[idx]!r}")

    severity = Severity.error if len(non_empty) < 2 else (Severity.warn if issues else Severity.ok)
    return CheckResult(
        "fmt_csv", severity,
        f"CSV 通过 ({len(data_rows)} 数据行)" if not issues else f"CSV 问题: {issues[0]}",
        "; ".join(issues[:3]) if issues else f"cols={header}", latency,
        {"usage": _usage},
    )


# ── HTML ──────────────────────────────────────────────────────────────────────

class _HTMLValidator(_html_parser.HTMLParser):
    """Minimal HTMLParser subclass to collect tags and detect errors."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tags_seen: set[str] = set()
        self.open_stack: list[str] = []
        self.errors: list[str] = []
        self.has_doctype = False

    def handle_decl(self, decl: str):
        if decl.lower().startswith("doctype html"):
            self.has_doctype = True

    def handle_starttag(self, tag: str, attrs):
        self.tags_seen.add(tag)
        # void elements don't need closing
        VOID = {"area","base","br","col","embed","hr","img","input",
                "link","meta","param","source","track","wbr"}
        if tag not in VOID:
            self.open_stack.append(tag)

    def handle_endtag(self, tag: str):
        VOID = {"area","base","br","col","embed","hr","img","input",
                "link","meta","param","source","track","wbr"}
        if tag in VOID:
            return
        if self.open_stack and self.open_stack[-1] == tag:
            self.open_stack.pop()
        # Don't flag mismatches as errors — browsers tolerate them; we just note them

    def get_unclosed(self) -> list[str]:
        return [t for t in self.open_stack if t not in ("html", "body", "head")]


async def check_format_html(
    base_url: str, client: httpx.AsyncClient, model: str
) -> CheckResult:
    """Generate a complete HTML page and validate structure."""
    prompt = (
        "Output ONLY a complete, valid HTML5 page (no markdown fences). "
        "The page must include: <!DOCTYPE html>, <html>, <head> with <title> and "
        "a <meta charset>, <body> containing: an <h1>, a <p>, an unordered list <ul> "
        "with at least 3 <li> items, and a <table> with header row and 2 data rows. "
        "Output nothing else."
    )
    content, latency, err, _usage = await _ask_format(base_url, client, model, prompt, 700)
    if err:
        return CheckResult("fmt_html", Severity.error, err, "", latency)
    if not content:
        return CheckResult("fmt_html", Severity.error, "返回空内容", "", latency)

    # Trim to first <!DOCTYPE or <html
    for marker in ["<!doctype", "<!DOCTYPE", "<html"]:
        if marker.lower() in content.lower():
            idx = content.lower().index(marker.lower())
            content = content[idx:]
            break

    validator = _HTMLValidator()
    try:
        validator.feed(content)
    except Exception as e:
        return CheckResult("fmt_html", Severity.error, f"HTML 解析异常: {e}", "", latency)

    issues: list[str] = []
    required_tags = {
        "html": "缺少 <html>",
        "head": "缺少 <head>",
        "title": "缺少 <title>",
        "body": "缺少 <body>",
        "h1": "缺少 <h1>",
        "p": "缺少 <p>",
        "ul": "缺少 <ul>",
        "li": "缺少 <li>",
        "table": "缺少 <table>",
        "tr": "缺少 <tr>",
    }
    for tag, msg in required_tags.items():
        if tag not in validator.tags_seen:
            issues.append(msg)

    if not validator.has_doctype:
        issues.append("缺少 <!DOCTYPE html>")

    unclosed = validator.get_unclosed()
    if unclosed:
        issues.append(f"未闭合标签: {unclosed[:4]}")

    # Check meta charset
    if "meta" not in validator.tags_seen:
        issues.append("缺少 <meta charset>")

    severity = Severity.error if len(issues) >= 3 else (Severity.warn if issues else Severity.ok)
    return CheckResult(
        "fmt_html", severity,
        f"HTML 合法 (共 {len(validator.tags_seen)} 种标签)" if not issues else f"HTML 问题: {issues[0]}",
        "; ".join(issues[:3]) if issues else f"tags={sorted(validator.tags_seen)[:8]}",
        latency,
        {"usage": _usage},
    )


# ── Markdown ──────────────────────────────────────────────────────────────────

async def check_format_markdown(
    base_url: str, client: httpx.AsyncClient, model: str
) -> CheckResult:
    """Generate a Markdown document and check for required structural elements."""
    prompt = (
        "Output a well-structured Markdown document (raw Markdown, no extra explanation). "
        "The document must contain ALL of the following:\n"
        "1. A level-1 heading (# Title)\n"
        "2. At least one level-2 heading (## Section)\n"
        "3. A paragraph of at least 2 sentences\n"
        "4. A bulleted list (- item) with at least 3 items\n"
        "5. A numbered list (1. item) with at least 3 items\n"
        "6. A code block (``` ... ```)\n"
        "7. A blockquote (> text)\n"
        "8. A hyperlink [text](url)\n"
        "9. Bold text (**bold**) and italic text (*italic*)\n"
        "10. A table with at least 2 columns and 2 data rows\n"
        "Topic: best practices for API design."
    )
    content, latency, err, _usage = await _ask_format(base_url, client, model, prompt, 800)
    if err:
        return CheckResult("fmt_markdown", Severity.error, err, "", latency)
    if not content:
        return CheckResult("fmt_markdown", Severity.error, "返回空内容", "", latency)

    checks_map = {
        "H1 标题":       bool(re.search(r"^# .+", content, re.MULTILINE)),
        "H2 标题":       bool(re.search(r"^## .+", content, re.MULTILINE)),
        "无序列表":      bool(re.search(r"^[-*+] .+", content, re.MULTILINE)),
        "有序列表":      bool(re.search(r"^\d+\. .+", content, re.MULTILINE)),
        "代码块":        bool(re.search(r"```[\s\S]+?```", content)),
        "引用块":        bool(re.search(r"^> .+", content, re.MULTILINE)),
        "超链接":        bool(re.search(r"\[.+?\]\(https?://[^\)]+\)", content)),
        "加粗文本":      bool(re.search(r"\*\*.+?\*\*", content)),
        "斜体文本":      bool(re.search(r"(?<!\*)\*(?!\*)([^*]+)\*(?!\*)", content)),
        "表格":          bool(re.search(r"^\|.+\|$", content, re.MULTILINE)),
        "表格分隔行":    bool(re.search(r"^\|[\s\-:|]+\|$", content, re.MULTILINE)),
    }

    missing = [name for name, present in checks_map.items() if not present]
    found = sum(1 for v in checks_map.values() if v)
    total = len(checks_map)

    # Count words as a proxy for paragraph content
    word_count = len(content.split())
    if word_count < 80:
        missing.append(f"内容过少 ({word_count} 词)")

    severity = (
        Severity.ok if not missing else
        Severity.warn if len(missing) <= 2 else
        Severity.error
    )
    return CheckResult(
        "fmt_markdown", severity,
        f"Markdown 通过 ({found}/{total} 元素)" if not missing else f"缺少 {len(missing)} 项: {missing[0]}",
        f"缺少: {missing}" if missing else f"词数={word_count} 全部元素存在",
        latency,
        {"usage": _usage},
    )


# ── YAML ─────────────────────────────────────────────────────────────────────

async def check_format_yaml(
    base_url: str, client: httpx.AsyncClient, model: str
) -> CheckResult:
    """Generate YAML config and validate parsability + structure."""
    if not HAS_YAML:
        return CheckResult(
            "fmt_yaml", Severity.skip,
            "跳过 (pip install pyyaml 后启用)", "", 0,
        )
    prompt = (
        "Output ONLY valid YAML (no markdown fences, no explanation). "
        "Generate a server configuration with these fields:\n"
        "  server:\n"
        "    host: (string)\n"
        "    port: (integer, 1-65535)\n"
        "    debug: (boolean)\n"
        "    workers: (integer)\n"
        "  database:\n"
        "    url: (string starting with postgres:// or mysql://)\n"
        "    pool_size: (integer)\n"
        "    timeout: (float, seconds)\n"
        "  logging:\n"
        "    level: (one of: DEBUG INFO WARNING ERROR)\n"
        "    handlers: (list of strings, at least 2)\n"
    )
    content, latency, err, _usage = await _ask_format(base_url, client, model, prompt, 400)
    if err:
        return CheckResult("fmt_yaml", Severity.error, err, "", latency)
    if not content:
        return CheckResult("fmt_yaml", Severity.error, "返回空内容", "", latency)

    try:
        data = _yaml.safe_load(content)
    except _yaml.YAMLError as e:
        return CheckResult(
            "fmt_yaml", Severity.error,
            f"YAML 解析失败: {str(e)[:80]}",
            content[:100].replace("\n", "↵"), latency,
        )

    if not isinstance(data, dict):
        return CheckResult("fmt_yaml", Severity.error, f"顶层应为 mapping，实际: {type(data).__name__}", "", latency)

    issues: list[str] = []
    server = data.get("server", {})
    db = data.get("database", {})
    logging_cfg = data.get("logging", {})

    if not isinstance(server, dict):
        issues.append("server 应为 mapping")
    else:
        if not isinstance(server.get("host"), str):
            issues.append("server.host 应为 string")
        port = server.get("port")
        if not isinstance(port, int) or not (1 <= port <= 65535):
            issues.append(f"server.port 应为 1-65535 整数，实际: {port!r}")
        if not isinstance(server.get("debug"), bool):
            issues.append("server.debug 应为 boolean")

    if not isinstance(db, dict):
        issues.append("database 应为 mapping")
    else:
        url = str(db.get("url", ""))
        if not (url.startswith("postgres://") or url.startswith("mysql://")):
            issues.append(f"database.url 应以 postgres:// 或 mysql:// 开头，实际: {url[:30]!r}")
        if not isinstance(db.get("pool_size"), int):
            issues.append("database.pool_size 应为 integer")

    if not isinstance(logging_cfg, dict):
        issues.append("logging 应为 mapping")
    else:
        level = logging_cfg.get("level", "")
        if level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
            issues.append(f"logging.level 应为 DEBUG/INFO/WARNING/ERROR，实际: {level!r}")
        handlers = logging_cfg.get("handlers", [])
        if not isinstance(handlers, list) or len(handlers) < 2:
            issues.append(f"logging.handlers 应为 ≥2 元素的 list，实际: {handlers!r}")

    severity = Severity.warn if issues else Severity.ok
    return CheckResult(
        "fmt_yaml", severity,
        f"YAML 通过 (top-keys={list(data.keys())})" if not issues else f"YAML 结构问题: {issues[0]}",
        "; ".join(issues[:3]) if issues else "",
        latency,
        {"usage": _usage},
    )


# ── TOML ─────────────────────────────────────────────────────────────────────

async def check_format_toml(
    base_url: str, client: httpx.AsyncClient, model: str
) -> CheckResult:
    """Generate TOML and validate parsing."""
    if not HAS_TOML:
        return CheckResult(
            "fmt_toml", Severity.skip,
            "跳过 (Python 3.11+ 内置或 pip install tomli)", "", 0,
        )
    prompt = (
        "Output ONLY valid TOML (no markdown fences, no explanation). "
        "Generate a project configuration containing:\n"
        "  [project]\n"
        "  name = string\n"
        "  version = semver string (e.g. \"1.0.0\")\n"
        "  python_requires = string\n"
        "  authors = array of inline tables with name and email\n\n"
        "  [project.dependencies]\n"
        "  (at least 3 key=string pairs)\n\n"
        "  [tool.lint]\n"
        "  enabled = boolean\n"
        "  max_line_length = integer\n"
        "  ignore = array of strings\n"
    )
    content, latency, err, _usage = await _ask_format(base_url, client, model, prompt, 400)
    if err:
        return CheckResult("fmt_toml", Severity.error, err, "", latency)
    if not content:
        return CheckResult("fmt_toml", Severity.error, "返回空内容", "", latency)

    try:
        data = _tomllib.loads(content)
    except Exception as e:
        return CheckResult(
            "fmt_toml", Severity.error,
            f"TOML 解析失败: {str(e)[:80]}",
            content[:100].replace("\n", "↵"), latency,
        )

    issues: list[str] = []
    project = data.get("project", {})
    if not isinstance(project.get("name"), str):
        issues.append("project.name 应为 string")
    if not isinstance(project.get("version"), str):
        issues.append("project.version 应为 string")
    authors = project.get("authors", [])
    if not isinstance(authors, list) or not authors:
        issues.append("project.authors 应为非空 array")

    tool = data.get("tool", {})
    lint = tool.get("lint", {}) if isinstance(tool, dict) else {}
    if not isinstance(lint.get("enabled"), bool):
        issues.append("tool.lint.enabled 应为 boolean")
    if not isinstance(lint.get("max_line_length"), int):
        issues.append("tool.lint.max_line_length 应为 integer")

    severity = Severity.warn if issues else Severity.ok
    return CheckResult(
        "fmt_toml", severity,
        f"TOML 通过 (sections={list(data.keys())})" if not issues else f"TOML 结构问题: {issues[0]}",
        "; ".join(issues[:3]) if issues else "",
        latency,
        {"usage": _usage},
    )


# ── SQL ───────────────────────────────────────────────────────────────────────

async def check_format_sql(
    base_url: str, client: httpx.AsyncClient, model: str
) -> CheckResult:
    """Generate SQL DDL+DML and validate structure heuristically."""
    prompt = (
        "Output ONLY valid SQL statements (no markdown fences, no explanation). "
        "Write SQL that does ALL of the following:\n"
        "1. CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT UNIQUE, created_at TIMESTAMP)\n"
        "2. CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(id), amount DECIMAL(10,2), status TEXT)\n"
        "3. INSERT 3 sample rows into users\n"
        "4. INSERT 3 sample rows into orders\n"
        "5. A SELECT with JOIN between users and orders, filtered by a WHERE clause, with ORDER BY\n"
        "6. An UPDATE statement\n"
        "7. A CREATE INDEX statement\n"
        "Output only the SQL, nothing else."
    )
    content, latency, err, _usage = await _ask_format(base_url, client, model, prompt, 700)
    if err:
        return CheckResult("fmt_sql", Severity.error, err, "", latency)
    if not content:
        return CheckResult("fmt_sql", Severity.error, "返回空内容", "", latency)

    # 宽松匹配辅助：大小写不敏感，支持 IF NOT EXISTS、反引号/双引号/方括号包裹的表名
    F = re.IGNORECASE

    def _has_create_table(tbl: str) -> bool:
        # CREATE [TEMPORARY] TABLE [IF NOT EXISTS] [`"[]?<tbl>[`"]]?
        return bool(re.search(
            rf"\bCREATE\s+(?:TEMPORARY\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?"
            rf"(?:\w+\.)?[`\"\[]?{tbl}[`\"\]]?\b",
            content, F,
        ))

    def _has_insert(tbl: str) -> bool:
        return bool(re.search(
            rf"\bINSERT\s+(?:OR\s+\w+\s+)?INTO\s+(?:\w+\.)?[`\"\[]?{tbl}[`\"\]]?\b",
            content, F,
        ))

    checks_map = {
        "CREATE TABLE users":   _has_create_table("users"),
        "CREATE TABLE orders":  _has_create_table("orders"),
        "INSERT INTO users":    _has_insert("users"),
        "INSERT INTO orders":   _has_insert("orders"),
        "SELECT … JOIN":        bool(re.search(r"\bSELECT\b.+\bJOIN\b", content, F | re.DOTALL)),
        "WHERE clause":         bool(re.search(r"\bWHERE\b", content, F)),
        "ORDER BY":             bool(re.search(r"\bORDER\s+BY\b", content, F)),
        "UPDATE statement":     bool(re.search(r"\bUPDATE\b", content, F)),
        "CREATE INDEX":         bool(re.search(r"\bCREATE\s+(?:UNIQUE\s+)?INDEX\b", content, F)),
        "PRIMARY KEY":          bool(re.search(r"\bPRIMARY\s+KEY\b", content, F)),
        "REFERENCES (FK)":      bool(re.search(r"\bREFERENCES\b", content, F)),
    }

    missing = [k for k, v in checks_map.items() if not v]
    found = sum(1 for v in checks_map.values() if v)

    # 分号数量：部分模型用换行代替分号，适当放宽
    stmt_count = content.count(";")
    if stmt_count < 3:
        missing.append(f"语句终止符太少 ({stmt_count} 个分号，期望 ≥3)")

    severity = (
        Severity.ok if not missing else
        Severity.warn if len(missing) <= 2 else
        Severity.error
    )
    return CheckResult(
        "fmt_sql", severity,
        f"SQL 通过 ({found}/{len(checks_map)} 检查项)" if not missing else f"SQL 缺少 {len(missing)} 项: {missing[0]}",
        f"缺少: {missing[:4]}" if missing else f"分号数={stmt_count}",
        latency,
        {"usage": _usage},
    )


# ── 格式测试汇总 ───────────────────────────────────────────────────────────────

FORMAT_CHECKS = [
    ("fmt_json",      check_format_json),
    ("fmt_jsonl",     check_format_jsonl),
    ("fmt_svg",       check_format_svg),
    ("fmt_svg_chart", check_format_svg_chart),
    ("fmt_svg_path",  check_format_svg_path),
    ("fmt_svg_defs",  check_format_svg_defs),
    ("fmt_xml",       check_format_xml),
    ("fmt_csv",       check_format_csv),
    ("fmt_html",      check_format_html),
    ("fmt_markdown",  check_format_markdown),
    ("fmt_yaml",      check_format_yaml),
    ("fmt_toml",      check_format_toml),
    ("fmt_sql",       check_format_sql),
]


def _is_connect_error(result: CheckResult) -> bool:
    """Return True when a CheckResult represents a transient connection failure."""
    if result.severity != Severity.error:
        return False
    haystack = (result.message or "") + str(result.detail or "")
    return "ConnectError" in haystack or "Connection refused" in haystack


def _is_rate_limited(result: CheckResult) -> bool:
    """Return True when the provider is throttling us (HTTP 429)."""
    haystack = (result.message or "") + str(result.detail or "")
    return "429" in haystack


async def _run_with_retry(fn, *args, retries: int = 2, base_delay: float = 2.0) -> CheckResult:
    """
    Call ``fn(*args)`` and retry on transient failures:
    - ConnectError → exponential back-off starting at ``base_delay`` (2 s, 4 s, …)
    - HTTP 429 rate-limit → fixed 30 s wait before each retry (up to 2 retries)

    If still rate-limited after all retries, the result is downgraded to SKIP so
    quota exhaustion does not pollute ERROR/WARN counts.
    """
    result = await fn(*args)
    for attempt in range(retries):
        if _is_connect_error(result):
            wait = base_delay * (2 ** attempt)   # 2 s, 4 s
        elif _is_rate_limited(result):
            wait = 30.0                           # respect provider rate-limit window
        else:
            break
        await asyncio.sleep(wait)
        result = await fn(*args)
    # After exhausting retries, convert persistent 429 → SKIP (not a compliance failure)
    if _is_rate_limited(result):
        return CheckResult(result.name, Severity.skip, "速率限制 (429)，已跳过", "", result.latency_ms)
    return result


async def run_format_checks(
    base_url: str, client: httpx.AsyncClient, model: str, check_delay: float = 0.0
) -> list[CheckResult]:
    """Run all structured-format generation checks sequentially."""
    results = []
    for _name, fn in FORMAT_CHECKS:
        if check_delay > 0 and results:   # skip delay before the very first check
            await asyncio.sleep(check_delay)
        result = await _run_with_retry(fn, base_url, client, model)
        results.append(result)
    return results


# ── Anthropic checks ───────────────────────────────────────────────────────────

async def check_anthropic_messages(
    base_url: str, client: httpx.AsyncClient, model: str
) -> CheckResult:
    url = _api_url(base_url, "/v1/messages")
    payload = {
        "model": model,
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "Reply with exactly: PONG"}],
    }
    t0 = time.monotonic()
    try:
        r = await client.post(url, json=payload)
        latency = (time.monotonic() - t0) * 1000
        if r.status_code == 200:
            try:
                data = r.json()
            except Exception as e:
                return CheckResult("anthropic_messages", Severity.error, "响应非 JSON", str(e), latency)

            issues = []
            for fld in ["id", "type", "role", "content", "model", "stop_reason", "usage"]:
                if fld not in data:
                    issues.append(f"缺少字段: {fld}")

            if data.get("type") != "message":
                issues.append(f"type 应为 message，实际: {data.get('type')}")
            if data.get("role") != "assistant":
                issues.append(f"role 应为 assistant，实际: {data.get('role')}")

            content_blocks = data.get("content", [])
            text = " ".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")

            if issues:
                return CheckResult("anthropic_messages", Severity.warn, "协议合规警告", "; ".join(issues), latency)
            return CheckResult(
                "anthropic_messages", Severity.ok,
                "Anthropic messages API 正常",
                f"content: {text[:80]}",
                latency,
            )
        elif r.status_code == 401:
            return CheckResult("anthropic_messages", Severity.error, "401 认证失败", r.text[:200], latency)
        else:
            return CheckResult("anthropic_messages", Severity.error, f"HTTP {r.status_code}", r.text[:200], latency)
    except httpx.TimeoutException:
        return CheckResult("anthropic_messages", Severity.error, "超时", "")
    except Exception as e:
        return CheckResult("anthropic_messages", Severity.error, str(e), "")


async def check_anthropic_streaming(
    base_url: str, client: httpx.AsyncClient, model: str
) -> CheckResult:
    url = _api_url(base_url, "/v1/messages")
    payload = {
        "model": model,
        "max_tokens": 64,
        "stream": True,
        "messages": [{"role": "user", "content": "Count 1 to 5"}],
    }
    events: list[dict] = []
    parse_errors: list[str] = []
    text_parts: list[str] = []
    first_latency: float = 0.0
    last_time: float = 0.0
    silence_gaps: list[float] = []

    t0 = time.monotonic()
    try:
        async with client.stream("POST", url, json=payload) as r:
            if r.status_code != 200:
                body = await r.aread()
                return CheckResult(
                    "anthropic_streaming", Severity.error,
                    f"HTTP {r.status_code}",
                    body.decode()[:200],
                    (time.monotonic() - t0) * 1000,
                )
            event_type = None
            async for line in r.aiter_lines():
                now = time.monotonic()
                if last_time and (now - last_time) > 5.0:
                    silence_gaps.append(round((now - last_time) * 1000))
                last_time = now
                if not first_latency and line:
                    first_latency = (now - t0) * 1000

                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    data_str = line[5:].strip()
                    try:
                        ev = json.loads(data_str)
                        ev["_event_type"] = event_type
                        events.append(ev)
                        if ev.get("type") == "content_block_delta":
                            delta = ev.get("delta", {})
                            if delta.get("type") == "text_delta":
                                text_parts.append(delta.get("text", ""))
                    except json.JSONDecodeError as e:
                        parse_errors.append(f"{data_str[:60]} — {e}")
    except httpx.TimeoutException:
        return CheckResult(
            "anthropic_streaming", Severity.error,
            f"超时 (>{TIMEOUT_READ}s)",
            f"已收到 {len(events)} events",
            (time.monotonic() - t0) * 1000,
        )
    except Exception as e:
        return CheckResult(
            "anthropic_streaming", Severity.error, str(e), "",
            (time.monotonic() - t0) * 1000,
        )

    total_latency = (time.monotonic() - t0) * 1000
    full_text = "".join(text_parts)
    event_types = [e.get("type") for e in events]

    issues = []
    if parse_errors:
        issues.append(f"SSE解析错误 x{len(parse_errors)}")
    if "message_start" not in event_types:
        issues.append("缺少 message_start 事件")
    if "message_delta" not in event_types and "message_stop" not in event_types:
        issues.append("缺少 message_stop 事件")
    if not full_text:
        issues.append("文本内容为空")
    if silence_gaps:
        issues.append(f"流中断 {len(silence_gaps)} 次")

    severity = Severity.error if not events else (Severity.warn if issues else Severity.ok)
    detail = f"events={len(events)} ttfb={first_latency:.0f}ms total={total_latency:.0f}ms content='{full_text[:60]}'"

    return CheckResult(
        "anthropic_streaming",
        severity,
        "Anthropic 流式正常" if not issues else f"流式问题: {issues[0]}",
        detail,
        total_latency,
        {"event_types": list(dict.fromkeys(event_types)), "silence_gaps": silence_gaps},
    )


# ── Run all checks ─────────────────────────────────────────────────────────────

async def run_openai_checks(
    base_url: str,
    api_key: str | None,
    model: str,
    extra_headers: dict,
    skip_formats: bool = False,
    check_delay: float = 0.0,
    format_only: bool = False,
) -> list[CheckResult]:
    """
    format_only=True: 只运行格式合规相关检测（chat sanity + json_mode + tool_calling + fmt_*）。
    省略 models_list / streaming / error_format / system_prompt / long_output / latency_p95。
    """
    async with make_client(api_key, extra_headers) as client:
        results = []
        if format_only:
            core_checks = [
                (check_openai_chat,         (base_url, client, model)),
                (check_openai_json_mode,    (base_url, client, model)),
                (check_openai_tool_calling, (base_url, client, model)),
            ]
        else:
            core_checks = [
                (check_openai_models,       (base_url, client)),
                (check_openai_chat,         (base_url, client, model)),
                (check_openai_streaming,    (base_url, client, model)),
                (check_openai_json_mode,    (base_url, client, model)),
                (check_openai_error_format, (base_url, client, model)),
                (check_openai_system_prompt,(base_url, client, model)),
                (check_openai_tool_calling, (base_url, client, model)),
                (check_openai_long_output,  (base_url, client, model)),
                (check_openai_latency_p95,  (base_url, client, model)),
            ]
        for fn, args in core_checks:
            if check_delay > 0 and results:
                await asyncio.sleep(check_delay)
            results.append(await _run_with_retry(fn, *args))

        if not skip_formats:
            fmt_results = await run_format_checks(base_url, client, model, check_delay=check_delay)
            results.extend(fmt_results)

        return results


async def run_anthropic_checks(
    base_url: str, api_key: str | None, model: str, extra_headers: dict
) -> list[CheckResult]:
    async with make_anthropic_client(api_key, extra_headers) as client:
        checks = [
            (check_anthropic_messages,  (base_url, client, model)),
            (check_anthropic_streaming, (base_url, client, model)),
        ]
        results = []
        for fn, args in checks:
            results.append(await _run_with_retry(fn, *args))
        return results


# ── Rendering ──────────────────────────────────────────────────────────────────

SEVERITY_STYLE = {
    Severity.ok: ("✅", "green"),
    Severity.warn: ("⚠️ ", "yellow"),
    Severity.error: ("❌", "red bold"),
    Severity.skip: ("⏭️ ", "dim"),
}


def render_report(report: DiagReport) -> None:
    summary = report.summary()
    title_color = "green" if report.passed() else "red"
    console.print()
    console.print(
        Panel(
            f"[bold]{report.target_url}[/bold]\n"
            f"Protocol: [cyan]{report.protocol}[/cyan]  Model: [cyan]{report.model}[/cyan]  Time: {report.timestamp}\n"
            f"[green]OK: {summary['ok']}[/green]  [yellow]WARN: {summary['warn']}[/yellow]  "
            f"[red]ERROR: {summary['error']}[/red]  [dim]SKIP: {summary['skip']}[/dim]",
            title=f"[{title_color}]LLM API 诊断报告[/{title_color}]",
            border_style=title_color,
            expand=False,
        )
    )

    table = Table(box=box.ROUNDED, expand=True, highlight=True)
    table.add_column("状态", width=4, justify="center")
    table.add_column("检测项", style="bold", min_width=22)
    table.add_column("结果", min_width=30)
    table.add_column("延迟", justify="right", width=10)
    table.add_column("详情", style="dim", min_width=30)

    for c in report.checks:
        icon, color = SEVERITY_STYLE[c.severity]
        latency_str = f"{c.latency_ms:.0f}ms" if c.latency_ms else "—"
        if c.latency_ms > 5000:
            latency_str = f"[red]{latency_str}[/red]"
        elif c.latency_ms > 2000:
            latency_str = f"[yellow]{latency_str}[/yellow]"
        table.add_row(
            icon,
            f"[{color}]{c.name}[/{color}]",
            f"[{color}]{c.message}[/{color}]",
            latency_str,
            c.detail[:120] if c.detail else "",
        )

    console.print(table)


def render_json_report(report: DiagReport) -> None:
    data = {
        "target_url": report.target_url,
        "protocol": report.protocol,
        "model": report.model,
        "timestamp": report.timestamp,
        "summary": report.summary(),
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
    console.print_json(json.dumps(data, ensure_ascii=False, indent=2))


# ── CLI ────────────────────────────────────────────────────────────────────────

@app.command()
def diagnose(
    url: str = typer.Argument(..., help="API base URL, e.g. https://api.openai.com"),
    model: str = typer.Option("gpt-4o-mini", "--model", "-m", help="Model ID to test"),
    api_key: str | None = typer.Option(None, "--api-key", "-k", envvar="LLM_API_KEY", help="API key (or set LLM_API_KEY env)"),
    protocol: Protocol = typer.Option(Protocol.auto, "--protocol", "-p", help="API protocol (auto/openai/anthropic)"),
    header: list[str] = typer.Option([], "--header", "-H", help="Extra headers: 'Key: Value'"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON report"),
    output_file: str | None = typer.Option(None, "--output", "-o", help="Save JSON report to file"),
    skip_formats: bool = typer.Option(False, "--skip-formats", help="Skip structured format generation checks (JSON/SVG/XML/CSV/HTML/Markdown/YAML/TOML/SQL)"),
    only_formats: bool = typer.Option(False, "--only-formats", help="Run ONLY structured format generation checks"),
):
    """
    Run full health diagnostics on an LLM API endpoint.

    Examples:

      llm-diagnose https://api.openai.com --model gpt-4o-mini -k sk-...

      llm-diagnose https://api.anthropic.com -p anthropic --model claude-3-5-haiku-20241022 -k sk-ant-...

      llm-diagnose http://localhost:11434 --model llama3 -p openai

      llm-diagnose https://api.openai.com --model gpt-4o -k sk-... --only-formats
    """
    extra_headers = {}
    for h in header:
        if ":" in h:
            k, v = h.split(":", 1)
            extra_headers[k.strip()] = v.strip()

    asyncio.run(_diagnose(url, model, api_key, protocol, extra_headers, json_output, output_file, skip_formats, only_formats))


async def _diagnose(
    url: str,
    model: str,
    api_key: str | None,
    protocol: Protocol,
    extra_headers: dict,
    json_output: bool,
    output_file: str | None,
    skip_formats: bool = False,
    only_formats: bool = False,
) -> None:
    # Detect protocol
    detected_protocol = protocol.value
    if protocol == Protocol.auto:
        with console.status("[cyan]检测 API 协议...[/cyan]"):
            detected_protocol = await detect_protocol(url, api_key)
        console.print(f"[dim]检测到协议: [cyan]{detected_protocol}[/cyan][/dim]")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = DiagReport(
        target_url=url,
        protocol=detected_protocol,
        model=model,
        timestamp=timestamp,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("运行诊断检测...", total=None)

        if detected_protocol == "anthropic":
            results = await run_anthropic_checks(url, api_key, model, extra_headers)
            if not skip_formats and not only_formats:
                pass  # anthropic checks don't include format tests yet
            if only_formats or (not skip_formats and detected_protocol != "anthropic"):
                # Format checks use OpenAI-compatible client regardless of detection
                async with make_client(api_key, extra_headers) as fmt_client:
                    results += await run_format_checks(url, fmt_client, model)
        else:
            results = await run_openai_checks(
                url, api_key, model, extra_headers,
                skip_formats=(skip_formats and not only_formats),
            )
            if only_formats:
                # Replace with only format results
                async with make_client(api_key, extra_headers) as fmt_client:
                    results = await run_format_checks(url, fmt_client, model)

        report.checks.extend(results)

    if json_output or output_file:
        data = {
            "target_url": report.target_url,
            "protocol": report.protocol,
            "model": report.model,
            "timestamp": report.timestamp,
            "summary": report.summary(),
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
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            console.print(f"[green]报告已保存: {output_file}[/green]")
        if json_output:
            console.print_json(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        render_report(report)

    if not report.passed():
        raise typer.Exit(code=1)


@app.command()
def compare(
    url_a: str = typer.Argument(..., help="第一个 API endpoint"),
    url_b: str = typer.Argument(..., help="第二个 API endpoint"),
    model: str = typer.Option("gpt-4o-mini", "--model", "-m"),
    api_key: str | None = typer.Option(None, "--api-key", "-k", envvar="LLM_API_KEY"),
    protocol: Protocol = typer.Option(Protocol.auto, "--protocol", "-p"),
):
    """Compare two API endpoints side by side."""

    async def _run():
        detected_a = protocol.value
        detected_b = protocol.value
        if protocol == Protocol.auto:
            with console.status("检测协议..."):
                detected_a, detected_b = await asyncio.gather(
                    detect_protocol(url_a, api_key),
                    detect_protocol(url_b, api_key),
                )

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_a = DiagReport(url_a, detected_a, model, timestamp)
        report_b = DiagReport(url_b, detected_b, model, timestamp)

        with console.status("运行 A 端诊断..."):
            report_a.checks.extend(await run_openai_checks(url_a, api_key, model, {}))
        with console.status("运行 B 端诊断..."):
            report_b.checks.extend(await run_openai_checks(url_b, api_key, model, {}))

        # Side-by-side table
        table = Table(title="API 对比", box=box.ROUNDED, expand=True)
        table.add_column("检测项")
        table.add_column(f"A: {url_a}", min_width=30)
        table.add_column(f"B: {url_b}", min_width=30)

        checks_b = {c.name: c for c in report_b.checks}
        for ca in report_a.checks:
            cb = checks_b.get(ca.name)
            icon_a, color_a = SEVERITY_STYLE[ca.severity]
            icon_b, color_b = SEVERITY_STYLE[cb.severity] if cb else ("—", "dim")
            cell_a = f"{icon_a} [{color_a}]{ca.message[:40]}[/{color_a}]\n[dim]{ca.latency_ms:.0f}ms[/dim]"
            cell_b = (
                f"{icon_b} [{color_b}]{cb.message[:40]}[/{color_b}]\n[dim]{cb.latency_ms:.0f}ms[/dim]"
                if cb else "—"
            )
            table.add_row(ca.name, cell_a, cell_b)

        console.print(table)

    asyncio.run(_run())


if __name__ == "__main__":
    app()
