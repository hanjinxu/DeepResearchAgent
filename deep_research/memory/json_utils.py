#***********************************************
#      Filename: json_utils.py
#   Description: 从模型自由文本中稳健地抠出 JSON
#***********************************************

"""本项目不使用 response_format（DeepSeek 不支持），统一走「自由输出 + 手动解析」。"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def extract_json_object(text: str) -> Dict[str, Any]:
    """从模型输出中抠出第一个合法 JSON 对象。

    依次尝试：去代码块围栏 → 直接 json.loads → 截取首尾大括号后再解析。
    全部失败则抛 json.JSONDecodeError。
    """
    raw = (text or "").strip()
    raw = _FENCE_RE.sub("", raw).strip()

    if not raw:
        raise json.JSONDecodeError("empty model output", raw or "", 0)

    try:
        # strict=False：允许字符串里出现裸换行/制表符等控制字符。
        # 模型输出多行文本时几乎必然带控制字符，strict=True 会直接报
        # "Invalid control character"，导致整段内容被丢弃（实测历史日志里 53% 的摘要因此失败）。
        parsed = json.loads(raw, strict=False)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return parsed[0]
    except json.JSONDecodeError:
        pass

    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        return json.loads(raw[start:end + 1], strict=False)

    raise json.JSONDecodeError("no JSON object found in model output", raw, 0)


def safe_extract_json_object(text: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """不抛异常的版本，解析失败返回 default（默认空 dict）。"""
    try:
        return extract_json_object(text)
    except Exception:
        return default if default is not None else {}
