#***********************************************
#      Filename: registry.py
#   Description: 自定义Tool注册中心
#***********************************************

"""
Tool 注册中心 — 可插拔式工具管理

设计模式：Registry（注册表模式）
与 search_factory.py 的 _PROVIDER_REGISTRY 设计理念一致。

核心能力：
  1. 外部模块 @register 即可注册工具，不修改核心代码
  2. 按 category 分组，支持按角色绑定不同工具集
  3. 支持函数式、Schema式、伪工具三种模式
"""

from typing import Dict, List, Optional, Any, Callable
from langchain_core.tools import BaseTool

from deep_research import logging as dr_logging

logger = dr_logging.get_logger(__name__)


# ===== Tool 注册表 =====

_TOOL_REGISTRY: Dict[str, BaseTool] = {}
_ROLE_TOOLS: Dict[str, List[str]] = {}


def register(
    tool_obj: BaseTool,
    roles: Optional[List[str]] = None,
    replace: bool = False,
) -> BaseTool:
    """
    注册一个 Tool 到全局注册表。

    用法：
        @register(roles=["supervisor", "researcher"])
        @tool
        def my_tool(...): ...

    或：
        register(some_tool, roles=["supervisor"])

    Args:
        tool_obj: LangChain Tool 对象
        roles: 哪些角色可以使用此工具（None = 所有角色）
        replace: 是否覆盖已注册的同名工具
    """
    name = tool_obj.name if hasattr(tool_obj, "name") else str(tool_obj)

    if name in _TOOL_REGISTRY and not replace:
        logger.warning(f"工具 '{name}' 已注册，跳过")
        return tool_obj

    _TOOL_REGISTRY[name] = tool_obj

    if roles:
        for role in roles:
            _ROLE_TOOLS.setdefault(role, []).append(name)

    logger.info(f"[REGISTRY] 注册工具: {name}, 可用角色: {roles or '所有'}")
    return tool_obj


def get_tools(role: Optional[str] = None) -> List[BaseTool]:
    """
    获取工具列表 — 按角色筛选。

    Args:
        role: 角色名，如 "supervisor"、"researcher"、"writer"
              为 None 时返回全部工具
    """
    if role is None:
        return list(_TOOL_REGISTRY.values())

    names = _ROLE_TOOLS.get(role, [])
    return [_TOOL_REGISTRY[n] for n in names if n in _TOOL_REGISTRY]


def get_tools_by_name() -> Dict[str, BaseTool]:
    """获取 name → Tool 映射，用于 tool_node 按名路由"""
    return dict(_TOOL_REGISTRY)


def list_tools() -> None:
    """打印所有注册的工具"""
    print(f"已注册 {len(_TOOL_REGISTRY)} 个工具：\n")

    # 按角色分组显示
    roles_used = set()
    for role, names in _ROLE_TOOLS.items():
        roles_used.add(role)
        print(f"  [{role}]")
        for name in names:
            t = _TOOL_REGISTRY.get(name)
            if t and t.__doc__:
                desc = t.__doc__.strip().split("\n")[0][:60]
            else:
                desc = ""
            print(f"    {name:30s} {desc}")
        print()

    # 显示未分配角色的工具
    assigned = set()
    for names in _ROLE_TOOLS.values():
        assigned.update(names)
    unassigned = [n for n in _TOOL_REGISTRY if n not in assigned]
    if unassigned:
        print(f"  [未分配角色]")
        for name in unassigned:
            t = _TOOL_REGISTRY.get(name)
            desc = t.__doc__.strip().split("\n")[0][:60] if t and t.__doc__ else ""
            print(f"    {name:30s} {desc}")
        print()


def clear() -> None:
    """清空注册表（用于测试）"""
    _TOOL_REGISTRY.clear()
    _ROLE_TOOLS.clear()
