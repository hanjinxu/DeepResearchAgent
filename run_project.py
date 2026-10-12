"""
DeepResearch Agent 项目执行脚本
等价于 run.ipynb 的执行流程：
  1. 构建带 checkpointer 的完整 Agent 图
  2. 执行调研任务
  3. 保存最终报告为 Markdown 文件

用法：
  python run_project.py                                # 默认任务
  python run_project.py "你的调研问题"                 # 自定义主题
  python run_project.py "余承东竹知了事件" --template legal_audit
  python run_project.py --template general "AI Agent Memory 调研"

模板选项：
  - auto        : 根据 query 关键词自动选择（默认）
  - legal_audit : 强制套用法务审计版 Layer 1 硬性骨架
  - general     : 强制套用通用调研报告模板

所有大模型调用统一使用 DeepSeek（见 config.yml 的 cognition.deepseek 配置）。
"""

import argparse
import asyncio
import sys
import re
import warnings
from datetime import date

from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage

from deep_research.agent_builder import deep_researcher_builder
from deep_research.prompts import is_legal_audit_query

warnings.filterwarnings("ignore")

DEFAULT_QUERY = (
    "我是个 AI 产品经理，正在考虑给产品增加个性化记忆功能。"
    "想系统了解一下现在 Agent 里的 Memory 模块都在往哪些方向发展，"
    "包括短期和长期记忆是怎么做的、各种技术路线的差异和取舍。"
    "你帮我写个调研报告，从工程落地和未来演进的角度分析一下哪些方向更值得投入"
)


def _slugify(text: str, max_len: int = 40) -> str:
    """从 query 生成安全的文件名后缀"""
    slug = re.sub(r"[^\w\u4e00-\u9fa5]+", "_", text).strip("_")
    return slug[:max_len] or "report"


def _parse_args(argv: list[str]) -> tuple[str, str]:
    """解析命令行参数。允许 query 位置参数 + --template 选项混排。"""
    parser = argparse.ArgumentParser(
        prog="run_project.py",
        description="DeepResearch Agent 项目执行入口（unified DeepSeek backend）",
    )
    parser.add_argument("query", nargs="?", default=None, help="调研主题（不传则用默认）")
    parser.add_argument(
        "--template",
        choices=["auto", "legal_audit", "general"],
        default="auto",
        help="报告模板选择：auto (默认，自动识别) / legal_audit (法务舆情骨架) / general (通用调研)",
    )
    ns = parser.parse_args(argv)
    return ns.query or DEFAULT_QUERY, ns.template


async def main():
    # ===== 1. 构建完整 Agent 图（带内存 checkpoint） =====
    checkpointer = InMemorySaver()
    full_agent = deep_researcher_builder.compile(checkpointer=checkpointer)
    print("✅ Agent 图构建完成")

    # ===== 2. 解析入参（query + 模板选择） =====
    research_query, tpl = _parse_args(sys.argv[1:])

    # 提前预览模板选择原因（auto 时）
    auto_decision = ""
    if tpl == "auto":
        auto_decision = (
            "✅ 命中法务舆情关键词 → 自动套用 legal_audit"
            if is_legal_audit_query(research_query)
            else "ℹ️ 未命中法务关键词 → 使用 general"
        )

    print(f"🔎 研究主题: {research_query[:80]}{'...' if len(research_query) > 80 else ''}")
    print(f"📐 报告模板: {tpl} {('— ' + auto_decision) if auto_decision else '(强制)'}")
    print(f"🚀 开始执行调研任务（预计 10~20 分钟）...\n{'-' * 60}")

    thread = {
        "configurable": {
            "thread_id": "1",
            "recursion_limit": 50,
            "report_template": tpl,
        }
    }
    result = await full_agent.ainvoke(
        {"messages": [HumanMessage(content=research_query)]},
        config=thread,
    )

    # ===== 3. 保存最终报告 =====
    final_report = result.get("final_report", "")
    if not final_report:
        print("⚠️ 未获取到 final_report，返回字段: ", list(result.keys()))
        return

    path = f"./output_report_{date.today().strftime('%Y-%m-%d')}_{_slugify(research_query)}.md"
    with open(path, "w", encoding="utf-8") as fd:
        fd.write(final_report)
    print(f"\n{'=' * 60}\n✅ 调研报告已保存: {path}\n{'=' * 60}")
    print(f"\n📄 报告预览（前 2000 字）:\n{'-' * 60}\n{final_report[:2000]}")


if __name__ == "__main__":
    asyncio.run(main())
