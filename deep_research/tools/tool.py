#***********************************************
#      Filename: tools.py
#   Description: 可调用工具列表  
#***********************************************

import os
from typing import Optional
from typing_extensions import Annotated, List, Literal
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool, InjectedToolArg

from deep_research import logging as dr_logging
from deep_research.utils import get_today_str
from deep_research.llm import get_chat_model
from deep_research.states import Summary
from deep_research.prompts import SUMMARIZE_PROMPT, REFINE_DRAFT_REPORT_PROMPT 
from deep_research.tools.search_factory import (
    SearchConfigError,
    get_search_client,
    get_search_defaults,
    get_search_provider,
)


logger = dr_logging.get_logger(__name__)

summarization_model = get_chat_model("researcher_summarizer")
writer_model = get_chat_model("writer")
MAX_CONTEXT_LENGTH = 250000
DEFAULT_MAX_CONTEXT = 1000
search_provider = None
search_client = None
search_defaults = None


def _ensure_search_runtime(raise_on_error: bool = True):
    """初始化搜索提供client/默认值以处理自定义搜索后端。
    """

    global search_provider, search_client, search_defaults

    try:
        if search_provider is None:
            logger.debug("Resolving search provider (lazy init)")
            search_provider = get_search_provider()
        if search_client is None:
            logger.debug("Resolving search client (lazy init)")
            search_client = get_search_client()
        if search_defaults is None:
            logger.debug("Resolving search defaults (lazy init)")
            search_defaults = get_search_defaults()
    except SearchConfigError as exc:
        logger.error("Search runtime initialization failed: %s", exc)
        if raise_on_error:
            raise
    except Exception as exc:
        logger.error("Unexpected error during search runtime init: %s", exc)
        if raise_on_error:
            raise

    return search_provider, search_client, search_defaults



# 提取requests/httpx里面的TIMEOUT EXCEPTION class, 这个异常用来排查问题非常有用
try:
    import requests
    _REQUESTS_TIMEOUT_EXC = (requests.exceptions.Timeout,)
except Exception:
    _REQUESTS_TIMEOUT_EXC = tuple()

try:
    import httpx
    _HTTPX_TIMEOUT_EXC = (httpx.TimeoutException,)
except Exception:
    _HTTPX_TIMEOUT_EXC = tuple()

_TIMEOUT_EXCEPTIONS = (TimeoutError,) + _REQUESTS_TIMEOUT_EXC + _HTTPX_TIMEOUT_EXC



# ===== SEARCH FUNCTIONS =====

def _resolve_search_runtime(client=None, provider=None, defaults=None, *, raise_on_error: bool = True):
    """解析 search provider/client/defaults参数，缓存由 search_factory 处理。
    raise_on_error：可以允许选择不抛出异常。默认情况下，会抛出。
    """
    runtime_provider, runtime_client, runtime_defaults = _ensure_search_runtime(raise_on_error=raise_on_error)
    resolved_provider = provider or runtime_provider
    resolved_client = client or runtime_client
    resolved_defaults = defaults or runtime_defaults

    return resolved_provider, resolved_client, resolved_defaults


def tavily_search_multiple(
    search_queries: List[str],
    max_results: Optional[int] = 3,
    topic: Optional[Literal["general", "news", "finance"]] = "general",
    include_raw_content: Optional[bool] = True,
    client=None,
    provider=None,
    defaults=None,
    timeout_seconds: Optional[int] = None,
) -> List[dict]:
    """根据搜索参数执行多个query的检索"""

    # 获取 search provider, search client, defaults参数
    provider, client, defaults_obj = _resolve_search_runtime(client, provider, defaults)

    if provider is None or client is None or defaults_obj is None:
        logger.error(
            "Search runtime not initialized (provider=%s, client=%s, defaults=%s)",
            bool(provider),
            bool(client),
            bool(defaults_obj),
        )
        raise SearchConfigError("Search runtime unavailable: provider/client/defaults could not be resolved")

    # 如果调用没有设置这几个参数，就使用默认参数
    effective_max_results = max_results if max_results is not None else defaults_obj.get("max_results", 3)
    effective_topic = topic if topic is not None else defaults_obj.get("topic", "general")
    effective_include_raw = include_raw_content if include_raw_content is not None else True
    effective_timeout = timeout_seconds if timeout_seconds is not None else defaults_obj.get("timeout_seconds")

    # 调用搜索函数，注：这里也可以使用AsyncTavilyClient实现并行调用
    search_docs = []
    for query in search_queries:
        try:
            result = provider.search(
                client,
                query,
                max_results=effective_max_results,
                include_raw_content=effective_include_raw,
                topic=effective_topic,
                timeout_seconds=effective_timeout,
            )
        except _TIMEOUT_EXCEPTIONS as exc:
            logger.error(
                "Search timeout for query='%s' topic='%s' timeout=%s: %s",
                query,
                effective_topic,
                effective_timeout,
                exc,
            )
            raise
        except Exception as exc:
            logger.error(
                "Search execution failed for query='%s' backend topic='%s': %s",
                query,
                effective_topic,
                exc,
            )
            raise

        search_docs.append(result)

    return search_docs



def summarize_webpage_content(webpage_content: str) -> str:
    """对一个网页的长文档进行总结

    Args:
        webpage_content: 网页原始内容 

    Returns:
        summary和key excerpts
    """
    try:
        # 组装prompt，要求JSON格式
        prompt = SUMMARIZE_PROMPT.format(
            webpage_content=webpage_content, 
            date=get_today_str()
        ) + "\n\n请以JSON格式返回，格式为：{\"summary\": \"...\", \"key_excerpts\": \"...\"}"

        # 直接调用模型，手动解析JSON
        response = summarization_model.invoke([HumanMessage(content=prompt)])
        import json, re
        text = response.content
        json_match = re.search(r'\{.*"summary".*\}', text, re.DOTALL)
        if json_match:
            # strict=False：模型返回的多行 summary 里常含裸换行，strict=True 会整段失败，
            # 失败后降级成「前 1000 字截断」，等于把证据砍掉（历史日志里 19/36 次）
            parsed = json.loads(json_match.group(), strict=False)
            summary = parsed.get("summary", "")
            key_excerpts = parsed.get("key_excerpts", "")
        else:
            summary = text[:2000]
            key_excerpts = ""

        # 格式化summary和key_excerpts
        formatted_summary = (
            f"<summary>\n{summary}\n</summary>\n\n"
            f"<key_excerpts>\n{key_excerpts}\n</key_excerpts>"
        )

        return formatted_summary

    except Exception as e:
        logger.error(f"Failed to summarize webpage: {str(e)}")
        # 如果报错，就取文档前1000字
        return webpage_content[:DEFAULT_MAX_CONTEXT] + "..."


def deduplicate_search_results(search_results: List[dict]) -> dict:
    """对urls去重，以免处理重复文档

    Args:
        search_results: 搜索结果列表List[Dict] 

    Returns:
        去重后的搜索结果Dict（按权威度贪心保留）
    """

    # ─── Step 1: URL 去重（原逻辑） ───
    url_unique = {}
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in url_unique:
                url_unique[url] = result

    results = list(url_unique.values())
    if len(results) <= 1:
        return url_unique

    # ─── Step 2: 信源权威度评分映射 ───
    # 从域名/标题/内容中识别信源名称
    def _infer_source(result: dict) -> str:
        url = result.get("url", "")
        title = result.get("title", "")
        raw = result.get("raw_content", "") or result.get("content", "")
        text = (url + " " + title + " " + raw[:500]).lower()

        keywords = {
            "财联社": "cls.cn",
            "华尔街见闻": "wallstreetcn",
            "36氪": "36kr",
            "澎湃新闻": "thepaper",
            "新华社": "xinhuanet",
            "人民网": "people.com",
            "第一财经": "yicai",
            "界面新闻": "jie mian",
            "地方媒体": ["qq.com", "sohu.com", "163.com", "sina.com"],
            "AI生成摘要站": "toutiao",
        }
        for name, pattern in keywords.items():
            if isinstance(pattern, list):
                if any(p in url for p in pattern):
                    return name
            elif pattern in url:
                return name
        return "其他"

    AUTHORITY = {
        "新华社": 0.95, "财联社": 0.95, "澎湃新闻": 0.90,
        "华尔街见闻": 0.90, "人民网": 0.90, "第一财经": 0.85,
        "36氪": 0.85, "界面新闻": 0.85, "地方媒体": 0.50,
        "AI生成摘要站": 0.30, "其他": 0.40,
    }

    # ─── Step 3: 语义去重 ───
    # 用 content + title 做轻量级 embedding 指纹
    import hashlib
    sim_groups = []  # [(idx, source, content_hash, authority), ...]

    for i, r in enumerate(results):
        source = _infer_source(r)
        # 用哈希指纹做快速近似（比 embedding 快 1000 倍，项目不需要引入额外模型）
        text_sig = (r.get("content", "")[:300] + r.get("title", "")).encode("utf-8")
        content_hash = hashlib.md5(text_sig).hexdigest()[:8]
        authority = AUTHORITY.get(source, 0.4)
        sim_groups.append((i, source, content_hash, authority))

    # 按权威度降序排列，贪心选择
    sim_groups.sort(key=lambda x: x[3], reverse=True)

    seen_hashes = set()
    selected_indices = []
    for idx, source, chash, authority in sim_groups:
        if chash not in seen_hashes:
            # 不同的语义指纹 → 保留
            seen_hashes.add(chash)
            selected_indices.append(idx)
        else:
            # 语义指纹重复：
            # 高权威来源保留并标记 cross_verified
            # 低权威来源丢弃
            if authority >= 0.8:
                results[idx]["tag"] = "cross_verified"
                selected_indices.append(idx)

    unique_results = {}
    for idx in selected_indices:
        r = results[idx]
        unique_results[r["url"]] = r

    return unique_results


def process_search_results(unique_results: dict) -> dict:
    """ 处理搜索结果（对raw content做summary）

    Args:
        unique_results: url去重后的search results 

    Returns:
        做完summary后的results
    """
    summarized_results = {}

    for url, result in unique_results.items():
        # Use existing content if no raw content for summarization
        if not result.get("raw_content"):
            content = result['content']
        else:
            # Summarize raw content for better processing
            content = summarize_webpage_content(result['raw_content'][:MAX_CONTEXT_LENGTH])

        summarized_results[url] = {
            'title': result['title'],
            'content': content
        }

    return summarized_results


def format_search_output(summarized_results: dict) -> str:
    """对summarize后的结果做格式化（选择title, url, summary三个核心字段）

    Args:
        summarized_results: summarize后的results 

    Returns:
        格式化后的输出 
    """
    if not summarized_results:
        return "No valid search results found. Please try different search queries or use a different search API."

    import re as _re
    formatted_output = "Search results: \n\n"

    for i, (url, result) in enumerate(summarized_results.items(), 1):
        tag = result.get("tag", "")
        tag_str = " [交叉验证]" if tag == "cross_verified" else ""
        # 从 url 推断信源
        source = "未知"
        for name, pat in [("财联社","cls.cn"),("华尔街见闻","wallstreetcn"),
                          ("36氪","36kr"),("澎湃新闻","thepaper"),
                          ("新华社","xinhuanet"),("人民网","people.com"),
                          ("第一财经","yicai"),("界面新闻","jiemian")]:
            if pat in url:
                source = name
                break
        formatted_output += f"\n\n--- SOURCE {i}: {result['title']} [{source}]{tag_str} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "-" * 80 + "\n"

    return formatted_output


# ===== 共识摘要（补全 dedup 的 compress_redundant 依赖） =====


def _cluster_by_event(results: list) -> list[list]:
    """
    按事件对搜索结果聚类。
    轻量实现：取每条的 content[:200] + title 做 MD5 分组。
    同事件报道的前 200 字高度重叠，不需要 embedding。
    """
    import hashlib
    from collections import defaultdict

    clusters = defaultdict(list)
    for r in results:
        sig = r.get("content", "")[:200] + r.get("title", "")
        key = hashlib.md5(sig.encode("utf-8")).hexdigest()[:8]
        clusters[key].append(r)
    return list(clusters.values())


def _synthesize_consensus(cluster: list) -> str:
    """
    多来源共识合成。
    用 LLM 将同一事件的 N 篇报道压缩为：
      【事件共识】（N来源交叉验证）
      - 共识结论：...
      - 分歧点：...
      - 信源清单：...
    """
    from langchain_core.messages import HumanMessage
    global summarization_model

    sources = [r.get("source", "未知") for r in cluster]
    source_list = ", ".join(f"{s}[{i+1}]" for i, s in enumerate(sources))

    contents = "\n\n".join(
        f"--- {r.get('source', '未知')} ---\n{r.get('content', '')[:2000]}"
        for r in cluster
    )

    prompt = f"""以下 {len(cluster)} 篇报道在说同一件事，请按要求合成：

要求输出格式：
【事件共识】（{len(cluster)}来源交叉验证）
- 共识结论：<多篇报道一致同意的事实>
- 独家信息：<哪家报道有独家内容>
- 分歧点：<报道间不一致的说法>
- 信源清单：{source_list}

报道内容：
{contents}"""

    response = summarization_model.invoke([HumanMessage(content=prompt)])
    return getattr(response, "content", str(response))


def compress_redundant(results: list) -> str:
    """
    对去重后的结果做共识摘要。

    多条报道都在说同一件事 → 不用每条都塞进 Context，
    而是让 LLM 合成一段："财联社、华尔街见闻、新华社均报道了X，关键数据为Y"
    """
    clusters = _cluster_by_event(results)

    output = []
    for cluster in clusters:
        if len(cluster) == 1:
            r = cluster[0]
            output.append(r.get("full_content") or r.get("content", ""))
        else:
            output.append(_synthesize_consensus(cluster))

    return "\n\n".join(output)


# ===== RESEARCH TOOLS =====

def tavily_search(
    query: str,
    max_results: Annotated[Optional[int], InjectedToolArg] = None,
    topic: Annotated[Optional[Literal["general", "news", "finance"]], InjectedToolArg] = None,
) -> str:
    """根据配置好的tavily API去web上搜索结果，并返回做完网页内容摘要后的结果

    Args:
        query: 搜索关键词(query).
        max_results: 返回的最大结果数量(可选参数)，默认是3
        topic: 搜索主题，参数可以为：general, news, finance,（可选参数），默认是"general"

    Returns:
        格式化后并且去重+做完网页内容摘要的结果
    """
    # 获取默认参数
    _, _, defaults = _ensure_search_runtime()
    if defaults is None:
        raise SearchConfigError("Search defaults unavailable; search runtime not initialized")

    # 获取搜索参数
    resolved_max_results = max_results if max_results is not None else defaults.get("max_results", 3)
    resolved_topic = topic if topic is not None else defaults.get("topic", "general")
    include_raw_content = defaults.get("include_raw_content")
    if not include_raw_content:
        include_raw_content = True

    # 执行搜索并返回
    search_results = tavily_search_multiple(
        [query],  # 转换成列表
        max_results=resolved_max_results,
        topic=resolved_topic,
        include_raw_content=include_raw_content,
    )

    # 对搜索url去重
    unique_results = deduplicate_search_results(search_results)

    # 对网页长文档做摘要
    summarized_results = process_search_results(unique_results)

    # 格式化输出
    return format_search_output(summarized_results)


def think_tool(reflection: str) -> str:
    """用于对研究进展和决策进行策略反思的工具。
    每次搜索后，使用此工具分析结果并系统地规划下一步行动。这会在研究工作流程中故意暂停，以便进行质量决策。

    何时使用它:
    - 收到搜索结果后：我找到了哪些关键信息？
    - 在决定下一步之前：我的答案是否足够全面？
    - 在评估研究空白时：哪些关键信息仍然缺失？
    - 在结束研究之前：我现在能提供一个完整的答案吗？

    反思内容应该强调以下这些方面：
    1. 当前研究的分析 — 我收集到了哪些具体信息？
    2. 差距评估 - 还缺少哪些关键信息？
    3. 质量评估 - 我是否有足够的证据/例子来提供一个好的答案？
    4. 战略决策 - 我应该继续搜索还是给出答案？

    Args:
        reflection：您对研究进展、发现、存在的差距以及下一步行动的详细反思。

    Returns:
        记录反思的内容，以供智能体决策参考
    """

    return f"Reflection recorded: {reflection}"



def refine_draft_report(research_brief: Annotated[str, InjectedToolArg], 
                        findings: Annotated[str, InjectedToolArg], 
                        draft_report: Annotated[str, InjectedToolArg]):

    """根据新的研究发现(findings)完善目前的报告草稿(draft_report)

    该工具会综合当前所有研究结果整理输出一份更全面的报告草稿。

    Args:
        research_brief：用户的研究请求。
        findings：针对用户请求收集的研究结果。
        draft_report：基于研究结果和用户请求的报告草稿。

    Returns:
        精炼后的报告草案
    """

    # 组装提示词
    draft_report_prompt = REFINE_DRAFT_REPORT_PROMPT.format(
        research_brief=research_brief,
        findings=findings,
        draft_report=draft_report,
        date=get_today_str()
    )

    # 调用大模型来修正
    draft_report_obj = writer_model.invoke([HumanMessage(content=draft_report_prompt)])

    # 如果返回是message则抽取content字段，否则直接返回
    return getattr(draft_report_obj, "content", draft_report_obj)


# 注册成LangChain工具

# 搜索tool
_tavily_search_tool = tool(parse_docstring=True)(tavily_search)

# 反思tool
_think_tool = tool(parse_docstring=True)(think_tool)

# 精修tool
_refine_draft_report_tool = tool(parse_docstring=True)(refine_draft_report)

