#***********************************************
#      Filename: rubric.py
#   Description: 业务评分卡——28 条可验证检查项
#***********************************************

"""按写作模板派生的评分卡。

设计原则：**每一条都必须可判定**。判不了的条目不该进评分卡，否则它会变成
"感觉好像不太好"这种无法复现的评价。

判定方式分两类：
    rule  —— 结构/格式/取值合法性，用正则和表格解析就能判，零成本、零漏检
    model —— 需要语义理解（如"核心争议是否点明""动作是否具体"）

覆盖比例：28 条里 24 条是 rule。这也是这套设计的核心收益——
业务评审真正在意的"有没有漏项、取值是否正确"，绝大部分根本不需要模型。

条目编号与模板小节的对应关系写死在 CHECKS 里，改模板时必须同步改这里。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from deep_research.flywheel.verifier import _table_rows, split_sections


@dataclass
class Check:
    id: str
    section: str          # 对应模板小节，如 "1.1"
    desc: str
    kind: str             # "rule" | "model"


@dataclass
class CheckResult:
    check_id: str
    passed: Optional[bool]        # None = 规则判不了，需模型
    evidence: str = ""
    desc: str = ""

    @property
    def status(self) -> str:
        if self.passed is None:
            return "?"
        return "PASS" if self.passed else "FAIL"


CHECKS: List[Check] = [
    # ---- ① 舆情事件概述 ----
    Check("A1", "1.1", "事件性质/涉及主体/触发场景/传播规模 四字段齐全", "rule"),
    Check("A2", "1.1", "涉及主体三分类（涉事方/关联方/监管方）齐全", "rule"),
    Check("A3", "1.2", "时间线表格有来源列且每行非空", "rule"),
    Check("A4", "1.2", "时间线每行时间含年月日", "rule"),
    Check("A5", "1.1", "首发平台与首发时间有值", "rule"),
    # ---- ② 关键事实梳理 ----
    Check("B1", "2.1", "每条已确认事实以 ✅ 开头并带 来源：[n]", "rule"),
    Check("B2", "2.2", "每条未确认传闻以 ❓ 开头并标可信度", "rule"),
    Check("B3", "2.3", "来源评级表 ≥5 行，可信度取值合法", "rule"),
    Check("B4", "2.3", "自媒体/KOL/匿名类来源未被评「高」", "rule"),
    Check("B5", "2.1", "已确认事实中无「疑似/可能/据传/网传」", "rule"),
    Check("B6", "2.2", "每条传闻都有「冲突点」且非空", "rule"),
    Check("B7", "1.1", "核心争议节点已点明", "model"),
    Check("B8", "2.1", "同一件事不同时出现在 2.1 与 2.2", "model"),
    # ---- ③ 法律风险识别 ----
    Check("C1", "3.1", "八类法律风险均有判断（不适用可标 ❌）", "rule"),
    Check("C2", "3.1", "每类风险有 ✅/❌/⚠️ 判断标记", "rule"),
    Check("C3", "3.1", "判 ✅/⚠️ 的风险给出了依据", "rule"),
    Check("C4", "3.2", "风险等级表四列齐全，等级∈{高,中,低}", "rule"),
    Check("C5", "3.2", "每行「判定依据」非空", "rule"),
    Check("C6", "3.2", "3.2 的风险类型 ⊆ 3.1 中标 ✅/⚠️ 的类型", "rule"),
    # ---- ④ 法务处置建议 ----
    Check("D1", "4.1", "五类动作（证据保全/法律函件/监管沟通/声明修订/暂不介入）齐全", "rule"),
    Check("D2", "4.1", "每类动作给出了具体执行方案", "model"),
    Check("D3", "4.2", "优先级表四列齐全（动作/优先级/时效/责任方）", "rule"),
    Check("D4", "4.2", "每行有时效标签", "rule"),
    Check("D5", "4.2", "每行有责任方", "rule"),
    Check("D6", "3.2", "判「高」的风险在 4.x 有对应处置动作", "model"),
    Check("D7", "4.3", "对外策略三字段（公关策略/沟通节奏/内部联动）齐全", "rule"),
    # ---- 全局 ----
    Check("E1", "", "来源列表编号连续无空隙", "rule"),
    Check("E2", "", "正文 [n] 引用都能在来源列表找到", "rule"),
    Check("E3", "", "正文长度 ≤ 8000 字", "rule"),
    Check("E4", "", "无「我认为/笔者建议」等口语化表述", "rule"),
]

CHECK_MAP = {c.id: c for c in CHECKS}


# ===== 内容层：只有这 7 条需要模型读 =====
# 与上面 26 条规则项的分工：
#     规则层 —— 字段在不在、编号对不对、字数超没超。脚本判，零分歧。
#     内容层 —— 要读懂才能判（核心争议、动作是不是空话、引用撑不撑得住）。
# 把格式项塞给模型是浪费：实测 8B 在规则项上只有 76%，且错误集中在
# "规则按小节号找、模型按内容找"这种定义分歧上，吵这种架没有产出。
CONTENT_CHECKS: List[Check] = [
    # ---- ① 定性与主体 ----
    Check("G1", "1.1", "事件定性准确，性质判断无偏差", "model"),
    Check("G2", "1.1", "核心争议点已点明（不是只罗列事实）", "model"),
    Check("G3", "1.1", "涉及主体完整，监管方/关联方无关键遗漏", "model"),
    # ---- ② 事实与信源 ----
    Check("G4", "2.1", "已确认事实的信源达「≥2 家权威或官方」", "model"),
    Check("G5", "2.1", "同一件事没有同时出现在「已确认」与「未确认」", "model"),
    Check("G6", "2.2", "传闻的可信度标注与来源类型匹配", "model"),
    Check("G7", "2.2", "冲突双方的具体说法都已摆出，不是只写「存在矛盾」", "model"),
    Check("G8", "1.2", "时间线无因果倒置，首发/爆发/回应等关键节点无遗漏", "model"),
    Check("G9", "2.1", "数字、日期与前文或信源一致，无篡改", "model"),
    # ---- ③ 法律风险 ----
    Check("G10", "3.1", "风险定性与法条对应，无张冠李戴", "model"),
    Check("G11", "3.2", "风险等级与判定依据一致（依据说已立案不能判「中」）", "model"),
    Check("G12", "3.1", "判定依据有具体法条或事实，不是空喊「可能构成侵权」", "model"),
    Check("G13", "3.1", "衍生风险无重大遗漏（证券/消费者/合同等）", "model"),
    # ---- ④ 法务处置 ----
    Check("G14", "4.1", "每个动作写了谁做、做什么、怎么做，不是空标题", "model"),
    Check("G15", "3.2", "被判「高」的风险都有对应动作，且时效与等级匹配", "model"),
    Check("G16", "4.1", "法定动作无遗漏（如数据泄露须 72h 报监管）", "model"),
    Check("G17", "4.3", "对外口径与 ③ 的风险结论不矛盾", "model"),
    # ---- 跨节一致性 ----
    Check("G18", "", "1.1 的事件定性与 3.2 的风险等级不矛盾", "model"),
    Check("G19", "", "2.3 的来源评级与 3.1 的风险结论不矛盾", "model"),
    Check("G20", "", "正文每条 [n] 引用的来源确实支持那句话", "model"),
]

CONTENT_MAP = {c.id: c for c in CONTENT_CHECKS}


def build_content_checklist() -> str:
    """渲染给红队的内容检查表（7 条）。

    只给内容项。格式项由脚本判，模型看不到——这样它不会被
    "小节号对不对"这类它判不准、也不该判的问题干扰。
    """
    return "\n".join(f"- {c.id} {c.desc}" for c in CONTENT_CHECKS)

# 八类法律风险的关键词（模板 3.1）
RISK_KINDS = [
    ("名誉权", ("名誉权", "商业信誉")),
    ("隐私合规", ("隐私",)),
    ("证券信息披露", ("证券", "信息披露")),
    ("消费者保护", ("消费者", "虚假宣传")),
    ("知识产权", ("知识产权", "商标", "著作权", "商业秘密")),
    ("反不正当竞争", ("反不正当竞争", "商业诋毁")),
    ("平台合规", ("平台合规", "监管口径")),
    ("跨境合规", ("跨境",)),
]

ACTIONS = [
    ("证据保全", ("证据保全",)),
    ("法律函件", ("法律函件", "律师函", "告知函", "删除通知")),
    ("监管沟通", ("监管沟通", "报备")),
    ("声明修订", ("声明修订", "对外口径")),
    ("暂不介入", ("暂不介入",)),
]

VAGUE_WORDS = ("疑似", "可能", "据传", "网传", "传闻", "据说")

TIER_OK = ("高", "中", "低")
TIME_RE = re.compile(r"\d+\s*(?:h|H|小时|天|个工作日|工作日)|24h|48h|72h|立即|当日|持续")
DATE_RE = re.compile(r"20\d{2}\s*[年\-/.]\s*\d{1,2}")


def _lines(section: str) -> List[str]:
    return [ln.strip() for ln in (section or "").splitlines() if ln.strip()]


def _items(section: str) -> List[str]:
    """只取顶层列表项（缩进 == 0），不含缩进的子条目。

    踩过的坑：阈值写成 indent <= 2，而模板里子条目正好缩进 2 空格，
    结果 `  - **依据**：…` 也被当成顶层项，"每类风险是否有判断标记"
    把 8 个子条目全判成无标记，整条 FAIL。
    """
    out: List[str] = []
    for ln in (section or "").splitlines():
        if not ln.strip():
            continue
        indent = len(ln) - len(ln.lstrip())
        if indent == 0 and ln.strip().startswith(("-", "*", "•")):
            out.append(ln.strip())
    return out


def _blocks(section: str) -> List[tuple]:
    """把「顶层条目 + 其缩进子条目」切成块，返回 [(顶层行, 整块文本)]。

    像 3.1 这种「- **风险名**：✅ 触发」加缩进「  - **依据**：…」的结构，
    只看单行会误判"没有依据"，必须看整块。
    """
    blocks: List[tuple] = []
    cur_head: Optional[str] = None
    cur_body: List[str] = []

    for ln in (section or "").splitlines():
        if not ln.strip():
            continue
        indent = len(ln) - len(ln.lstrip())
        if indent == 0 and ln.strip().startswith(("-", "*", "•")):
            if cur_head is not None:
                blocks.append((cur_head, "\n".join(cur_body)))
            cur_head, cur_body = ln.strip(), [ln]
        elif cur_head is not None:
            cur_body.append(ln)

    if cur_head is not None:
        blocks.append((cur_head, "\n".join(cur_body)))
    return blocks


def _cols(section: str) -> List[str]:
    """表头列名。"""
    for ln in _lines(section):
        if ln.startswith("|"):
            return [c.strip() for c in ln.strip("|").split("|")]
    return []


def build_checklist() -> str:
    """把 CHECKS 渲染成红队 prompt 里的检查表。

    单一数据源：红队看到的就是这里生成的，改 CHECKS 等于同时改了 prompt 和校验器，
    不会出现"prompt 说查 30 条、校验器只认 26 条"这种错位。

    按小节聚合而不是按 CHECKS 顺序直接输出——CHECKS 里同一小节的条目不一定相邻
    （B5 属 2.1 但排在 B3/B4 之后），直接输出会让红队看到 2.2→2.3→2.1 这种乱序。
    """
    groups: Dict[str, List[Check]] = {}
    order: List[str] = []
    for c in CHECKS:
        key = c.section or "全局"
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(c)

    lines: List[str] = []
    for key in order:
        lines.append(f"\n### {key}")
        for c in groups[key]:
            lines.append(f"- {c.id} {c.desc}")
    return "\n".join(lines).strip()


def check_report(draft: str) -> List[CheckResult]:
    """跑完 28 条，返回每条结果（model 类返回 passed=None）。"""
    secs = split_sections(draft or "")
    body = draft or ""
    out: List[CheckResult] = []

    def add(cid: str, passed: Optional[bool], ev: str = "") -> None:
        out.append(CheckResult(cid, passed, ev, CHECK_MAP[cid].desc))

    s11, s12 = secs.get("1.1", ""), secs.get("1.2", "")
    s21, s22, s23 = secs.get("2.1", ""), secs.get("2.2", ""), secs.get("2.3", "")
    s31, s32 = secs.get("3.1", ""), secs.get("3.2", "")
    s41, s42, s43 = secs.get("4.1", ""), secs.get("4.2", ""), secs.get("4.3", "")

    # --- A1/A2/A5 ---
    need = ("事件性质", "涉及主体", "触发场景", "传播规模")
    miss = [k for k in need if k not in s11]
    add("A1", not miss, "缺：" + "、".join(miss) if miss else "四字段齐全")

    need2 = ("涉事方", "关联方", "监管方")
    miss2 = [k for k in need2 if k not in s11]
    add("A2", not miss2, "缺：" + "、".join(miss2) if miss2 else "三分类齐全")

    has_first = "首发平台" in s11
    has_time = bool(re.search(r"首发[^。\n]{0,10}(时间|[:：])", s11)) or "首发时间" in s11
    add("A5", has_first and has_time,
        "首发平台/时间不全" if not (has_first and has_time) else "已标注")

    # --- A3/A4 ---
    rows12 = _table_rows(s12)
    if not rows12:
        add("A3", False, "1.2 无表格")
        add("A4", False, "1.2 无表格")
    else:
        empty = [r for r in rows12 if len(r) < 3 or not r[2].strip() or r[2].strip() in ("—", "-")]
        add("A3", not empty, f"{len(empty)} 行来源为空" if empty else f"{len(rows12)} 行均有来源")
        bad_date = [r for r in rows12 if not DATE_RE.search(r[0] if r else "")]
        add("A4", not bad_date, f"{len(bad_date)} 行时间缺年月日" if bad_date else "时间格式正常")

    # --- B1/B2/B5/B6 ---
    # B1 只要求"带引用编号"，不强求模板里的"来源："前缀。
    # 真实报告写成 `…\[1\]\[2\]` 而没写"来源："——业务上等价，
    # 按字面判会把一份合格报告判成 8/8 全错，失去区分度。
    # 前缀缺失单独在 evidence 里提示，供改模板/改写作时参考。
    items21 = _items(s21)
    bad1 = [x for x in items21 if "✅" not in x or not re.search(r"\[\d+\]", x.replace("\\", ""))]
    has_prefix = any(re.search(r"来源[:：]", x) for x in items21)
    add("B1", bool(items21) and not bad1,
        f"{len(items21)} 条中 {len(bad1)} 条不符" if bad1
        else f"{len(items21)} 条均带引用" + ("" if has_prefix else "（未按模板写「来源：」前缀）"))

    items22 = _items(s22)
    bad2 = [x for x in items22 if "❓" not in x or "可信度" not in x]
    add("B2", bool(items22) and not bad2,
        f"{len(items22)} 条中 {len(bad2)} 条不符" if bad2 else f"{len(items22)} 条均合规")

    vague = [w for w in VAGUE_WORDS if w in s21]
    add("B5", not vague, "2.1 出现：" + "、".join(vague) if vague else "无模糊表述")

    no_conflict = [x for x in items22 if "冲突点" not in x]
    add("B6", bool(items22) and not no_conflict,
        f"{len(no_conflict)} 条缺冲突点" if no_conflict else "均有冲突点")

    # --- B3/B4 ---
    rows23 = _table_rows(s23)
    ok_rows = [r for r in rows23 if len(r) >= 2 and any(t in r[1] for t in TIER_OK)]
    add("B3", len(rows23) >= 5 and len(ok_rows) == len(rows23),
        f"{len(rows23)} 行，{len(rows23) - len(ok_rows)} 行取值不合法")

    self_media = ("自媒体", "KOL", "大V", "匿名", "网友", "知乎", "微博")
    # 刻意不含「百科」：维基/百度百科是聚合型来源，评为「中」属行业惯例，
    # 纳入会稳定误报（和 verifier 的判断保持一致）。
    too_high = [
        r[0] for r in rows23
        if len(r) >= 2 and any(h in r[0] for h in self_media)
        and any(t in r[1] for t in ("高", "中")) and "低" not in r[1]
    ]
    add("B4", not too_high, "误评偏高：" + "；".join(too_high[:2]) if too_high else "未发现误评")

    # --- C1/C2/C3 ---
    hit = [name for name, kws in RISK_KINDS if any(k in s31 for k in kws)]
    miss_risk = [n for n, _ in RISK_KINDS if n not in hit]
    add("C1", not miss_risk, "缺：" + "、".join(miss_risk) if miss_risk else f"{len(hit)} 类均覆盖")

    blocks31 = _blocks(s31)
    heads = [h for h, _ in blocks31]
    no_mark = [h for h in heads if not any(m in h for m in ("✅", "❌", "⚠️"))]
    add("C2", bool(heads) and not no_mark,
        f"{len(no_mark)} 条无判断标记" if no_mark else f"{len(heads)} 条均有标记")

    # 依据可能在子条目里，所以查整块而不是单行
    need_basis = [(h, b) for h, b in blocks31 if ("✅" in h or "⚠️" in h)]
    no_basis = [h for h, b in need_basis if "依据" not in b]
    add("C3", not no_basis, f"{len(no_basis)} 条缺依据" if no_basis else "均有依据")

    # --- C4/C5/C6 ---
    rows32 = _table_rows(s32)
    cols32 = _cols(s32)
    need_cols = ("风险类型", "等级", "依据", "后果")
    col_ok = all(any(n in c for c in cols32) for n in need_cols)
    tier_ok = all(any(t in r[1] for t in TIER_OK) for r in rows32 if len(r) >= 2)
    add("C4", bool(rows32) and col_ok and tier_ok,
        f"{len(rows32)} 行，列{'齐' if col_ok else '缺'}，等级{'合法' if tier_ok else '非法'}")

    no_basis32 = [r for r in rows32 if len(r) < 3 or not r[2].strip()]
    add("C5", not no_basis32, f"{len(no_basis32)} 行缺依据" if no_basis32 else "均有依据")

    # C6：3.2 的类型必须在 3.1 中标 ✅/⚠️
    # 名称要归一化再比：3.1 写「名誉权 / 商业信誉」，3.2 写「名誉权/商业信誉（对二创…）」，
    # 直接前缀匹配会因为空格、斜杠、括号后缀而失败。
    def _norm_name(s: str) -> str:
        return re.sub(r"[^\w\u4e00-\u9fff]", "", s or "")

    flagged = [_norm_name(h) for h, _ in blocks31 if any(m in h for m in ("✅", "⚠️"))]
    orphan = []
    for r in rows32:
        if not r:
            continue
        name = _norm_name(r[0])
        if not name:
            continue
        if not any(name.startswith(f[:4]) or f[:4] in name for f in flagged if f):
            orphan.append(r[0].strip("* 　")[:14])
    add("C6", not orphan,
        "3.2 有 3.1 未标风险的类型：" + "；".join(orphan[:2]) if orphan
        else f"{len(rows32)} 类都能在 3.1 找到对应")

    # --- D1/D3/D4/D5/D7 ---
    miss_act = [n for n, kws in ACTIONS if not any(k in s41 for k in kws)]
    add("D1", not miss_act, "缺：" + "、".join(miss_act) if miss_act else "五类齐全")

    rows42 = _table_rows(s42)
    cols42 = _cols(s42)
    need42 = ("动作", "优先级", "时效", "责任")
    col42_ok = all(any(n in c for c in cols42) for n in need42)
    add("D3", bool(rows42) and col42_ok,
        f"{len(rows42)} 行，列{'齐' if col42_ok else '缺'}")

    bad_time = [
        r[0] for r in rows42
        if len(r) < 3 or not TIME_RE.search(r[2] or "")
    ]
    add("D4", bool(rows42) and not bad_time,
        f"{len(bad_time)} 行缺时效" if bad_time
        else ("均有失效标签" if not rows42 else "均有时效标签"))

    bad_owner = [r[0] for r in rows42 if len(r) < 4 or not r[3].strip()]
    add("D5", bool(rows42) and not bad_owner,
        f"{len(bad_owner)} 行缺责任方" if bad_owner else "均有责任方")

    need43 = ("公关策略", "沟通节奏", "内部联动")
    miss43 = [k for k in need43 if k not in s43]
    add("D7", not miss43, "缺：" + "、".join(miss43) if miss43 else "三字段齐全")

    # --- E1/E2/E3/E4 ---
    src_block = body[body.find("来源列表"):] if "来源列表" in body else ""
    nums = [int(n) for n in re.findall(r"^\s*\[(\d+)\]", src_block, re.M)]
    if nums:
        continuous = nums == list(range(1, len(nums) + 1))
        add("E1", continuous,
            f"编号 {nums[:3]}…{'连续' if continuous else '有断号'}")
    else:
        add("E1", False, "找不到来源列表编号")

    cited = {int(n) for n in re.findall(r"\[(\d+)\]", body)}
    missing = sorted(cited - set(nums)) if nums else []
    add("E2", bool(nums) and not missing,
        f"[{','.join(map(str, missing[:4]))}] 在来源列表里找不到" if missing
        else "引用与来源列表一致")

    add("E3", len(body) <= 8000, f"{len(body)} 字" + ("（超限）" if len(body) > 8000 else ""))

    spoken = [w for w in ("我认为", "笔者建议", "我觉得") if w in body]
    add("E4", not spoken, "出现：" + "、".join(spoken) if spoken else "无口语化表述")

    # --- model 类，占位 ---
    add("B7", None, "需模型：核心争议是否点明")
    add("B8", None, "需模型：2.1 与 2.2 是否重叠")
    add("D2", None, "需模型：动作方案是否具体")
    add("D6", None, "需模型：高风险是否有对应动作")

    order = {c.id: i for i, c in enumerate(CHECKS)}
    out.sort(key=lambda r: order.get(r.check_id, 99))
    return out


def summarize(results: Sequence[CheckResult]) -> Dict[str, object]:
    rule = [r for r in results if r.passed is not None]
    passed = [r for r in rule if r.passed]
    failed = [r for r in rule if not r.passed]
    return {
        "total": len(results),
        "rule_checked": len(rule),
        "passed": len(passed),
        "failed": len(failed),
        "pass_rate": round(len(passed) / len(rule), 3) if rule else 0.0,
        "failed_ids": [r.check_id for r in failed],
        "model_pending": [r.check_id for r in results if r.passed is None],
    }


def compare_verdicts(verdicts: Sequence, draft: str) -> Dict[str, object]:
    """把红队的逐条判定与规则真值比对，算出可验证的准确率。

    这是新版红队的**奖励信号**：只统计"规则能判"的 26 条，
    红队在这 26 条上判对的比例就是 accuracy。
    剩下的 4 条（核心争议、2.1/2.2 重叠、动作具体性、高风险对应动作）
    规则判不了，不计入，避免用噪声污染奖励。

    三方对账比单一分数有用：
        agree     —— 红队判对
        disagree  —— 红队判错（能直接看出它哪类问题看不准）
        missing   —— 红队漏判（连条目都没写，说明没按格式输出）
        extra     —— 红队编了不存在的条目（ID 不在检查表里）
    """
    truth = {r.check_id: r.passed for r in check_report(draft) if r.passed is not None}
    by_id = {v.check_id: v for v in verdicts}

    agree, disagree = [], []
    for cid, t in truth.items():
        v = by_id.get(cid)
        if v is None or v.passed is None:
            continue
        (agree if v.passed == t else disagree).append(cid)

    missing = [cid for cid in truth if cid not in by_id or by_id[cid].passed is None]
    extra = [cid for cid in by_id if cid not in CHECK_MAP]
    judged = len(agree) + len(disagree)

    return {
        "judged": judged,
        "agree": len(agree),
        "disagree": len(disagree),
        "accuracy": round(len(agree) / judged, 3) if judged else 0.0,
        "agree_ids": agree,
        "disagree_ids": disagree,
        "missing_ids": missing,
        "extra_ids": extra,
        "ground_truth_pass": sum(1 for t in truth.values() if t),
        "ground_truth_total": len(truth),
    }


def _main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="按业务评分卡检查报告")
    ap.add_argument("draft")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    text = Path(args.draft).read_text(encoding="utf-8", errors="replace")
    results = check_report(text)
    s = summarize(results)

    print(f"{args.draft}｜{s['passed']}/{s['rule_checked']} 通过"
          f"（{s['pass_rate']:.0%}）｜待模型判 {len(s['model_pending'])} 条")
    print("-" * 72)
    for r in results:
        if r.passed is None and not args.verbose:
            continue
        mark = {"PASS": "✓", "FAIL": "✗", "?": "·"}[r.status]
        print(f" {mark} {r.check_id:<3} {r.desc[:44]:<46} {r.evidence[:40]}")
    if s["failed_ids"]:
        print(f"\n未通过：{', '.join(s['failed_ids'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
