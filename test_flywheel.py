#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""批评飞轮离线自测（不需要 GPU / 不需要 API key）。

覆盖四块：解析、可用性门、奖励与优势、失败归因与导出。
运行：python test_flywheel.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from deep_research.flywheel import (
    Critique,
    Verification,
    build_groups,
    cluster_failures,
    critique_fingerprint,
    export_grpo,
    failure_signature,
    filter_critiques,
    group_variance_report,
    is_failure,
    load_export,
    normalize_advantages,
    parse_critiques,
    promote_to_regression,
    score_group,
    score_output,
    score_sample,
    severity_of,
    subitem_stats,
    summarize_doc,
)

PASSED = 0
FAILED = 0
FAILS: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  [OK]   {name}")
    else:
        FAILED += 1
        FAILS.append(name)
        print(f"  [FAIL] {name}  {extra}")


RAW_TWO = """【原文位置】 2.3 信息来源可信度评级 第 4 行
【问题】 2.3 来源可信度评级错误：把微博营销号评为"中"，与 3.2 的"高风险"结论不匹配
【解决方法】 将等级调整为"低"，并补充说明该账号为营销号转载

【原文位置】 1.2 关键时间线 第 2 行
【问题】 1.2 时间线节点与来源错配：时间写 8 月 6 日，来源是 8 月 9 日的报道
【解决方法】 将时间改为 8 月 9 日，或替换为当日发布的原始通报链接"""

DRAFT = (
    "## ① 舆情事件概述\n事件定性为公共危机事件。\n"
    "## ② 关键事实梳理\n公司股价在事件发生后下跌了百分之五，市场反应剧烈。\n"
    "## ③ 法律风险识别\n风险等级：高。\n"
)


# ===== 1. 解析 =====
print("\n=== 1. 解析红队输出 ===")
crits = parse_critiques(RAW_TWO)
check("解析出 2 条批评", len(crits) == 2, f"实际 {len(crits)}")
check("位置字段提取正确", crits[0].target_section.startswith("2.3 信息来源可信度评级"))
check("问题字段提取正确", "营销号" in crits[0].problem)
check("建议字段提取正确", "调整为" in crits[0].suggestion)
check("子项编号抽取 2.3", crits[0].subitem == "2.3", crits[0].subitem)
check("子项编号抽取 1.2", crits[1].subitem == "1.2", crits[1].subitem)
check("严重度由子项推导（2.3→major）", crits[0].severity == "major", crits[0].severity)

check("PASS 解析为空", parse_critiques("PASS") == [])
check("带围栏的 PASS 解析为空", parse_critiques("```\nPASS\n```") == [])
check("空输入解析为空", parse_critiques("") == [])

tag_spaced = "【 原文位置 】 3.2 风险等级\n【问题】 3.2 风险等级误判：已被官方澄清却判高风险\n【解决方法】 改为低"
c = parse_critiques(tag_spaced)
check("标签含空格仍可解析", len(c) == 1 and c[0].subitem == "3.2", str(len(c)))

no_fix = "【原文位置】 2.2\n【问题】 2.2 冲突点未标注，遗漏了反方说法"
c = parse_critiques(no_fix)
check("缺【解决方法】仍保留", len(c) == 1 and c[0].suggestion == "", str(len(c)))

qty = "【原文位置】 2.1\n【问题】 2.1 数字写错：原文 2.2亿被写成 22 亿\n【解决方法】 核对原文"
c = parse_critiques(qty)
check("数量 2.2亿 不被认成子项", c[0].subitem == "2.1", c[0].subitem)

check("严重度映射 2.1→critical", severity_of("2.1") == "critical")
check("未知子项回落 major", severity_of("9.9") == "major")

# ===== 2. 可用性门 =====
print("\n=== 2. 可用性门（trainable）===")
ok = filter_critiques(crits, DRAFT)
check("正常批评全部可用", len(ok.usable) == 2 and ok.trainable is True, str(len(ok.usable)))

r = filter_critiques([Critique(target_section="", problem="2.1 风险定性错误，把名誉权当成数据合规纠纷")], DRAFT)
check("无定位被拦", len(r.usable) == 0 and r.dropped[0].rule == "no_target", r.reason)

r = filter_critiques([Critique(target_section="全文", problem="2.1 风险定性错误，应当修正")], DRAFT)
check("含糊定位（全文）被拦", len(r.usable) == 0 and r.dropped[0].rule == "no_target")

r = filter_critiques([Critique(target_section="2.1", problem="建议加强论证")], DRAFT)
check("空话被拦", len(r.usable) == 0 and r.dropped[0].rule == "vague_problem", str(len(r.usable)))

r = filter_critiques([Critique(target_section="2.1", problem="补充 2.1 节的来源标注")], DRAFT)
check("含编号的短建议不判空话", len(r.usable) == 1, r.reason)

copied = "公司股价在事件发生后下跌了百分之五，市场反应剧烈"
r = filter_critiques([Critique(target_section="2.1", problem=copied)], DRAFT)
check("复述草稿被拦", len(r.usable) == 0 and r.dropped[0].rule == "restates_draft", r.reason)

judged = copied + "，但缺少官方通报来源，不能作为已确认事实"
r = filter_critiques([Critique(target_section="2.1", problem=judged)], DRAFT)
check("引用原句并做判断的不被误杀", len(r.usable) == 1, r.reason)

dup = [
    Critique(target_section="2.2", problem="2.2 冲突点未标注，遗漏反方说法"),
    Critique(target_section="2.2", problem="2.2 冲突点缺失，双方说法没有并列展示"),
]
r = filter_critiques(dup, DRAFT)
check("同批重复被拦", len(r.usable) == 1 and r.dropped[0].rule == "duplicate_critique", str(len(r.usable)))

r = filter_critiques([], DRAFT)
check("无批评则不可训", r.trainable is False and r.reason == "no_critique", r.reason)

r = filter_critiques([Critique(target_section="2.2", problem="2.2 冲突点未标注，遗漏了反方说法")], DRAFT)
check("一条可用即可训", r.trainable is True)

# 关键设计：误报不进过滤器，留给 reward 当负样本
mis = [Critique(target_section="2.2", problem="2.2 冲突点未标注，遗漏了反方说法")]
r = filter_critiques(mis, DRAFT)
check("误报不在这里被丢弃（交给 reward）", r.trainable is True and len(r.usable) == 1)

# ===== 3. 奖励与优势 =====
print("\n=== 3. 奖励与优势 ===")
c1 = Critique(target_section="2.1", problem="2.1 风险定性错误，把名誉权纠纷写成数据合规")
s = score_sample("p", [c1], [Verification(c1.critique_id, "valid")])
check("一条 critical 命中 = 1.0", abs(s.total - 1.0) < 1e-6, str(s.total))

s = score_sample("p", [c1], [Verification(c1.critique_id, "invalid")])
check("一条 critical 误报 = -1.0", abs(s.total + 1.0) < 1e-6, str(s.total))

c2 = Critique(target_section="3.3", problem="3.3 时效要求缺失，没有标注截止时间")
s = score_sample("p", [c2], [Verification(c2.critique_id, "valid")])
check("minor 命中 = 0.2", abs(s.total - 0.2) < 1e-6, str(s.total))

s = score_sample("p", [c1], [])
check("未核实按 unverifiable，1 条免罚", abs(s.total) < 1e-6, str(s.total))

two_unver = [
    Critique(target_section="2.1", problem="2.1 风险定性错误，依据法条不对应"),
    Critique(target_section="2.2", problem="2.2 冲突点未标注，漏了反方说法"),
]
s = score_sample("p", two_unver, [])
check("第 2 条不可验证开始扣 0.1", abs(s.total + 0.1) < 1e-6, str(s.total))

many = [
    Critique(target_section=f"2.{i % 3 + 1}", problem=f"2.{i % 3 + 1} 风险等级误判，与官方定性不一致 x{i}")
    for i in range(5)
]
s = score_sample("p", many, [Verification(c.critique_id, "valid") for c in many])
check("奖励截断到上界 4.0", abs(s.total - 4.0) < 1e-6, str(s.total))

adv = normalize_advantages([0.5, 0.5, 0.5, 0.5])
check("组内无方差 → 优势全 0", adv == [0.0, 0.0, 0.0, 0.0], str(adv))

adv = normalize_advantages([1.0, 2.0, 3.0, 4.0])
check("归一化均值约 0", abs(sum(adv)) < 1e-6, str(sum(adv)))
check("归一化标准差约 1", abs((sum(a * a for a in adv) / 4) ** 0.5 - 1.0) < 1e-3)

rep = group_variance_report({"a": [1, 1, 1, 1], "b": [0.2, 1.0, 0.5, 1.5]})
check("方差体检：1/2 组可用", rep["usable_groups"] == 1 and rep["groups"] == 2, str(rep))
check("方差体检给结论", "勉强" in rep["verdict"], rep["verdict"])

# ===== 4. 归因与导出 =====
print("\n=== 4. 失败归因与导出 ===")
bad = score_output(DRAFT, "【原文位置】 全文\n【问题】 建议加强论证")
check("不可用样本被判失败", is_failure(bad) is True)
check("签名带过滤原因", failure_signature(bad).startswith("filtered:"), failure_signature(bad))

_ai_raw = "【原文位置】 2.1\n【问题】 2.1 风险等级误判，与官方定性不一致\n【解决方法】 改为低"
_ai_crit = parse_critiques(_ai_raw)[0]
all_invalid = score_output(
    DRAFT, _ai_raw, [Verification(_ai_crit.critique_id, "invalid")]
)
check("全误报的样本也算失败", is_failure(all_invalid) is True)

good = score_output(
    DRAFT,
    RAW_TWO,
    [
        Verification(parse_critiques(RAW_TWO)[0].critique_id, "valid"),
        Verification(parse_critiques(RAW_TWO)[1].critique_id, "valid"),
    ],
)
check("正常样本不算失败", is_failure(good) is False, str(good.total))
check("正常样本奖励为正", good.total > 0, str(good.total))

clusters = cluster_failures([bad, bad, bad], min_count=2)
check("聚类达到阈值", len(clusters) == 1 and clusters[0].count == 3, str(len(clusters)))
check("聚类给分类", bool(clusters[0].suggested_category), clusters[0].suggested_category)
check("低于阈值不聚类", cluster_failures([bad], min_count=2) == [])

reg1 = promote_to_regression(clusters[0], DRAFT, "v1")
reg2 = promote_to_regression(clusters[0], DRAFT, "v1")
check("回归用例 ID 可复现", reg1["case_id"] == reg2["case_id"], str(reg1["case_id"]))
check("回归用例带期望", "min_valid_subitem" in reg1["expect"])

stats = subitem_stats([good])
check("子项统计有精度列", stats and "precision" in stats[0], str(stats[:1]))

g1 = score_group(DRAFT, [RAW_TWO, RAW_TWO])
g2 = score_group("另一份草稿内容，用于区分分组。", [RAW_TWO])
check("同一草稿的 k 条落同组", g1[0].prompt == g1[1].prompt)
check("不同草稿分到不同组", g1[0].prompt != g2[0].prompt)

groups = build_groups(g1 + g2)
check("聚成 2 个组", len(groups) == 2, str(len(groups)))

with tempfile.TemporaryDirectory() as td:
    path = Path(td) / "samples.json"
    doc = export_grpo(groups, path, min_group_size=2)
    check("导出文件已写入", path.exists())
    loaded = load_export(path)
    check("format 正确", loaded["format"] == "critique-grpo/v1", str(loaded.get("format")))
    check("组数正确（单样本组被跳过）", len(loaded["groups"]) == 1, str(len(loaded["groups"])))
    check("跳过组被记录", loaded["stats"]["skipped_groups"], str(loaded["stats"]))
    g0 = loaded["groups"][0]
    check("组内带 mean/std", "mean" in g0 and "std" in g0)
    check("优势数量与样本数一致", len(g0["advantages"]) == g0["n"], str(len(g0["advantages"])))
    check("统计含过滤原因", "filter_reasons" in loaded["stats"])
    check("摘要可打印", "critique-grpo/v1" in summarize_doc(loaded), summarize_doc(loaded))

    raw_json = json.loads(path.read_text(encoding="utf-8"))
    check("样本里不含 advantage 字段（归一化留给训练侧）",
          "advantage" not in raw_json["groups"][0]["samples"][0])

# ===== 5. 核实器 =====
print("\n=== 5. 核实器（规则层）===")
from deep_research.flywheel.verifier import (  # noqa: E402
    CombinedVerifier,
    RuleVerifier,
    split_sections,
)

SEC_DRAFT = """## ② 关键事实梳理
### 2.1 已确认事实
1. 华为于 8 月 4 日发布声明。[1]
### 2.3 信息来源可信度评级
| 来源类型 | 可信度 | 说明 |
| --- | --- | --- |
| 微博营销号转载 | **中** | 未经核实 |
| 主流媒体 | **高** | 权威 |

## ④ 法务处置建议
### 4.1 处置动作
证据保全、发律师函。
"""
secs = split_sections(SEC_DRAFT)
check("按编号切段", {"2.1", "2.3", "4.1"} <= set(secs), str(sorted(secs)))

rv = RuleVerifier(SEC_DRAFT)
check("子项 3.3（时效缺失）→ 判有错", rv.check("3.3").exists is True, str(rv.check("3.3")))
check("子项 2.3 无规则（需完整风险全集）", rv.check("2.3").exists is None)
check("子项 1.2 无规则（需跨源比对）", rv.check("1.2").exists is None)
check("章节号 4.1 不当子项用（旧 bug 回归）", rv.check("4.1").exists is None)
check("章节号 2.3 不当子项用（旧 bug 回归）", rv.check("2.3").exists is None)

with_time = SEC_DRAFT.replace("证据保全、发律师函。", "24h 内完成证据保全，72 小时内发律师函。")
check("4.x 有时效标签 → 判无缺失", RuleVerifier(with_time).check("3.3").exists is False)

date_draft = """### 1.2 关键时间线
| 时间 | 事件节点 | 来源 |
| --- | --- | --- |
| 2026年8月9日 | 媒体集中报道 | 2026-08-06 的报道 |
"""
check("1.2 日期倒置规则已下线（不再误判）",
      RuleVerifier(date_draft).check("1.2").exists is None,
      str(RuleVerifier(date_draft).check("1.2")))

c_valid = Critique(target_section="④ 法务处置建议 > 4.1 处置动作",
                   problem="3.3 时效要求缺失：4.x 全段没有截止时间")
v = RuleVerifier(SEC_DRAFT).verify(c_valid)
check("规则核实给出 valid", v.verdict == "valid" and v.verified_by == "rule", str(v.verdict))

c_unsup = Critique(target_section="② 关键事实梳理 > 2.1", problem="2.1 把传闻写进了已确认事实区")
v = RuleVerifier(SEC_DRAFT).verify(c_unsup)
check("判不了的给 unverifiable", v.verdict == "unverifiable", v.verdict)

cv = CombinedVerifier(SEC_DRAFT, judge=None)
vers = cv.verify_all([c_valid, c_unsup])
check("无 judge 时保持 unverifiable", [x.verdict for x in vers] == ["valid", "unverifiable"],
      str([x.verdict for x in vers]))
cov = cv.coverage(vers)
check("覆盖率算得对", cov["judged"] == 1 and abs(cov["coverage"] - 0.5) < 1e-9, str(cov))

# ===== 汇总 =====
print("\n" + "=" * 64)
print(f"结果：{PASSED}/{PASSED + FAILED} 通过")
if FAILS:
    print("失败项：")
    for f in FAILS:
        print(f"  - {f}")
print("=" * 64)
raise SystemExit(0 if FAILED == 0 else 1)
