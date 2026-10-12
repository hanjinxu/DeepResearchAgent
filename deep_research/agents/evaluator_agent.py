#***********************************************
#      Filename: evaluator_agent.py
#   Description: 报告评估智能体 
#***********************************************

from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model

from deep_research.llm import get_chat_model
from deep_research.states import EvaluationResult 
from deep_research.prompts import DRAFT_EVALUATOR_PROMPT


# 初始化Judge Model 
judge_model = get_chat_model("evaluator")


def evaluate_draft_quality(research_brief: str, draft_report: str) -> EvaluationResult:
    """
    此函数实现了"self-evolution"评分机制。用另一个LLM作为评判者，以评估报告草稿相对于原始简报的质量。
    打分的分数和原因会返回给supervisor智能体。
    """

    # 组装prompt
    eval_prompt = DRAFT_EVALUATOR_PROMPT.format(
            research_brief = research_brief,
            draft_report = draft_report
    ) + "\n\n请以JSON格式返回，格式为：{\"comprehensiveness_score\": 0-10, \"accuracy_score\": 0-10, \"coherence_score\": 0-10, \"reason\": \"...\"}"

    # 直接调用模型，手动解析JSON
    response = judge_model.invoke([HumanMessage(content=eval_prompt)])
    import json, re
    text = response.content
    json_match = re.search(r'\{.*"comprehensiveness_score".*\}', text, re.DOTALL)
    if json_match:
        parsed = json.loads(json_match.group())
    else:
        parsed = {"comprehensiveness_score": 5, "accuracy_score": 5, "coherence_score": 5, "reason": text[:200]}
    
    return EvaluationResult(
        comprehensiveness_score=parsed.get("comprehensiveness_score", 5),
        accuracy_score=parsed.get("accuracy_score", 5),
        coherence_score=parsed.get("coherence_score", 5),
        reason=parsed.get("reason", str(text[:300]))
    )
