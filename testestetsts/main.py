from typing import Literal
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage  # 引入消息类型
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict


# 1. 定义自定义工具
@tool
def get_weather(city):
    """查询指定城市的天气"""
    weather_data = {
        "北京": "晴",
        "上海": "多云",
        "广州": "小雨",
    }
    return weather_data.get(city, f"暂时无法查询 {city} 的天气")


tools = [get_weather]

# 2. 大模型绑定工具
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)


# 3. 状态
class State(TypedDict):
    messages: Annotated[list, add_messages]


# 4.  Prompt
def call_llm(state: State):
    system_prompt = SystemMessage(content="""
    你是一个专业、严谨且乐于助人的天气预报助手。
    你的职责是准确调用工具查询天气，并用简洁、亲切的中文回复用户。
    如果查询不到结果，请礼貌地告知用户并建议换个城市试试。
    """)
    messages_with_prompt = [system_prompt] + state["messages"]
    response = llm.invoke(messages_with_prompt)
    return {"messages": [response]}


# 5. 路由函数
def should_continue(state: State) -> Literal["tools", END]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# 6. 构建工作流图®
builder = StateGraph(State)
builder.add_node("agent", call_llm)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue)
builder.add_edge("tools", "agent")

graph = builder.compile()

if __name__ == "__main__":
    inputs = {"messages": [HumanMessage(content="帮我查查上海的天气怎么样？")]}


    final_response = graph.invoke(inputs)
    print("回复:", final_response["messages"][-1].content)