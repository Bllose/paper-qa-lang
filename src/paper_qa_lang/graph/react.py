"""通用 ReAct Agent Graph — 支持 LLM + Tool Calling 的基础图。

提供标准的 ReAct 循环:
1. 调用 LLM (需已 bind_tools)
2. 执行 tool_calls 并返回结果
3. 循环直到 LLM 输出无 tool_calls 的最终答案

用法::

    from langchain_mcp_adapters.tools import load_mcp_tools
    from paper_qa_lang.graph.react import build_react_graph

    tools = await load_mcp_tools(None, connection={...})
    graph = build_react_graph(llm.bind_tools(tools), tools)
    result = await graph.ainvoke({
        "messages": [HumanMessage(content="...")]
    })
    answer = result["messages"][-1].content
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


class ReActState(TypedDict):
    """ReAct 循环状态。

    Attributes:
        messages: 消息历史，会自动追加而非覆盖
        error:    节点执行错误信息（若有）
    """

    messages: Annotated[list[BaseMessage], add_messages]
    error: str | None


def build_react_graph(llm: BaseChatModel, tools: list[BaseTool]) -> Any:
    """构建并编译一个标准 ReAct Agent 图。

    图结构::

        [call_model] ─→ _router ── "tools" ──→ [tools] ──→ call_model
                              │
                              └── "end" ──→ [END]

    每次 ``call_model`` 节点返回带 ``tool_calls`` 的消息时，
    流程进入 ``tools`` 节点执行调用，结果以 ``ToolMessage``
    追加回状态，然后重新进入 ``call_model``。
    当 LLM 返回不带 ``tool_calls`` 的消息时，流程结束。

    Parameters
    ----------
    llm:
        已通过 ``bind_tools()`` 绑定了工具定义的 LLM 实例。
    tools:
        ``BaseTool`` 实例列表，用于实际执行工具调用。

    Returns
    -------
    编译后的 ``StateGraph`` (调用 ``ainvoke`` 执行)。
    """

    tool_map = {t.name: t for t in tools}

    # ── 辅助 ────────────────────────────────────────────

    def _has_tool_calls(msg: BaseMessage) -> bool:
        return hasattr(msg, "tool_calls") and bool(msg.tool_calls)

    # ── 节点 ────────────────────────────────────────────

    def _call_model(state: ReActState) -> dict:
        msgs = state.get("messages", [])
        if not msgs:
            return {"error": "No messages to process"}

        try:
            response = llm.invoke(msgs)
            logger.debug(
                "LLM responded: tool_calls=%d",
                len(getattr(response, "tool_calls", [])),
            )
            return {"messages": [response]}
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            # 返回一个无 tool_calls 的消息来终止循环
            return {
                "messages": [AIMessage(content=f"[Error] LLM call failed: {e}")],
                "error": f"LLM call error: {e}",
            }

    async def _call_tools(state: ReActState) -> dict:
        msgs = state.get("messages", [])
        if not msgs or not _has_tool_calls(msgs[-1]):
            return {"messages": []}

        last = msgs[-1]
        tool_msgs: list[ToolMessage] = []

        for tc in last.tool_calls:
            name = tc.get("name", "")
            args = tc.get("args", {})
            tid = tc.get("id", "")

            tool = tool_map.get(name)
            if tool is None:
                logger.warning("Unknown tool called: %s", name)
                tool_msgs.append(
                    ToolMessage(content=f"Error: unknown tool '{name}'", tool_call_id=tid)
                )
                continue

            try:
                result = await tool.ainvoke(args)
                content = str(result)
                logger.debug("Tool %s returned %d chars", name, len(content))
            except Exception as e:
                logger.error("Tool %s execution failed: %s", name, e)
                content = f"Tool '{name}' execution error: {e}"

            tool_msgs.append(ToolMessage(content=content, tool_call_id=tid))

        return {"messages": tool_msgs}

    # ── 路由 ────────────────────────────────────────────

    def _router(state: ReActState) -> Literal["tools", "end"]:
        if state.get("error"):
            return "end"
        msgs = state.get("messages", [])
        if msgs and _has_tool_calls(msgs[-1]):
            return "tools"
        return "end"

    # ── 装配 ────────────────────────────────────────────

    graph = StateGraph(ReActState)

    graph.add_node("call_model", _call_model)
    graph.add_node("tools", _call_tools)

    graph.set_entry_point("call_model")

    graph.add_conditional_edges("call_model", _router, {"tools": "tools", "end": END})

    graph.add_edge("tools", "call_model")

    return graph.compile()
