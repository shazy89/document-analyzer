from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from operator import add

from general_agent.config import GeneralAgentConfig
from general_agent.prompts import SYSTEM_PROMPT
from general_agent.schemas import UXAgentState
from general_agent.tools import GeneralAgentTools

_PROFILES_PATH = Path(__file__).parent / "data" / "profiles.json"


def _load_profiles() -> dict:
    if _PROFILES_PATH.exists():
        with _PROFILES_PATH.open() as f:
            return json.load(f)
    return {"profiles": {}}


def _save_profiles(data: dict) -> None:
    _PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _PROFILES_PATH.open("w") as f:
        json.dump(data, f, indent=2)


def profile_loader(state: UXAgentState) -> dict:
    """LangGraph node — load an existing profile or create one if missing."""
    user_id = state.get("user_id") or "anonymous"

    data = _load_profiles()

    if user_id not in data["profiles"]:
        data["profiles"][user_id] = {
            "user_id": user_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "preferences": {},
            "history_summary": "",
        }
        _save_profiles(data)
        logging.getLogger(__name__).info("profile_loader: created new profile user_id=%s", user_id)
    else:
        logging.getLogger(__name__).info("profile_loader: loaded profile user_id=%s", user_id)

    return {"user_profile": data["profiles"][user_id]}


logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    errors: Annotated[list[str], add]


class GeneralAgent:
    def __init__(
        self,
        config: GeneralAgentConfig,
        agent_tools: GeneralAgentTools,
    ) -> None:
        self.config = config
        self.system_prompt = SYSTEM_PROMPT
        self.agent_tools = agent_tools
        self.tools = list(agent_tools.tools.values())
        self.tools_by_name = {t.name: t for t in self.tools}
        self._llm = self._build_llm()
        self._graph = self.build_graph()

        logger.info(
            "GeneralAgent initialized (model=%s, tools=%s)",
            self.config.model_name,
            list(self.tools_by_name.keys()),
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> GeneralAgent:
        config = GeneralAgentConfig.from_env()
        agent_tools = GeneralAgentTools(config)
        return cls(config=config, agent_tools=agent_tools)

    def _build_llm(self):
        model = ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            api_key=self.config.api_key,
            base_url=self.config.api_base,
        )
        return model.bind_tools(self.tools) if self.tools else model

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, message: str, thread_id: str) -> str:
        """Send a message and return the assistant's reply.

        All calls sharing the same *thread_id* share conversation history.
        """
        config = {"configurable": {"thread_id": thread_id}}
        result = self._graph.invoke(
            {"messages": [("human", message)]},
            config=config,
        )
        return result["messages"][-1].content

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------

    def build_graph(self):
        builder  = StateGraph(UXAgentState)

        builder.add_node("profile_loader", profile_loader)
        builder.add_node("context_analyzer", context_analyzer)
        builder.add_node("question_decider", question_decider)
        builder.add_node("discovery_questions", discovery_questions)
        builder.add_node("ux_planner", ux_planner)
        builder.add_node("wireframe_generator", wireframe_generator)
        builder.add_node("ux_reviewer", ux_reviewer)
        builder.add_node("final_response", final_response)

        builder.set_entry_point("profile_loader")

        builder.add_edge("profile_loader", "context_analyzer")
        builder.add_edge("context_analyzer", "question_decider")

#         builder.add_conditional_edges(
#     "question_decider",
#         route_after_question_decider,
#     {
#         "discovery_questions": "discovery_questions",
#         "ux_planner": "ux_planner",
#     }
# )

        builder.add_edge("discovery_questions", END)

        builder.add_edge("ux_planner", "wireframe_generator")
        builder.add_edge("wireframe_generator", "ux_reviewer")
        builder.add_edge("ux_reviewer", "final_response")
        builder.add_edge("final_response", END)

        return builder.compile(checkpointer=MemorySaver())

    # ------------------------------------------------------------------
    # Node implementations
    # ------------------------------------------------------------------

    def _has_tool_calls(self, state: AgentState) -> bool:
        last = state["messages"][-1]
        return bool(getattr(last, "tool_calls", None))

    def _call_llm(self, state: AgentState) -> dict:
        messages = state["messages"]
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages
        response = self._llm.invoke(messages)
        return {"messages": [response]}

    def _execute_tool_calls(self, state: AgentState) -> dict:
        last_message = state["messages"][-1]
        tool_messages: list[ToolMessage] = []
        errors: list[str] = []

        for call in last_message.tool_calls:
            tool_name = call["name"]
            args = call.get("args", {})
            selected_tool = self.tools_by_name.get(tool_name)

            if not selected_tool:
                content = f"Unknown tool: {tool_name}"
                errors.append(content)
                tool_messages.append(
                    ToolMessage(content=content, name=tool_name, tool_call_id=call["id"])
                )
                continue

            try:
                result = selected_tool.invoke(args)
                tool_messages.append(
                    ToolMessage(
                        content=str(result),
                        name=tool_name,
                        tool_call_id=call["id"],
                    )
                )
            except Exception as exc:
                error_msg = f"Tool '{tool_name}' raised an error: {exc}"
                errors.append(error_msg)
                logger.exception("Tool execution failed: %s", tool_name)
                tool_messages.append(
                    ToolMessage(
                        content=error_msg,
                        name=tool_name,
                        tool_call_id=call["id"],
                    )
                )

        return {"messages": tool_messages, "errors": errors}


agent = GeneralAgent.from_env()

print(agent.run("My name is Erdoan.", thread_id="session-1"))
print(agent.run("What is my name?", thread_id="session-1"))  # remembers

print(agent.run("What is my name?", thread_id="session-2"))  # doesn't know