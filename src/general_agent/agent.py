from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages

from operator import add

from general_agent.config import GeneralAgentConfig
from general_agent.prompts import SYSTEM_PROMPT, UX_SYSTEM_PROMPT
from general_agent.schemas import UXAgentState, QuestionDecision, create_initial_ux_state
from general_agent.tools import GeneralAgentTools

_PROFILES_PATH = Path(__file__).parent / "data" / "profiles.json"


def _load_profiles() -> dict:
    if _PROFILES_PATH.exists():
        with _PROFILES_PATH.open() as f:
            return json.load(f)
    return {"profiles": []}


def profile_loader(state: UXAgentState) -> dict:
    """LangGraph node — load an existing profile."""
    user_id = state.get("user_id")
    profile_id = state.get("profile_id")

    data = _load_profiles()
    
    profile = next((p for p in data["profiles"] if p["owner_user_id"] == user_id and p["id"] == profile_id), None)

    if profile is None:
        return {"profile_data": {}}
    
    return {"profile_data": profile}



def question_decider(state: UXAgentState) -> dict:
    """Decide if more context is needed before planner can proceed."""
  
    return None  # TODO: implement this node using the QUESTION_DECIDER_PROMPT and the QuestionDecision schema


def route_after_question_decider(state: UXAgentState) -> str:
    if state.get("should_ask_questions"):
        return "discovery_questions"
    return "ux_planner"



def final_response(state: UXAgentState) -> dict:
    """Return final planner-facing response to the user."""
    content = state.get("profile_data") or "I could not build a plan response yet."
    if isinstance(content, dict):
        content = json.dumps(content)
    return {"messages": [AIMessage(content=content)]}


logger = logging.getLogger(__name__)



class DesignerAgent:
    def __init__(
        self,
        config: GeneralAgentConfig,
        agent_tools: GeneralAgentTools,
    ) -> None:
        self.config = config
        self.system_prompt = UX_SYSTEM_PROMPT
        self.agent_tools = agent_tools
        self.tools = list(agent_tools.tools.values())
        self.tools_by_name = {t.name: t for t in self.tools}
        self._llm = self._build_llm()
        self._graph = self.build_graph()

        logger.info(
            "DesignerAgent initialized (model=%s, tools=%s)",
            self.config.model_name,
            list(self.tools_by_name.keys()),
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> DesignerAgent:
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

    def run(
        self,
        message: str,
        thread_id: str,
        user_id: str,
        profile_id: str,
        is_new_session: bool = False,
    ) -> str:
        config = {"configurable": {"thread_id": thread_id}} 

        if is_new_session:
            graph_input = create_initial_ux_state(
                message=message,
                user_id=user_id,
                profile_id=profile_id,
            )
        else:
            graph_input = {
                "messages": [HumanMessage(content=message)],
                "user_id": user_id,
                "profile_id": profile_id,
            }   

        result = self._graph.invoke(graph_input, config=config)

        return result.get("final_instructions") or result["messages"][-1].content

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------

    def build_graph(self):
        builder  = StateGraph(UXAgentState)
        
        builder.add_node("profile_loader", profile_loader)
        # builder.add_node("context_analyzer", context_analyzer)
        # builder.add_node("profile_updater", profile_updater)
        # builder.add_node("question_decider", question_decider)
        # builder.add_node("discovery_questions", discovery_questions)
        # builder.add_node("ux_planner", ux_planner)
        builder.add_node("final_response", final_response)

        builder.add_edge(START, "profile_loader")
        builder.add_edge("profile_loader", "final_response")
        #builder.add_edge("context_analyzer", "profile_updater")
        #builder.add_edge("profile_updater", "question_decider")

        #builder.add_conditional_edges(
        #    "question_decider",
        #    route_after_question_decider,
        #    {
        #        "discovery_questions": "discovery_questions",
        #        "ux_planner": "ux_planner",
        #    },
        #)

        #builder.add_edge("discovery_questions", END)
        #builder.add_edge("ux_planner", "final_response")
        builder.add_edge("final_response", END)

        return builder.compile(checkpointer=MemorySaver())

    # ------------------------------------------------------------------
    # Node implementations
    # ------------------------------------------------------------------

    def _has_tool_calls(self, state: UXAgentState) -> bool:
        last = state["messages"][-1]
        return bool(getattr(last, "tool_calls", None))

    def _call_llm(self, state: UXAgentState) -> dict:
        messages = state["messages"]
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages
        response = self._llm.invoke(messages)
        return {"messages": [response]}
    
    def _context_analyzer(self, state: UXAgentState) -> dict:
        profile = state.get("profile_data", {})
        session_context = {
            "current_profile": profile.get("name", ""),
            "description": profile.get("description", ""),
            "target_users": profile.get("target_users", []),
            "core_use_cases": profile.get("core_use_cases", []),
        }  

    def _execute_tool_calls(self, state: UXAgentState) -> dict:
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


if __name__ == "__main__":
    agent = DesignerAgent.from_env()
    print(agent.run("Get Me the users data", thread_id="session-1", user_id="user_001", profile_id="profile_001", is_new_session=True))
