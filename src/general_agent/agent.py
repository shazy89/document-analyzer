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
from general_agent.schemas import UXAgentState, QuestionDecision
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
            "org_id": state.get("org_id"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "preferences": {},
            "history_summary": "",
        }
        _save_profiles(data)
        logging.getLogger(__name__).info("profile_loader: created new profile user_id=%s", user_id)
    else:
        logging.getLogger(__name__).info("profile_loader: loaded profile user_id=%s", user_id)

    return {"user_profile": data["profiles"][user_id]}


def context_analyzer(state: UXAgentState) -> dict:
    """LangGraph node — extract core UX context from the latest user message."""
    messages = state.get("messages", [])
    user_profile = state.get("user_profile") or {}
    preferences = user_profile.get("preferences") or {}

    last_user_message = ""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
            if isinstance(content, str):
                last_user_message = content
            elif isinstance(content, list):
                last_user_message = " ".join(
                    str(part) for part in content if isinstance(part, (str, dict))
                )
            else:
                last_user_message = str(content)
            break

    parsed_fields = _extract_context_fields(last_user_message)

    ux_goal = parsed_fields.get("ux_goal") or state.get("ux_goal") or preferences.get("ux_goal")
    target_user = (
        parsed_fields.get("target_user") or state.get("target_user") or preferences.get("target_user")
    )
    main_job = parsed_fields.get("main_job") or state.get("main_job") or preferences.get("main_job")

    # Keep this node focused: set request and normalize context fields expected downstream.
    user_request = last_user_message.strip()
    missing_context = [
        field_name
        for field_name, value in {
            "ux_goal": ux_goal,
            "target_user": target_user,
            "main_job": main_job,
        }.items()
        if not value
    ]

    return {
        "user_request": user_request,
        "ux_goal": ux_goal,
        "target_user": target_user,
        "main_job": main_job,
        "user_flow": state.get("user_flow") or [],
        "friction_points": state.get("friction_points") or [],
        "constraints": state.get("constraints") or [],
        "missing_context": missing_context,
    }


def _extract_context_fields(text: str) -> dict[str, str]:
    """Extract key context fields from a free-form user message."""
    patterns = {
        "ux_goal": r"(?im)^\s*(?:ux_goal|goal)\s*:\s*(.+?)\s*$",
        "target_user": r"(?im)^\s*(?:target_user|user|audience)\s*:\s*(.+?)\s*$",
        "main_job": r"(?im)^\s*(?:main_job|job|task)\s*:\s*(.+?)\s*$",
    }
    extracted: dict[str, str] = {}

    for field_name, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            extracted[field_name] = match.group(1).strip()

    return extracted


def profile_updater(state: UXAgentState) -> dict:
    """Persist context fields extracted from the latest user turn into profile data."""
    user_profile = state.get("user_profile") or {}
    if not user_profile:
        return {}

    preferences = dict(user_profile.get("preferences") or {})
    updated = False
    for key in ("ux_goal", "target_user", "main_job"):
        value = state.get(key)
        if isinstance(value, str) and value.strip():
            normalized = value.strip()
            if preferences.get(key) != normalized:
                preferences[key] = normalized
                updated = True

    if not updated:
        return {}

    data = _load_profiles()
    user_id = state.get("user_id") or user_profile.get("user_id") or "anonymous"
    existing_profile = data["profiles"].get(user_id, user_profile)
    existing_profile["preferences"] = preferences
    existing_profile["updated_at"] = datetime.now(timezone.utc).isoformat()
    data["profiles"][user_id] = existing_profile
    _save_profiles(data)

    return {"user_profile": existing_profile}


def question_decider(state: UXAgentState) -> dict:
    """Decide if more context is needed before planner can proceed."""
    missing_context = state.get("missing_context") or []
    should_ask_questions = len(missing_context) > 0

    question_map = {
        "ux_goal": "What is the UX goal for this request?",
        "target_user": "Who is the target user for this feature/design?",
        "main_job": "What is the main user job-to-be-done?",
    }
    discovery_questions = [question_map[field] for field in missing_context if field in question_map]

    return {
        "should_ask_questions": should_ask_questions,
        "missing_context": missing_context,
        "discovery_questions": discovery_questions,
    }


def route_after_question_decider(state: UXAgentState) -> str:
    if state.get("should_ask_questions"):
        return "discovery_questions"
    return "ux_planner"


def discovery_questions(state: UXAgentState) -> dict:
    """Ask user only for missing context fields."""
    questions = state.get("discovery_questions") or []
    if not questions:
        questions = ["Can you share more context for your request?"]

    prompt = (
        "I need a bit more context before planning.\n\n"
        + "\n".join(f"- {question}" for question in questions)
        + "\n\n"
        + "Reply in this format:\n"
        + "ux_goal: ...\n"
        + "target_user: ...\n"
        + "main_job: ..."
    )
    return {"messages": [AIMessage(content=prompt)]}


def ux_planner(state: UXAgentState) -> dict:
    """Minimal planner handoff when required context is present."""
    final_answer = (
        "Great, I have enough context for planning.\n"
        f"- UX goal: {state.get('ux_goal') or 'N/A'}\n"
        f"- Target user: {state.get('target_user') or 'N/A'}\n"
        f"- Main job: {state.get('main_job') or 'N/A'}"
    )
    return {"final_answer": final_answer}


def final_response(state: UXAgentState) -> dict:
    """Return final planner-facing response to the user."""
    content = state.get("final_answer") or "I could not build a plan response yet."
    return {"messages": [AIMessage(content=content)]}


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
        self.system_prompt = UX_SYSTEM_PROMPT
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
        builder.add_node("profile_updater", profile_updater)
        builder.add_node("question_decider", question_decider)
        builder.add_node("discovery_questions", discovery_questions)
        builder.add_node("ux_planner", ux_planner)
        builder.add_node("final_response", final_response)

        builder.add_edge(START, "profile_loader")
        builder.add_edge("profile_loader", "context_analyzer")
        builder.add_edge("context_analyzer", "profile_updater")
        builder.add_edge("profile_updater", "question_decider")

        builder.add_conditional_edges(
            "question_decider",
            route_after_question_decider,
            {
                "discovery_questions": "discovery_questions",
                "ux_planner": "ux_planner",
            },
        )

        builder.add_edge("discovery_questions", END)
        builder.add_edge("ux_planner", "final_response")
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
    agent = GeneralAgent.from_env()
    print(agent.run("My name is Erdoan.", thread_id="session-1"))
