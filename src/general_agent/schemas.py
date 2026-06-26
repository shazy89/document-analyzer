from typing import Annotated, TypedDict, Optional, List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field
from typing import List
from typing import Annotated, Any, Dict, List, Optional, TypedDict
from operator import add

from langchain_core.messages import BaseMessage

ShortText = Annotated[str, Field(max_length=180)]
class SessionContext(BaseModel):
    product_name: Optional[ShortText] = Field(
        default=None,
        description="Known product name, if available."
    )
    domain: Optional[ShortText] = Field(
        default=None,
        description="Known product or business domain, if available."
    )
    known_target_users: List[ShortText] = Field(default_factory=list, max_length=5)
    known_workflows: List[ShortText] = Field(default_factory=list, max_length=5)
    ux_preferences: List[ShortText] = Field(default_factory=list, max_length=5)
    constraints: List[ShortText] = Field(default_factory=list, max_length=5)
    business_context: List[ShortText] = Field(default_factory=list, max_length=5)


class TaskContext(BaseModel):
    request_type: Optional[ShortText] = None
    product_area: Optional[ShortText] = None
    ux_goal: Optional[ShortText] = None
    target_user: Optional[ShortText] = None
    main_job: Optional[ShortText] = None
    known_requirements: List[ShortText] = Field(default_factory=list, max_length=6)
    assumptions: List[ShortText] = Field(default_factory=list, max_length=5)
    constraints: List[ShortText] = Field(default_factory=list, max_length=5)
    likely_friction: List[ShortText] = Field(default_factory=list, max_length=5)
    risks: List[ShortText] = Field(default_factory=list, max_length=5)
    success_criteria: List[ShortText] = Field(default_factory=list, max_length=5)


class QuestionDecision(BaseModel):
    should_ask_questions: bool = Field(
        default=False,
        description="Whether discovery questions should be asked before planning."
    )
    discovery_questions: List[ShortText] = Field(
        default_factory=list,
        max_length=5,
        description="Up to 5 targeted questions to resolve the most critical missing context."
    )
    missing_context: List[ShortText] = Field(
        default_factory=list,
        max_length=5,
        description="The specific missing context items that prompted these questions."
    )


class ContextAnalysis(BaseModel):
    session_context: SessionContext = Field(
        default_factory=SessionContext,
        description="Compact reusable context about the current UX session, product, user profile, and reusable information."
    )
    task_context: TaskContext = Field(
        default_factory=TaskContext,
        description="Compact structured analysis of the current user request only."
    )
    missing_context: List[ShortText] = Field(
        default_factory=list,
        max_length=5,
        description="Important missing information that could materially affect the UX design."
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence from 0 to 1 that enough context exists to continue with UX planning."
    )
class UXPlan(BaseModel):
    objective: str
    target_user: str
    assumptions: list[str]
    recommended_flow: list[str]
    page_structure: list[str]
    key_interactions: list[str]
    states_and_edge_cases: list[str]
    risks: list[str]
    next_steps: list[str] 

class UXAgentState(TypedDict):
    # Conversation
    messages: Annotated[list[BaseMessage], add_messages]

    # Identity / profile selection
    user_id: str
    profile_id: Optional[str]
    profile_name: Optional[str]

    # Loaded durable profile context
    profile_data: Dict[str, Any]

    # Temporary working context
    session_context: Dict[str, Any] 
    task_context: Dict[str, Any] # This is the current request after analysis.
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Discovery / control flow
    missing_context: List[str]
    should_ask_questions: bool
    discovery_questions: List[str]

    # Agent outputs
    ux_plan: UXPlan
    ui_plan: Dict[str, Any]

    # Final Copilot-ready output
    final_instructions: Optional[str]

    # Diagnostics
    errors: Annotated[list[str], add]
    
  
    


def create_initial_ux_state(
    *,
    message: str,
    user_id: str,
    profile_id: str,
) -> UXAgentState:
    if not message.strip():
        raise ValueError("message is required")

    if not user_id:
        raise ValueError("user_id is required")

    if not profile_id:
        raise ValueError("profile_id is required")

    return {
        "messages": [HumanMessage(content=message)],

        "user_id": user_id,
        "profile_id": profile_id,
        "profile_name": "",

        "profile_data": {},

        "session_context": {},
        "task_context": {},

        "missing_context": [],
        "should_ask_questions": False,
        "discovery_questions": [],

        "ux_plan": {},
        "ui_plan": {},

        "final_instructions": "",

        "errors": [],
    }    
