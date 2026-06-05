from typing import TypedDict, Optional, List, Dict, Any
from pydantic import BaseModel, Field
from typing import List


class QuestionDecision(BaseModel):
    should_ask_questions: bool
    missing_context: List[str] = Field(default_factory=list)
    discovery_questions: List[str] = Field(default_factory=list)


class UXAgentState(TypedDict):
    user_request: str
    user_id: str  # used to look up / create the profile in profiles.json

    # Reusable memory/profile
    user_profile: Dict[str, Any]
    product_profile: Dict[str, Any]

    # Current task understanding
    ux_goal: Optional[str]
    target_user: Optional[str]
    main_job: Optional[str]
    user_flow: Optional[List[str]]
    friction_points: Optional[List[str]]
    constraints: Optional[List[str]]

    # Control flow
    missing_context: List[str]
    should_ask_questions: bool
    discovery_questions: List[str]

    # Outputs
    ux_recommendations: Optional[List[str]]
    wireframe: Optional[str]
    interaction_behavior: Optional[List[str]]
    validation_notes: Optional[List[str]]
    final_answer: Optional[str]