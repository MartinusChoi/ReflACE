from typing import Annotated, Sequence, Dict, Any
from operator import add

from langchain.agents.middleware import AgentState
from langchain.messages import AnyMessage

from .core.playbook import PlayBook


# -----------------------------------------------------------------------------------------------------
# ReAct Agent State
# -----------------------------------------------------------------------------------------------------
class ReActState(AgentState):
    # field for track token usages
    input_tokens: Annotated[int, add]
    output_tokens: Annotated[int, add]
    total_tokens: Annotated[int, add]

# -----------------------------------------------------------------------------------------------------
# Reflexion Agent State
# -----------------------------------------------------------------------------------------------------
class ReflexionState(AgentState):
    # fields for reflexion agent
    trajectory: Sequence[AnyMessage]
    evaluation: str
    reflections: Annotated[Sequence[str], add]

    # field for track token usages
    input_tokens: Annotated[int, add]
    output_tokens: Annotated[int, add]
    total_tokens: Annotated[int, add]

# -----------------------------------------------------------------------------------------------------
# ACE (Agentic Context Engineering) Agent State
# -----------------------------------------------------------------------------------------------------
class ACEState(AgentState):
    # field for ace agent
    trajectory: Sequence[AnyMessage]
    evaluation: str
    playbook: PlayBook
    reflection: Dict[str, Any]
    curation: Dict[str, Any]


    # field for track token usages
    input_tokens: Annotated[int, add]
    output_tokens: Annotated[int, add]
    total_tokens: Annotated[int, add]