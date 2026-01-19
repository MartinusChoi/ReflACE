from typing import Annotated, Sequence, Dict
from operator import add

from langchain.agents.middleware import AgentState
from langchain.messages import AnyMessage


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
    playbook: Dict[str, Dict[str, str]]
    reflector_output: Dict[str, str]
    curator_output: Dict[str, str]


    # field for track token usages
    input_tokens: Annotated[int, add]
    output_tokens: Annotated[int, add]
    total_tokens: Annotated[int, add]