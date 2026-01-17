from typing import Annotated, Sequence
from operator import add

from langchain.agents.middleware import AgentState
from langchain.messages import AnyMessage


# -----------------------------------------------------------------------------------------------------
# Agent State
# -----------------------------------------------------------------------------------------------------
class ReActState(AgentState):
    # field for track token usages
    input_tokens: Annotated[int, add]
    output_tokens: Annotated[int, add]
    total_tokens: Annotated[int, add]


class ReflexionState(AgentState):
    # fields for reflexion agent
    trajectory: Sequence[AnyMessage]
    evaluation: str
    reflections: Annotated[Sequence[str], add]

    # field for track token usages
    input_tokens: Annotated[int, add]
    output_tokens: Annotated[int, add]
    total_tokens: Annotated[int, add]