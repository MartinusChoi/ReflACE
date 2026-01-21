from typing import Annotated, Sequence, List, Dict, Any, TypedDict
from operator import add

from langchain.messages import AnyMessage

from langgraph.graph.message import add_messages

from .core.playbook import PlayBook


# -----------------------------------------------------------------------------------------------------
# ReAct Agent State
# -----------------------------------------------------------------------------------------------------
class ReActState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    
    # field for track token usages
    input_tokens: Annotated[int, add]
    output_tokens: Annotated[int, add]
    total_tokens: Annotated[int, add]

    # field for gather latency for each nodes.
    latency: Annotated[float, add]

# -----------------------------------------------------------------------------------------------------
# Reflexion Agent State
# -----------------------------------------------------------------------------------------------------
class ReflexionState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

    # fields for reflexion agent
    trajectory: Sequence[AnyMessage]
    evaluation: str
    reflections: Annotated[Sequence[str], add]

    # field for track token usages
    input_tokens: Annotated[int, add]
    output_tokens: Annotated[int, add]
    total_tokens: Annotated[int, add]

    # field for gather latency for each nodes.
    latency: Annotated[float, add]

# -----------------------------------------------------------------------------------------------------
# ACE (Agentic Context Engineering) Agent State
# -----------------------------------------------------------------------------------------------------
class ACEState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

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

    # field for gather latency for each nodes.
    latency: Annotated[float, add]