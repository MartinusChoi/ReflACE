from ..core.trajectory import Trajectory
from ..core.messages import AIMessage

def is_agent_finished(trajectory: Trajectory) -> bool:
    if isinstance(trajectory.messages[-1], AIMessage) or "complete_task" in trajectory.messages[-1].content:
        return True
    
    return False