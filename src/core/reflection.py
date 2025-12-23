from .messages import AIMessage
from typing import List

class ReflectionHistory:
    def __init__(
        self,
        max_size:int = 3
    ):
        self.history = []
        self.max_size = max_size
    
    def add_reflection(
        self,
        messages: List[AIMessage]
    ) -> None:
        current_reflection = "<reflection>\n"
        for msg in messages:
            current_reflection += f"{msg.content}\n"
        current_reflection += "</reflection>\n\n"

        self.history.append(current_reflection)

        if len(self.history) > self.max_size:
            self.history.pop(0)
    
    def get_history(self) -> str:
        return "".join(self.history) if len(self.history) > 0 else "None"
    
    def reset(self) -> None:
        self.history = []