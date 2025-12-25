from typing import List, Union
from .messages import BaseMessage, UserMessage, AIMessage, ToolCallMessage, ToolCallOutputMessage
import json

class Trajectory:
    def __init__(
        self,
        messages: List[BaseMessage | UserMessage | AIMessage | ToolCallMessage | ToolCallOutputMessage]
    ):
        self.messages = messages
    
    def append(
        self,
        message: Union[BaseMessage, UserMessage, AIMessage, ToolCallMessage, ToolCallOutputMessage]
    ):
        self.messages.append(message)
    
    def reset(self):
        self.messages = []
    
    def to_chat_prompt(self):
        return [msg.to_dict() for msg in self.messages]
    
    def to_str(self) -> str:
        trajectory_str = "<current trajectory>\n"
        for prompt in self.to_chat_prompt():
            if ('role' in prompt) and (prompt['role'] == 'user'):
                trajectory_str += "<User>\n"
                trajectory_str += f"{prompt['content']}\n"
                trajectory_str += "</User>\n"
            elif ('role' in prompt) and (prompt['role'] == 'assistant'):
                trajectory_str += "<Assistant>\n"
                trajectory_str += f"{prompt['content']}\n"
                trajectory_str += "</Assistant>\n"
            elif ('type' in prompt) and (prompt['type'] == 'function_call'):
                trajectory_str += "<Assistant>\n"
                trajectory_str += f"{json.loads(prompt['arguments'])['code']}\n"
                trajectory_str += "</Assistant>\n"
            elif ('type' in prompt) and (prompt['type'] == 'function_call_output'):
                trajectory_str += "<Environment>\n"
                trajectory_str += f"{prompt['output']}\n"
                trajectory_str += "</Environment>\n"
        trajectory_str += "</current trajectory>\n\n"
        return trajectory_str
    
    def __repr__(self):
        history = "\n\n".join([msg.__repr__() for msg in self.messages])
        return f"Trajectory(messages={history})"