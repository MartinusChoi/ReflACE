import os
from typing import List, Dict, Any, Optional
from ..core.messages import (
    ChatMessageList, 
    ToolMessageList,
    ToolCallMessage,
    AIMessage
)

try:
    from openai import OpenAI
    from openai.types.responses.response import Response
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class OpenAIClient:
    def __init__(
        self, 
        model_name: str = "gpt-4o", 
        temperature: float = 0.0,
        tools: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.tools = tools
        self.system_prompt = system_prompt

        if not OPENAI_AVAILABLE:
            raise ImportError("⛔️ OpenAI module not found.")
        if not os.getenv("OPENAI_API_KEY"):
            raise UserWarning("⚠️ OpenAI API key not set.")
        
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    def get_response(
        self, 
        messages: List[Dict[str, str]]
    ) -> str:
        """
        Get a Response instance from the OpenAI API using the Response API.

        Args:
            messages (List[Dict[str, str]]): List of messages to send to the OpenAI API.

        Returns:
            message_list (ChatMessageList | ToolMessageList): The message list from the OpenAI API.
        """
        if not OPENAI_AVAILABLE:
            raise ValueError("⛔️ OpenAI module not found.")
        
        response: Response = self.client.responses.create(
            model=self.model_name,
            instructions=self.system_prompt,
            temperature=self.temperature,
            tools=self.tools,
            input=messages
        )
        
        tool_calls = [item for item in response.output if item.type == 'function_call']

        if not tool_calls:
            messages = []

            for output in response.output:
                if hasattr(output, 'content') and output.type == 'message':
                    for content in output.content:
                        if content.type == 'output_text':
                            messages.append(AIMessage(content=content.text))
            return ChatMessageList(messages)

        else:
            messages = []

            for tool_call in tool_calls:
                messages.append(
                    ToolCallMessage(
                        msg_type='function_call',
                        call_id=tool_call.call_id,
                        name=tool_call.name,
                        arguments=tool_call.arguments
                    )
                )
            return ToolMessageList(messages)