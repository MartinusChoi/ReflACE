import os
from typing import List, Dict, Any, Optional
from ..core.messages import (
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

        messages = []
        for output in response.output:
            if output.type == 'function_call':
                messages.append(
                    ToolCallMessage(
                        msg_type='function_call',
                        call_id=output.call_id,
                        name=output.name,
                        arguments=output.arguments
                    )
                )
            elif output.type == 'message':
                if hasattr(output, 'content') and output.type == 'message':
                    for content in output.content:
                        if content.type == 'output_text':
                            messages.append(AIMessage(content=content.text))
            else:
                raise ValueError(f"Unknown output type: {output.type}")
        
        return messages

