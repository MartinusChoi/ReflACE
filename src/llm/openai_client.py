import os
from typing import List, Dict, Any, Optional

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
        temperature: float = 0.0
    ):
        self.model_name = model_name
        self.temperature = temperature

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
            Response: The Response instance from the OpenAI API.
        """
        if not OPENAI_AVAILABLE:
            raise ValueError("⛔️ OpenAI module not found.")
        
        try:
            # Attempt to use the Responses API as requested
            try:
                response: Response = self.client.responses.create(
                    model=self.model_name,
                    temperature=self.temperature,
                    input=messages
                )
                
                # Parsing Output
                if hasattr(response, 'output'):
                    for output in response.output:
                        if hasattr(output, 'content'):
                            for content in output.content:
                                if hasattr(content, "text"):
                                    pass
                        if hasattr(item, 'type') and item.type == "message":
                             if isinstance(item.content, str):
                                 return item.content
                             elif isinstance(item.content, list):
                                 # Extract text content from blocks
                                 return "".join([
                                    c.text for c in item.content if hasattr(c, 'type') and c.type == "text"
                                    ])

            except AttributeError:
                # client.responses not found on this version of SDK
                pass
            
            # Fallback to standard chat.completions.create if responses API failed or was missing
            # or if we couldn't parse the result from it.
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                stop=stop
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return ""

    def get_embedding(
        self, 
        text: str
    ) -> List[float]:
        """
        Get embedding for a text string using text-embedding-3-small (default).

        Args:
            text (str): The text string to get the embedding for.

        Returns:
            List[float]: The embedding for the text string.
        """
        if not OPENAI_AVAILABLE:
            raise ValueError("⛔️ OpenAI module not found.")

        try:
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return []
