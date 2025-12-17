import os
from typing import List, Dict, Any, Optional

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: 'openai' module not found. Using Mock Client.")

class OpenAIClient:
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = os.getenv("OPENAI_API_KEY")
        if OPENAI_AVAILABLE and self.api_key:
             openai.api_key = self.api_key

    def chat_completion(self, messages: List[Dict[str, str]], stop: Optional[List[str]] = None) -> str:
        """
        Get a completion from the OpenAI API.
        """
        if not OPENAI_AVAILABLE or not self.api_key:
            return self._mock_completion(messages)

        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                stop=stop
            )
            return response.choices[0].message['content']
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return ""

    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a text string using text-embedding-3-small (default).
        """
        if not OPENAI_AVAILABLE or not self.api_key:
            return [0.0] * 1536 # Mock embedding

        try:
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response['data'][0]['embedding']
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return []

    def _mock_completion(self, messages):
        """Mock responses for testing agent loops."""
        last_msg = messages[-1]['content']
        if "Task:" in last_msg or "history" in last_msg.lower():
            # Likely an action generation prompt
            return "Thought: I should eat the apple.\nAction: eat apple"
        elif "Analyze the following" in last_msg:
             # Reflection or Insight generation
             return "- Always check your surroundings.\n- Verify prerequisites before action."
        return "Thought: I don't know.\nAction: look"
