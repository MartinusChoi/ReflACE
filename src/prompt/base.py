from dataclasses import dataclass
from typing import List

# -------------------------------------------------------------------------------------
# Base Prompt Templates class
# -------------------------------------------------------------------------------------
@dataclass
class PromptTemplate:
    template: str
    input_variables: List[str] = None
    temperature: float = 0.0
    model: str = 'gpt-4o',