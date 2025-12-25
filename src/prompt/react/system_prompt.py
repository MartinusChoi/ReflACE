from ..base import PromptTemplate
from pathlib import Path

# -------------------------------------------------------------------------------------
# ReAct Agent System Prompt Templates for ReAct Agent
# -------------------------------------------------------------------------------------
react_system_prompt = PromptTemplate(
    model='gpt-4o',
    temperature=0.0,
    input_variables=[],
    template=Path(__file__).parent.joinpath("templates/react_system.txt").read_text(encoding='utf-8')
)