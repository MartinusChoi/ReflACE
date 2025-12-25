from ..base import PromptTemplate
from pathlib import Path

# -------------------------------------------------------------------------------------
# ReAct Agent Input Prompt Templates
# -------------------------------------------------------------------------------------
react_input_prompt = PromptTemplate(
    model='gpt-4o',
    temperature=0.0,
    input_variables=['first_name', 'last_name', 'email', 'phone_number', 'instruction'],
    template=Path(__file__).parent.joinpath("templates/react_input.txt").read_text(encoding='utf-8')
)