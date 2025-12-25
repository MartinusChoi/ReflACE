from ..base import PromptTemplate
from pathlib import Path

# -------------------------------------------------------------------------------------
# ACE Reflector Model Input Prompt Templates
# -------------------------------------------------------------------------------------
ace_reflector_input_prompt = PromptTemplate(
    model="gpt-4o",
    temperature=0,
    input_variables=["instruction", "trajectory", "playbook"],
    template=Path(__file__).parent.joinpath('templates/reflector_input.txt').read_text()
)

# -------------------------------------------------------------------------------------
# ACE Generator(ReAct) Agent Input Prompt Templates
# -------------------------------------------------------------------------------------
ace_generator_input_prompt = PromptTemplate(
    model="gpt-4o",
    temperature=0,
    input_variables=['first_name', 'last_name', 'email', 'phone_number', 'instruction', 'playbook'],
    template=Path(__file__).parent.joinpath('templates/generator_input.txt').read_text()
)
