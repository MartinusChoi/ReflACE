from ..base import PromptTemplate
from pathlib import Path

# -------------------------------------------------------------------------------------
# ACE Reflector Model System Prompt Templates
# -------------------------------------------------------------------------------------
ace_reflector_system_prompt = PromptTemplate(
    model="gpt-4o",
    temperature=0,
    input_variables=[],
    template=Path(__file__).parent.joinpath('templates/reflector_system.txt').read_text()
)

# -------------------------------------------------------------------------------------
# ACE Generator(ReAct) Model System Prompt Templates
# -------------------------------------------------------------------------------------
ace_generator_system_prompt = PromptTemplate(
    model='gpt-4o',
    temperature=0.0,
    input_variables=None,
    template=Path(__file__).parent.joinpath('templates/generator_system.txt').read_text()
)