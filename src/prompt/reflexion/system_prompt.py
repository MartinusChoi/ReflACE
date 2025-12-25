from ..base import PromptTemplate
from pathlib import Path

# -------------------------------------------------------------------------------------
# Reflector Module System Prompt Template in Reflexion Agent
# -------------------------------------------------------------------------------------
reflexion_reflector_system_prompt = PromptTemplate(
    model='gpt-4o',
    temperature=0.0,
    input_variables=[],
    template=Path(__file__).parent.joinpath("templates/reflector_system.txt").read_text(encoding="utf-8")
)

# -------------------------------------------------------------------------------------
# Reflector Module System Prompt Template with Ground Truth Information in Reflexion Agent
# -------------------------------------------------------------------------------------
reflexion_reflector_with_gt_system_prompt = PromptTemplate(
  model='gpt-4o',
  temperature=0.0,
  input_variables=[],
  template=Path(__file__).parent.joinpath("templates/reflector_with_gt_system.txt").read_text(encoding="utf-8")
)

# -------------------------------------------------------------------------------------
# Actor Agent System Prompt Template in Reflexion Agent
# -------------------------------------------------------------------------------------
reflexion_actor_system_prompt = PromptTemplate(
    model='gpt-4o',
    temperature=0.0,
    input_variables=[],
    template=Path(__file__).parent.joinpath("templates/actor_system.txt").read_text(encoding="utf-8")
)