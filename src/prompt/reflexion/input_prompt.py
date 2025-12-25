from ..base import PromptTemplate
from pathlib import Path

# -------------------------------------------------------------------------------------
# Reflector Module Input Prompt Templates in Reflexion Agent
# -------------------------------------------------------------------------------------
reflexion_reflector_input_prompt = PromptTemplate(
    model='gpt-4o',
    temperature=0.0,
    input_variables=["instruction", "reflection_history", "trajectory", "failure_report", "first_name", "last_name", "email", "phone_number"],
    template=Path(__file__).parent.joinpath("templates/reflector_input.txt").read_text(encoding="utf-8")
)

# -------------------------------------------------------------------------------------
# Reflector Module Input Prompt Templates with Ground Truth in Reflexion Agent
# -------------------------------------------------------------------------------------
reflexion_reflector_with_gt_input_prompt = PromptTemplate(
    model='gpt-4o',
    temperature=0.0,
    input_variables=["instruction", "reflection_history", "trajectory", "failure_report", "first_name", "last_name", "email", "phone_number", "ground_truth_api_calls", "ground_truth_apis", "ground_truth_apps", "ground_truth_code", "ground_truth_required_apis", "ground_truth_required_apps"],
    template=Path(__file__).parent.joinpath("templates/reflector_with_gt_input.txt").read_text(encoding="utf-8")
)

# -------------------------------------------------------------------------------------
# Actor Agent Input Prompt Templates for Reflexion Agent Actor Module Core
# -------------------------------------------------------------------------------------
reflexion_actor_input_prompt = PromptTemplate(
    model='gpt-4o',
    temperature=0.0,
    input_variables=['first_name', 'last_name', 'email', 'phone_number', 'instruction', 'reflection_history'],
    template=Path(__file__).parent.joinpath("templates/actor_input.txt").read_text(encoding="utf-8")
)