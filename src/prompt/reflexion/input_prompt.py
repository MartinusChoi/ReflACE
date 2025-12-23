from ..base import PromptTemplate


reflexion_reflector = PromptTemplate(
    model='gpt-4o',
    temperature=0.0,
    input_variables=["instruction", "reflection_history", "trajectory", "success"],
    template="""
Please generate a reflection on the Actor Agent's Action and Reflector's Reflection History by analyzing the **Python code**, **execution logs**, and **reflection** of those actions.

**Task**: 
{instruction}

**Task Status**: 
{success}

**Reflection History**: 
{reflection_history}

**Trajectory (Code & Execution logs)**: 
{trajectory}
"""
)