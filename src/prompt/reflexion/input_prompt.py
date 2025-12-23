from ..base import PromptTemplate


reflexion_reflector = PromptTemplate(
    model='gpt-4o',
    temperature=0.0,
    input_variables=["instruction", "reflection_history", "trajectory"],
    template="""
Task Status: Task Failed. 
Objective: Generate a reflection on the Actor Agent's Action and Reflector's Reflection History by analyzing the Python code, execution logs, and reflection of those actions.

Requirements:
1. Pinpoint the exact failure points and logic errors in the code history.
2. Specify clear, actionable improvements for the next attempt.
3. Constraint: Do not use pronouns (e.g., 'it', 'this', 'that'). Refer to variables, functions, and logic by their specific names.
4. Constraint: Use brief, direct, and non-abstract sentences.

**Task**: 
{instruction}

**reflection history**: 
{reflection_history}

**trajectory**: 
{trajectory}
"""
)