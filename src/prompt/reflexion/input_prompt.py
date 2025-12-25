from ..base import PromptTemplate


# -------------------------------------------------------------------------------------
# Reflector Input Prompt Templates in Reflexion Agent
# -------------------------------------------------------------------------------------
reflexion_reflector_input_prompt = PromptTemplate(
    model='gpt-4o',
    temperature=0.0,
    input_variables=["instruction", "reflection_history", "trajectory", "success", "first_name", "last_name", "email", "phone_number"],
    template="""
Please generate a reflection on the Actor Agent's Action Trajectory and your Reflection History by analyzing the **Python code**, **execution logs**, and **reflection** of those actions.

**Supervisor Information**:
first name : {first_name}
last name : {last_name}
email : {email}
phone number : {phone_number}

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



# -------------------------------------------------------------------------------------
# ReAct Agent Input Prompt Templates for Reflexion Agent Actor Module Core
# -------------------------------------------------------------------------------------
reflexion_actor_input_prompt = PromptTemplate(
    model='gpt-4o',
    temperature=0.0,
    input_variables=['first_name', 'last_name', 'email', 'phone_number', 'instruction', 'reflection_history'],
    template="""
Using these 'APIs' and 'reflection history' of your previous actions, now generate code to solve the actual task:

My name is {first_name} {last_name}. 
My personal email is {email} and phone number is {phone_number}.

**Task**:
{instruction}

**Reflection History**:
{reflection_history}
"""
)