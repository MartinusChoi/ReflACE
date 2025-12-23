from ..base import PromptTemplate

# -------------------------------------------------------------------------------------
# ReAct Agent Input Prompt Templates
# -------------------------------------------------------------------------------------
react_agent = PromptTemplate(
    template="""
Using these APIs, now generate code to solve the actual task:

My name is {first_name} {last_name}. 
My personal email is {email} and phone number is {phone_number}.

**Task**:
{instruction}
""",
    input_variables=['first_name', 'last_name', 'email', 'phone_number', 'instruction']
) 

# -------------------------------------------------------------------------------------
# ReAct Agent Input Prompt Templates for Reflexion Agent Actor Module Core
# -------------------------------------------------------------------------------------
reflexion_actor = PromptTemplate(
    template="""
Using these 'APIs' and 'reflection history' of your previous actions, now generate code to solve the actual task:

My name is {first_name} {last_name}. 
My personal email is {email} and phone number is {phone_number}.

**Task**:
{instruction}

**Reflection History**:
{reflection_history}
""",
    input_variables=['first_name', 'last_name', 'email', 'phone_number', 'instruction', 'reflection_history']
)


# -------------------------------------------------------------------------------------
# ReAct Agent Input Prompt Templates for ACE Agent Generator Module Core
# -------------------------------------------------------------------------------------
ace_generator = PromptTemplate(
    template="""
""",
    input_variables=[]
) 


# -------------------------------------------------------------------------------------
# ReAct Agent Input Prompt Templates for ReflACE Agent Actor Module Core
# -------------------------------------------------------------------------------------
reflace_actor = PromptTemplate(
    template="""
""",
    input_variables=[]
)
