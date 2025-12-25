from ..base import PromptTemplate

# -------------------------------------------------------------------------------------
# ReAct Agent Input Prompt Templates
# -------------------------------------------------------------------------------------
react_only_input_prompt = PromptTemplate(
    template="""
Using these APIs, now generate code to solve the actual task:

My name is {first_name} {last_name}. 
My personal email is {email} and phone number is {phone_number}.

**Task**:
{instruction}
""",
    input_variables=['first_name', 'last_name', 'email', 'phone_number', 'instruction']
)