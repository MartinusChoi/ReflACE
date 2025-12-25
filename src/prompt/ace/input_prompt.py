from ..base import PromptTemplate

ace_reflector_input_prompt = PromptTemplate(
    model="gpt-4o",
    temperature=0,
    input_variables=["instruction", "trajectory", "playbook"],
    template="""
**Task:**
{instruction}

**Model Action Trajectory:**
{trajectory}

**Part of Playbook that's used by the generator to answer the question:**
{playbook}

**Answer in this exact JSON format:**
{{
  "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations]",
  "error_identification": "[What specifically went wrong in the reasoning?]",
  "root_cause_analysis": "[Why did this error occur? What concept was misunderstood?]",
  "correct_approach": "[What should the model have done instead?]",
  "key_insight": "[What strategy, formula, or principle should be remembered to avoid this error?]",
  "bullet_tags": [
    {{"id": "calc-00001", "tag": "helpful"}},
    {{"id": "fin-00002", "tag": "harmful"}}
  ]
}}
"""
)



# -------------------------------------------------------------------------------------
# ReAct Agent Input Prompt Templates for ACE Agent Generator Module Core
# -------------------------------------------------------------------------------------
react_with_playbook_input_prompt = PromptTemplate(
    model="gpt-4o",
    temperature=0,
    input_variables=['first_name', 'last_name', 'email', 'phone_number', 'instruction', 'playbook'],
    template="""
Using these 'APIs' and 'playbook', now generate code to solve the actual task:

My name is {first_name} {last_name}. 
My personal email is {email} and phone number is {phone_number}.

**Task**:
{instruction}

**ACE Playbook**: - Read the Playbook first, then execute the task by explicitly leveraging each relevant section:

<PLAYBOOK>

{playbook}

</PLAYBOOK>
"""
)
