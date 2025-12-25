import os
from dotenv import load_dotenv
from ..env.appworld_env import AppWorldEnv
from ..agent.react import ReActAgent
from ..agent.reflexion import ReflexionAgent
from ..llm.openai_client import OpenAIClient
from ..llm.tools import TOOLS
from ..prompt.react.system_prompt import react_system_prompt
from ..prompt.reflexion.system_prompt import (
    reflexion_actor_system_prompt,
    reflexion_reflector_system_prompt,
    reflexion_reflector_with_gt_system_prompt
)

# --------------------------------------------------------------------------------------------------------------
# Setup Agent
# --------------------------------------------------------------------------------------------------------------
def setup_agent(
    agent: str = "react",
    model_name: str = None,
    temperature: float = None,
    use_ground_truth: bool = False,
):
    """
    Setup the agent.
    """

    if agent == 'react':
        actor_system_prompt = react_system_prompt.template
        
        actor_client = OpenAIClient(
            model_name=model_name if model_name is not None else actor_system_prompt.model,
            temperature=temperature if temperature is not None else actor_system_prompt.temperature,
            tools=TOOLS,
            system_prompt=actor_system_prompt.template
        )

        agent = ReActAgent(actor_client)

    elif agent == 'reflexion':
        actor_system_prompt = reflexion_actor_system_prompt
        reflector_system_prompt = reflexion_reflector_system_prompt if not use_ground_truth else reflexion_reflector_with_gt_system_prompt
        
        actor_client = OpenAIClient(
            model_name=model_name if model_name is not None else actor_system_prompt.model,
            temperature=temperature if temperature is not None else actor_system_prompt.temperature,
            tools=TOOLS,
            system_prompt=actor_system_prompt.template
        )
        if use_ground_truth:
            reflector_client = OpenAIClient(
                model_name=model_name if model_name is not None else reflector_system_prompt.model,
                temperature=temperature if temperature is not None else reflector_system_prompt.temperature,
                system_prompt=reflector_system_prompt.template
            )
        else:
            reflector_client = OpenAIClient(
                model_name=model_name if model_name is not None else reflector_system_prompt.model,
                temperature=temperature if temperature is not None else reflector_system_prompt.temperature,
                tools=TOOLS,
                system_prompt=reflector_system_prompt.template
            )

        agent = ReflexionAgent(
            actor_client=actor_client,
            reflector_client=reflector_client,
            use_ground_truth=use_ground_truth
        )
    elif agent == 'ace':
        raise NotImplementedError()
    elif agent == 'reflace':
        raise NotImplementedError()
    else:
        raise ValueError(f"Invalid agent type: {agent}")

    return agent


# --------------------------------------------------------------------------------------------------------------
# Setup Benchmark Environment
# --------------------------------------------------------------------------------------------------------------
def setup_env(
    task_type:str,
    task_id:str,
    experiment_name:str,
):
    env_wrapper = AppWorldEnv(task_type=task_type)
    return env_wrapper


# --------------------------------------------------------------------------------------------------------------
# Setup Evaluation Pipeline
# --------------------------------------------------------------------------------------------------------------
def setup_pipeline(
    agent: str = "react",
    model_name: str = "gpt-4o",
    temperature: float = 0.0,
    task_type: str = "dev",
    task_id: int = 0,
    experiment_name: str = "SampleExperiment",
    use_ground_truth: bool = False,
):
    """
    Setup the evaluation pipeline.
    Checks for API keys and environment setup.
    """
    load_dotenv(os.path.join('config', '.env'))
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY environment variable not set.")
    
    agent = setup_agent(
        agent=agent,
        model_name=model_name,
        temperature=temperature,
        use_ground_truth=use_ground_truth
    )

    env_wrapper = setup_env(
        task_type=task_type, 
        task_id=task_id, 
        experiment_name=experiment_name
    )
    
    print("✅ Pipeline setup complete.\n\n")

    return agent, env_wrapper
