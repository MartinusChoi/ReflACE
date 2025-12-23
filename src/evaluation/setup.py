import os
from dotenv import load_dotenv
from ..env.appworld_env import AppWorldEnv
from ..agent.react import ReActAgent
from ..agent.reflexion import ReflexionAgent
from ..llm.openai_client import OpenAIClient
from ..llm.tools import TOOLS

# --------------------------------------------------------------------------------------------------------------
# Setup Agent
# --------------------------------------------------------------------------------------------------------------
def setup_agent(
    agent: str = "react",
    model_name: str = "gpt-4o",
    temperature: float = 0.0,
):
    """
    Setup the agent.
    """

    if agent == 'react':
        actor_system_prompt_path = os.path.join('src', 'prompt', 'react', 'system_prompt_oneshot.txt')
        if os.path.exists(actor_system_prompt_path):
            with open(actor_system_prompt_path, 'r', encoding='utf-8') as f:
                actor_system_prompt = f.read()
        else:
            raise FileNotFoundError(f"Prompt file not found at {actor_system_prompt_path}")
        
        actor_client = OpenAIClient(
            model_name=model_name,
            temperature=temperature,
            tools=TOOLS,
            system_prompt=actor_system_prompt
        )

        agent = ReActAgent(actor_client)

    elif agent == 'reflexion':
        actor_system_prompt_path = os.path.join('src', 'prompt', 'react', 'system_prompt_oneshot.txt')
        reflector_system_prompt_path = os.path.join('src', 'prompt', 'reflexion', 'reflector_system_prompt.txt')

        if os.path.exists(actor_system_prompt_path) and os.path.exists(reflector_system_prompt_path):
            with open(actor_system_prompt_path, 'r', encoding='utf-8') as f:
                actor_system_prompt = f.read()
            with open(reflector_system_prompt_path, 'r', encoding='utf-8') as f:
                reflector_system_prompt = f.read()
        else:
            raise FileNotFoundError(f"Prompt files not found at {actor_system_prompt_path} or {reflector_system_prompt_path}")
        
        actor_client = OpenAIClient(
            model_name=model_name,
            temperature=temperature,
            tools=TOOLS,
            system_prompt=actor_system_prompt
        )
        reflector_client = OpenAIClient(
            model_name=model_name,
            temperature=temperature,
            system_prompt=reflector_system_prompt
        )

        agent = ReflexionAgent(
            actor_client=actor_client,
            reflector_client=reflector_client,
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
        temperature=temperature
    )

    env_wrapper = setup_env(
        task_type=task_type, 
        task_id=task_id, 
        experiment_name=experiment_name
    )
    
    print("✅ Pipeline setup complete.")

    return agent, env_wrapper
