from pathlib import Path

ACTOR_INPUT_PROMPT = Path(__file__).parent.joinpath("templates/reflexion/actor_input.txt").read_text(encoding="utf-8")
ACTOR_SYSTEM_PROMPT = Path(__file__).parent.joinpath("templates/reflexion/actor_system.txt").read_text(encoding="utf-8")

REFLECTOR_SYSTEM_PROMPT = Path(__file__).parent.joinpath("templates/reflexion/reflector_system.txt").read_text(encoding="utf-8")
REFLECTOR_INPUT_PROMPT = Path(__file__).parent.joinpath("templates/reflexion/reflector_input.txt").read_text(encoding="utf-8")

REFLECTOR_WITH_GT_SYSTEM_PROMPT = Path(__file__).parent.joinpath("templates/reflexion/reflector_with_gt_system.txt").read_text(encoding="utf-8")
REFLECTOR_WITH_GT_INPUT_PROMPT = Path(__file__).parent.joinpath("templates/reflexion/reflector_with_gt_input.txt").read_text(encoding="utf-8")