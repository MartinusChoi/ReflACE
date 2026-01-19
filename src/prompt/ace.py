from pathlib import Path

GENERATOR_SYSTEM_PROMPT = Path(__file__).parent.joinpath("templates/ace/generator_system.txt").read_text(encoding='utf-8')
GENERATOR_INPUT_PROMPT = Path(__file__).parent.joinpath("templates/ace/generator_input.txt").read_text(encoding='utf-8')
GENERATOR_RESPONSE_MODULE_SYSTEM_PROMPT = Path(__file__).parent.joinpath("templates/ace/generator_response_module_system.txt").read_text(encoding='utf-8')
GENERATOR_RESPONSE_MODULE_INPUT_PROMPT = Path(__file__).parent.joinpath("templates/ace/generator_response_module_input.txt").read_text(encoding='utf-8')

REFLECTOR_SYSTEM_PROMPT = Path(__file__).parent.joinpath("templates/ace/reflector_system.txt").read_text(encoding='utf-8')
REFLECTOR_INPUT_PROMPT = Path(__file__).parent.joinpath("templates/ace/reflector_input.txt").read_text(encoding='utf-8')

REFLECTOR_WITH_GT_SYSTEM_PROMPT = Path(__file__).parent.joinpath("templates/ace/reflector_with_gt_system.txt").read_text(encoding='utf-8')
REFLECTOR_WITH_GT_INPUT_PROMPT = Path(__file__).parent.joinpath("templates/ace/reflector_with_gt_input.txt").read_text(encoding='utf-8')

CURATOR_SYSTEM_PROMPT = Path(__file__).parent.joinpath("templates/ace/curator_system.txt").read_text(encoding='utf-8')
CURATOR_INPUT_PROMPT = Path(__file__).parent.joinpath("templates/ace/curator_input.txt").read_text(encoding='utf-8')