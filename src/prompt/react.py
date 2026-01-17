from pathlib import Path

SYSTEM_PROMPT = Path(__file__).parent.joinpath("templates/react/system.txt").read_text(encoding='utf-8')
INPUT_PROMPT = Path(__file__).parent.joinpath("templates/react/input.txt").read_text(encoding='utf-8')