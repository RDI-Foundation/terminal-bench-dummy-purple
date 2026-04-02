import json

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger


class Agent:
    def __init__(self):
        self.messenger = Messenger()

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)
        response = _build_response(input_text)

        await updater.update_status(
            TaskState.working, new_agent_text_message("Processing...")
        )
        await updater.add_artifact(
            parts=[Part(root=TextPart(text=response))],
            name="Result",
        )


def _build_response(input_text: str) -> str:
    """Handle the terminal-bench-shell-v1 protocol.

    Responds with a final message immediately — the dummy agent does not
    execute any commands.
    """
    try:
        payload = json.loads(input_text)
    except json.JSONDecodeError:
        return json.dumps({"kind": "final", "output": input_text})

    if not isinstance(payload, dict):
        return json.dumps({"kind": "final", "output": input_text})

    match payload.get("kind"):
        case "task" | "exec_result":
            return json.dumps({
                "kind": "final",
                "output": "Dummy purple does not execute commands.",
            })
        case _:
            return json.dumps({"kind": "final", "output": input_text})
