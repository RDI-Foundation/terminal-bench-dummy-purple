import asyncio
import json
import sys
from pathlib import Path

import pytest
from a2a.types import Message, Part, Role, TextPart


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent import Agent  # noqa: E402


class FakeUpdater:
    def __init__(self):
        self.statuses = []
        self.artifacts = []
        self.requests = []
        self.request_event = asyncio.Event()

    async def update_status(self, state, message):
        self.statuses.append((state, message))

    async def add_artifact(self, *, parts, name):
        self.artifacts.append((name, parts))

    async def requires_input(self, message):
        self.requests.append(message)
        self.request_event.set()

    def new_agent_message(self, *, parts):
        return Message(
            kind="message",
            role=Role.agent,
            parts=parts,
            message_id="agent-msg",
        )


def make_message(text: str, *, context_id: str) -> Message:
    return Message(
        kind="message",
        role=Role.user,
        parts=[Part(root=TextPart(text=text))],
        message_id="msg-1",
        context_id=context_id,
    )


@pytest.mark.asyncio
async def test_dummy_purple_requests_callback_and_finishes():
    agent = Agent()
    updater = FakeUpdater()

    run_task = asyncio.create_task(
        agent.run(
            make_message(
                json.dumps({"kind": "task", "instruction": "solve"}),
                context_id="ctx-1",
            ),
            updater,
        )
    )

    await asyncio.wait_for(updater.request_event.wait(), timeout=1)
    assert json.loads(updater.requests[0].parts[0].root.text) == {
        "kind": "exec_request",
        "command": "pwd",
        "timeout": 5,
    }

    await agent.run(
        make_message(
            json.dumps(
                {
                    "kind": "exec_result",
                    "exit_code": 0,
                    "stdout": "/workspace\n",
                    "stderr": "",
                }
            ),
            context_id="ctx-1",
        ),
        FakeUpdater(),
    )
    await run_task

    result = json.loads(updater.artifacts[-1][1][0].root.text)
    assert result["kind"] == "final"
    assert "exit_code=0" in result["output"]
