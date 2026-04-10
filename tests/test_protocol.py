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

    async def update_status(self, state, message):
        self.statuses.append((state, message))

    async def add_artifact(self, *, parts, name):
        self.artifacts.append((name, parts))


def make_message(text: str, *, context_id: str = "ctx-1") -> Message:
    return Message(
        kind="message",
        role=Role.user,
        parts=[Part(root=TextPart(text=text))],
        message_id="msg-1",
        context_id=context_id,
    )


@pytest.mark.asyncio
async def test_task_message_returns_final():
    agent = Agent()
    updater = FakeUpdater()

    await agent.run(
        make_message(json.dumps({"kind": "task", "instruction": "solve"})),
        updater,
    )

    assert len(updater.artifacts) == 1
    result = json.loads(updater.artifacts[0][1][0].root.text)
    assert result["kind"] == "final"


@pytest.mark.asyncio
async def test_exec_result_returns_final():
    agent = Agent()
    updater = FakeUpdater()

    await agent.run(
        make_message(json.dumps({
            "kind": "exec_result",
            "exit_code": 0,
            "stdout": "/workspace\n",
            "stderr": "",
        })),
        updater,
    )

    assert len(updater.artifacts) == 1
    result = json.loads(updater.artifacts[0][1][0].root.text)
    assert result["kind"] == "final"


@pytest.mark.asyncio
async def test_plain_text_returns_final():
    agent = Agent()
    updater = FakeUpdater()

    await agent.run(
        make_message("hello world"),
        updater,
    )

    assert len(updater.artifacts) == 1
    result = json.loads(updater.artifacts[0][1][0].root.text)
    assert result["kind"] == "final"
    assert result["output"] == "hello world"
