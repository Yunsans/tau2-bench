"""Trajectory records for tools executed inside an agent turn."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from tau2.data_model.message import AssistantMessage, Message, ToolCall, ToolMessage


@dataclass(frozen=True)
class ToolExecution:
    """One tool call that has already been executed by the environment.

    Attributes:
        call: The tool call sent by the agent.
        result: The environment response produced for ``call``.
        requested_at: Timestamp captured immediately before execution.
    """

    call: ToolCall
    result: ToolMessage
    requested_at: str

    def to_messages(self) -> tuple[AssistantMessage, ToolMessage]:
        """Convert this execution to the standard tau2 trajectory messages."""
        return (
            AssistantMessage(
                role="assistant",
                tool_calls=[self.call],
                timestamp=self.requested_at,
            ),
            self.result,
        )


@runtime_checkable
class ToolExecutionSource(Protocol):
    """Agent extension for reporting tools completed during its latest turn."""

    def drain_tool_executions(self) -> list[ToolExecution]:
        """Return and clear executions completed during the latest agent turn."""
        ...


def materialize_tool_executions(executions: list[ToolExecution]) -> list[Message]:
    """Convert completed executions to the standard interleaved trajectory."""
    messages: list[Message] = []
    for execution in executions:
        messages.extend(execution.to_messages())
    return messages
