from typing import Callable, Optional

import httpx

from tau2.agent.external import (
    ExternalAgent,
    ExternalAgentContext,
    make_external_agent_factory,
)
from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
from tau2.data_model.tasks import EnvAssertion, Task
from tau2.environment.environment import Environment
from tau2.orchestrator.orchestrator import Orchestrator, Role
from tau2.registry import registry
from tau2.runner.build import build_agent
from tau2.user.user_simulator import UserSimulator


class ToolCallingDriver:
    """External driver fixture that calls a tau2 tool over HTTP."""

    def __init__(self) -> None:
        self.context: Optional[ExternalAgentContext] = None
        self.tool_result: Optional[dict] = None
        self.stopped = False

    def start(self, context: ExternalAgentContext) -> None:
        self.context = context

    def respond(self, user_message: str) -> str:
        assert self.context is not None
        headers = {"Authorization": self.context.tool_endpoint.authorization_header}
        unauthorized = httpx.get(f"{self.context.tool_endpoint.url}/tools")
        assert unauthorized.status_code == 401
        tools = httpx.get(f"{self.context.tool_endpoint.url}/tools", headers=headers)
        tools.raise_for_status()
        assert "create_task" in {
            tool["function"]["name"] for tool in tools.json()["tools"]
        }

        response = httpx.post(
            f"{self.context.tool_endpoint.url}/tools/create_task",
            headers=headers,
            json={"user_id": "user_1", "title": "Important Meeting"},
        )
        response.raise_for_status()
        self.tool_result = response.json()
        return "The task was created successfully."

    def stop(self) -> None:
        self.stopped = True


def test_external_agent_records_standard_tool_trajectory(
    domain_name: str,
    get_environment: Callable[[], Environment],
    base_task: Task,
) -> None:
    environment = get_environment()
    driver = ToolCallingDriver()
    agent = ExternalAgent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        task=base_task,
        tool_executor=environment.get_response,
        driver_factory=lambda: driver,
    )
    user = UserSimulator(
        instructions="You are a user simulator.",
        llm="gpt-3.5-turbo",
        llm_args={"temperature": 0.0},
    )
    orchestrator = Orchestrator(
        domain=domain_name,
        user=user,
        agent=agent,
        environment=environment,
        task=base_task,
    )

    try:
        orchestrator.initialize()
        user_message = UserMessage(
            role="user", content="Please create the meeting task."
        )
        orchestrator.trajectory.append(user_message)
        orchestrator.message = user_message
        orchestrator.from_role = Role.USER
        orchestrator.to_role = Role.AGENT

        orchestrator.step()
        trajectory = orchestrator.get_trajectory()

        assert driver.context is not None
        assert driver.context.task_id == base_task.id
        assert driver.context.domain_policy == environment.get_policy()
        assert len(driver.context.message_history) == 1
        assert driver.tool_result is not None
        assert driver.tool_result["error"] is False

        tool_call_message = trajectory[-3]
        tool_result_message = trajectory[-2]
        final_message = trajectory[-1]
        assert isinstance(tool_call_message, AssistantMessage)
        assert tool_call_message.tool_calls[0].name == "create_task"
        assert isinstance(tool_result_message, ToolMessage)
        assert tool_result_message.id == tool_call_message.tool_calls[0].id
        assert isinstance(final_message, AssistantMessage)
        assert final_message.content == "The task was created successfully."
        assert orchestrator.num_errors == 0
        Orchestrator.validate_message_history(trajectory)
        environment.run_env_assertion(
            EnvAssertion(
                env_type="assistant",
                func_name="assert_number_of_tasks",
                arguments={"user_id": "user_1", "expected_number": 2},
            )
        )
    finally:
        agent.stop()

    assert driver.stopped


def test_build_agent_supplies_restricted_tool_executor(
    monkeypatch,
    get_environment: Callable[[], Environment],
    base_task: Task,
) -> None:
    driver = ToolCallingDriver()
    name = "test_external_agent"
    monkeypatch.setitem(
        registry._agent_factories,
        name,
        make_external_agent_factory(lambda: driver),
    )

    agent = build_agent(name, get_environment(), task=base_task)

    assert isinstance(agent, ExternalAgent)
    agent.stop()
