import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import httpx
import pytest
from pydantic import BaseModel

from tau2.agent.base_agent import AgentError
from tau2.agent.external import (
    ExternalAgent,
    ExternalAgentContext,
    make_external_agent_factory,
)
from tau2.cli import add_run_args
from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
from tau2.data_model.simulation import TerminationReason, TextRunConfig
from tau2.data_model.tasks import EnvAssertion, Task
from tau2.environment.environment import Environment
from tau2.evaluator.evaluator import EvaluationType
from tau2.orchestrator.orchestrator import Orchestrator, Role
from tau2.registry import registry
from tau2.runner.build import build_agent, build_text_orchestrator
from tau2.runner.simulation import run_simulation
from tau2.user.user_simulator import UserSimulator
from tau2.user.user_simulator_base import (
    STOP,
    HalfDuplexUser,
    ValidUserInputMessage,
)


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


class ConfiguredDriver(ToolCallingDriver):
    """Driver fixture loaded from a declarative import path."""

    instances: list["ConfiguredDriver"] = []

    def __init__(self, marker: str) -> None:
        super().__init__()
        self.marker = marker
        self.__class__.instances.append(self)


def test_cli_accepts_external_agent_configuration() -> None:
    parser = argparse.ArgumentParser()
    add_run_args(parser)

    args = parser.parse_args(
        [
            "--agent",
            "external_agent",
            "--external-agent-config",
            '{"driver":"package.module:create_driver"}',
        ]
    )

    assert args.external_agent_config == {"driver": "package.module:create_driver"}


class OutOfTurnDriver:
    """Driver fixture that attempts tool calls outside respond()."""

    def __init__(self) -> None:
        self.context: Optional[ExternalAgentContext] = None
        self.start_status: Optional[int] = None
        self.late_status: Optional[int] = None
        self.release_late_call = threading.Event()
        self.late_call_finished = threading.Event()

    def _call_tool(self) -> int:
        assert self.context is not None
        response = httpx.post(
            f"{self.context.tool_endpoint.url}/tools/create_task",
            headers={"Authorization": self.context.tool_endpoint.authorization_header},
            json={"user_id": "user_1", "title": "Out of turn"},
        )
        return response.status_code

    def start(self, context: ExternalAgentContext) -> None:
        self.context = context
        self.start_status = self._call_tool()

    def respond(self, user_message: str) -> str:
        def call_late() -> None:
            self.release_late_call.wait()
            self.late_status = self._call_tool()
            self.late_call_finished.set()

        threading.Thread(target=call_late, daemon=True).start()
        return "No tool was needed."

    def stop(self) -> None:
        self.release_late_call.set()


class BlockingDriver:
    """Driver fixture whose selected lifecycle method blocks until stopped."""

    def __init__(self, phase: str) -> None:
        self.phase = phase
        self.release = threading.Event()
        self.stop_calls = 0

    def start(self, context: ExternalAgentContext) -> None:
        if self.phase == "start":
            self.release.wait(timeout=0.2)

    def respond(self, user_message: str) -> str:
        if self.phase == "respond":
            self.release.wait(timeout=0.2)
        return "Finished."

    def stop(self) -> None:
        self.stop_calls += 1
        self.release.set()


class FailingStartDriver(BlockingDriver):
    """Driver fixture that allocates state and then fails to start."""

    def __init__(self) -> None:
        super().__init__(phase="none")

    def start(self, context: ExternalAgentContext) -> None:
        raise RuntimeError("startup failed")


class InvalidResponseDriver(BlockingDriver):
    """Driver fixture that violates the response type requirement."""

    def __init__(self) -> None:
        super().__init__(phase="none")

    def respond(self, user_message: str) -> str:
        return None


class ScriptedUserState(BaseModel):
    """State for the keyless user used by end-to-end external agent tests."""

    turns: int = 0


class ScriptedUser(HalfDuplexUser[ScriptedUserState]):
    """Ask for one task, then stop after the external agent responds."""

    def get_init_state(self, message_history=None) -> ScriptedUserState:
        return ScriptedUserState()

    def generate_next_message(
        self,
        message: ValidUserInputMessage,
        state: ScriptedUserState,
    ) -> tuple[UserMessage, ScriptedUserState]:
        state.turns += 1
        if state.turns == 1:
            return (
                UserMessage(role="user", content="Create the Important Meeting task."),
                state,
            )
        return UserMessage(role="user", content=STOP), state


class MultipleToolDriver(ToolCallingDriver):
    """Driver fixture that executes a read followed by a write."""

    def respond(self, user_message: str) -> str:
        assert self.context is not None
        headers = {"Authorization": self.context.tool_endpoint.authorization_header}
        read_response = httpx.post(
            f"{self.context.tool_endpoint.url}/tools/get_users",
            headers=headers,
            json={},
        )
        read_response.raise_for_status()
        create_response = httpx.post(
            f"{self.context.tool_endpoint.url}/tools/create_task",
            headers=headers,
            json={"user_id": "user_1", "title": "Important Meeting"},
        )
        create_response.raise_for_status()
        return "The task was created successfully."


class ErrorToolDriver(ToolCallingDriver):
    """Driver fixture that invokes an unavailable environment tool."""

    def respond(self, user_message: str) -> str:
        assert self.context is not None
        response = httpx.post(
            f"{self.context.tool_endpoint.url}/tools/unavailable_tool",
            headers={"Authorization": self.context.tool_endpoint.authorization_header},
            json={},
        )
        response.raise_for_status()
        assert response.json()["error"] is True
        return "I could not complete that request."


class ConcurrentDriver(ToolCallingDriver):
    """Driver fixture that synchronizes calls across two isolated agents."""

    def __init__(self, barrier: threading.Barrier, title: str) -> None:
        super().__init__()
        self.barrier = barrier
        self.title = title

    def respond(self, user_message: str) -> str:
        assert self.context is not None
        self.barrier.wait(timeout=1)
        response = httpx.post(
            f"{self.context.tool_endpoint.url}/tools/create_task",
            headers={"Authorization": self.context.tool_endpoint.authorization_header},
            json={"user_id": "user_1", "title": self.title},
        )
        response.raise_for_status()
        return f"Created {self.title}."


def make_test_external_agent(
    environment: Environment,
    task: Task,
    driver,
    *,
    startup_timeout: float = 1,
    turn_timeout: float = 1,
) -> ExternalAgent:
    """Build an ExternalAgent around a test driver."""
    return ExternalAgent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        task=task,
        tool_executor=environment.get_response,
        driver_factory=lambda: driver,
        startup_timeout=startup_timeout,
        turn_timeout=turn_timeout,
    )


def make_test_orchestrator(
    environment: Environment,
    task: Task,
    driver,
    *,
    max_errors: int = 10,
) -> Orchestrator:
    """Build a keyless orchestrator around one external test adapter."""
    return Orchestrator(
        domain="mock",
        agent=make_test_external_agent(environment, task, driver),
        user=ScriptedUser(),
        environment=environment,
        task=task,
        max_errors=max_errors,
    )


def test_external_agent_full_run_replays_multiple_tools_in_evaluator(
    get_environment: Callable[[], Environment],
    base_task: Task,
) -> None:
    driver = MultipleToolDriver()
    simulation = run_simulation(
        make_test_orchestrator(get_environment(), base_task, driver),
        evaluation_type=EvaluationType.ENV,
    )

    tool_call_messages = [
        message
        for message in simulation.messages
        if isinstance(message, AssistantMessage) and message.is_tool_call()
    ]
    tool_results = [
        message for message in simulation.messages if isinstance(message, ToolMessage)
    ]
    assert [message.tool_calls[0].name for message in tool_call_messages] == [
        "get_users",
        "create_task",
    ]
    assert [result.id for result in tool_results] == [
        message.tool_calls[0].id for message in tool_call_messages
    ]
    assert simulation.termination_reason == TerminationReason.USER_STOP
    assert simulation.reward_info is not None
    assert simulation.reward_info.reward == 1.0
    assert driver.stopped


def test_external_tool_errors_count_toward_orchestrator_limit(
    get_environment: Callable[[], Environment],
    base_task: Task,
) -> None:
    driver = ErrorToolDriver()
    simulation = run_simulation(
        make_test_orchestrator(get_environment(), base_task, driver, max_errors=1),
        evaluation_type=EvaluationType.ENV,
    )

    tool_results = [
        message for message in simulation.messages if isinstance(message, ToolMessage)
    ]
    assert len(tool_results) == 1
    assert tool_results[0].error is True
    assert simulation.termination_reason == TerminationReason.TOO_MANY_ERRORS
    assert simulation.reward_info is not None
    assert simulation.reward_info.reward == 0.0
    assert driver.stopped


def test_concurrent_external_agents_use_isolated_gateways(
    get_environment: Callable[[], Environment],
    base_task: Task,
) -> None:
    barrier = threading.Barrier(2)
    environments = [get_environment(), get_environment()]
    drivers = [
        ConcurrentDriver(barrier, "First"),
        ConcurrentDriver(barrier, "Second"),
    ]

    def run_turn(index: int):
        agent = make_test_external_agent(environments[index], base_task, drivers[index])
        try:
            state = agent.get_init_state()
            agent.generate_next_message(
                UserMessage(role="user", content="Create it"), state
            )
            return agent.drain_tool_executions()
        finally:
            agent.stop()

    with ThreadPoolExecutor(max_workers=2) as executor:
        executions = list(executor.map(run_turn, range(2)))

    assert drivers[0].context is not None
    assert drivers[1].context is not None
    assert drivers[0].context.tool_endpoint.url != drivers[1].context.tool_endpoint.url
    assert [records[0].call.arguments["title"] for records in executions] == [
        "First",
        "Second",
    ]
    for environment in environments:
        environment.run_env_assertion(
            EnvAssertion(
                env_type="assistant",
                func_name="assert_number_of_tasks",
                arguments={"user_id": "user_1", "expected_number": 2},
            )
        )


def test_external_agent_stops_driver_when_startup_times_out(
    get_environment: Callable[[], Environment],
    base_task: Task,
) -> None:
    driver = BlockingDriver(phase="start")
    agent = make_test_external_agent(
        get_environment(), base_task, driver, startup_timeout=0.02
    )

    with pytest.raises(AgentError, match="startup timed out"):
        agent.get_init_state()

    assert driver.stop_calls == 1
    agent.stop()
    assert driver.stop_calls == 1


def test_external_agent_stops_driver_when_turn_times_out(
    get_environment: Callable[[], Environment],
    base_task: Task,
) -> None:
    driver = BlockingDriver(phase="respond")
    agent = make_test_external_agent(
        get_environment(), base_task, driver, turn_timeout=0.02
    )
    state = agent.get_init_state()

    with pytest.raises(AgentError, match="turn timed out"):
        agent.generate_next_message(UserMessage(role="user", content="Hello"), state)

    assert driver.stop_calls == 1
    agent.stop()
    assert driver.stop_calls == 1


def test_external_agent_cleans_up_driver_after_start_failure(
    get_environment: Callable[[], Environment],
    base_task: Task,
) -> None:
    driver = FailingStartDriver()
    agent = make_test_external_agent(get_environment(), base_task, driver)

    with pytest.raises(RuntimeError, match="startup failed"):
        agent.get_init_state()

    assert driver.stop_calls == 1


def test_external_agent_cleans_up_after_invalid_response(
    get_environment: Callable[[], Environment],
    base_task: Task,
) -> None:
    driver = InvalidResponseDriver()
    agent = make_test_external_agent(get_environment(), base_task, driver)
    state = agent.get_init_state()

    with pytest.raises(AgentError, match="non-empty text"):
        agent.generate_next_message(UserMessage(role="user", content="Hello"), state)

    assert driver.stop_calls == 1


def test_external_agent_rejects_tool_calls_outside_active_turn(
    get_environment: Callable[[], Environment],
    base_task: Task,
) -> None:
    environment = get_environment()
    driver = OutOfTurnDriver()
    agent = ExternalAgent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        task=base_task,
        tool_executor=environment.get_response,
        driver_factory=lambda: driver,
    )

    try:
        state = agent.get_init_state()
        assert driver.start_status == 409

        _, state = agent.generate_next_message(
            UserMessage(role="user", content="Hello"), state
        )
        assert agent.drain_tool_executions() == []

        driver.release_late_call.set()
        assert driver.late_call_finished.wait(timeout=1)
        assert driver.late_status == 409
        assert agent.drain_tool_executions() == []
        environment.run_env_assertion(
            EnvAssertion(
                env_type="assistant",
                func_name="assert_number_of_tasks",
                arguments={"user_id": "user_1", "expected_number": 1},
            )
        )
    finally:
        agent.stop()


def test_external_agent_builds_from_serializable_run_config(
    base_task: Task,
) -> None:
    ConfiguredDriver.instances.clear()
    config = TextRunConfig(
        domain="mock",
        agent="external_agent",
        external_agent={
            "driver": "test_external_agent:ConfiguredDriver",
            "driver_args": {"marker": "loaded"},
            "startup_timeout": 1,
            "turn_timeout": 1,
        },
    )

    restored = TextRunConfig.model_validate(config.model_dump(mode="json"))
    orchestrator = build_text_orchestrator(restored, base_task)

    assert isinstance(orchestrator.agent, ExternalAgent)
    assert len(ConfiguredDriver.instances) == 1
    assert ConfiguredDriver.instances[0].marker == "loaded"
    orchestrator.agent.stop()


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
