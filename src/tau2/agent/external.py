"""Adapter for evaluating agents that run outside the tau2 process."""

import secrets
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from importlib import import_module
from queue import Queue
from typing import Callable, Optional, Protocol, TypeVar, cast, runtime_checkable

import uvicorn
from fastapi import Body, FastAPI, Header, HTTPException
from loguru import logger
from pydantic import BaseModel

from tau2.agent.base_agent import AgentError, HalfDuplexAgent, ValidAgentInputMessage
from tau2.agent.tool_execution import ToolExecution
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import ExternalAgentConfig
from tau2.data_model.tasks import Task
from tau2.environment.tool import Tool
from tau2.utils.utils import get_now

ToolExecutor = Callable[[ToolCall], ToolMessage]
DriverCallResult = TypeVar("DriverCallResult")


@dataclass(frozen=True)
class ExternalToolEndpoint:
    """Authenticated HTTP endpoint exposing the current tau2 environment tools."""

    url: str
    bearer_token: str

    @property
    def authorization_header(self) -> str:
        """Return the HTTP Authorization header value for this endpoint."""
        return f"Bearer {self.bearer_token}"


@dataclass(frozen=True)
class ExternalAgentContext:
    """Tau2-owned inputs supplied when an external agent session starts."""

    task_id: str
    domain_policy: str
    tool_endpoint: ExternalToolEndpoint
    message_history: tuple[Message, ...]


@runtime_checkable
class ExternalAgentDriver(Protocol):
    """Lifecycle interface implemented by an external agent adapter."""

    def start(self, context: ExternalAgentContext) -> None:
        """Start one external agent session using the supplied tau2 context."""
        ...

    def respond(self, user_message: str) -> str:
        """Run one turn and return after all of that turn's tool calls complete."""
        ...

    def stop(self) -> None:
        """Release resources and unblock any in-progress start or respond call."""
        ...


ExternalAgentDriverFactory = Callable[[], ExternalAgentDriver]


class ExternalAgentState(BaseModel):
    """State tracked by tau2 for an external agent session."""

    turns: int = 0


class ExternalToolGateway:
    """Serve one environment's tools and record completed calls for evaluation."""

    def __init__(
        self,
        tools: list[Tool],
        executor: ToolExecutor,
        host: str = "127.0.0.1",
    ) -> None:
        self._tools = tools
        self._executor = executor
        self._host = host
        self._token = secrets.token_urlsafe(32)
        self._executions: list[ToolExecution] = []
        self._execution_lock = threading.Lock()
        self._turn_active = False
        self._server: Optional[uvicorn.Server] = None
        self._server_thread: Optional[threading.Thread] = None
        self._socket: Optional[socket.socket] = None
        self._endpoint: Optional[ExternalToolEndpoint] = None
        self.app = self._build_app()

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="tau2 external tool gateway")

        @app.get("/health")
        def health(authorization: Optional[str] = Header(default=None)) -> dict:
            self._authorize(authorization)
            return {"status": "ok"}

        @app.get("/tools")
        def list_tools(authorization: Optional[str] = Header(default=None)) -> dict:
            self._authorize(authorization)
            return {"tools": [tool.openai_schema for tool in self._tools]}

        @app.post("/tools/{tool_name}", response_model=ToolMessage)
        def execute_tool(
            tool_name: str,
            arguments: dict = Body(),
            authorization: Optional[str] = Header(default=None),
        ) -> ToolMessage:
            self._authorize(authorization)
            call = ToolCall(
                id=f"external-{uuid.uuid4()}",
                name=tool_name,
                arguments=arguments,
                requestor="assistant",
            )
            requested_at = get_now()
            with self._execution_lock:
                if not self._turn_active:
                    raise HTTPException(
                        status_code=409,
                        detail="Tool calls are only allowed during an active agent turn",
                    )
                result = self._executor(call)
                self._executions.append(
                    ToolExecution(
                        call=call,
                        result=result,
                        requested_at=requested_at,
                    )
                )
            return result

        return app

    def _authorize(self, authorization: Optional[str]) -> None:
        expected = f"Bearer {self._token}"
        if authorization is None or not secrets.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="Invalid bearer token")

    def start(self) -> ExternalToolEndpoint:
        """Start the loopback server and return its authenticated endpoint."""
        if self._endpoint is not None:
            return self._endpoint

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self._host, 0))
        server_socket.listen(128)
        port = server_socket.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(self.app, log_level="warning", lifespan="off")
        )
        thread = threading.Thread(
            target=server.run,
            kwargs={"sockets": [server_socket]},
            name=f"tau2-tool-gateway-{port}",
            daemon=True,
        )
        self._server = server
        self._server_thread = thread
        self._socket = server_socket
        thread.start()

        deadline = time.monotonic() + 5
        while not server.started and thread.is_alive():
            if time.monotonic() >= deadline:
                self.stop()
                raise RuntimeError("Timed out starting external tool gateway")
            time.sleep(0.01)
        if not server.started:
            self.stop()
            raise RuntimeError("External tool gateway failed to start")

        self._endpoint = ExternalToolEndpoint(
            url=f"http://{self._host}:{port}",
            bearer_token=self._token,
        )
        return self._endpoint

    def begin_turn(self) -> None:
        """Open one agent turn for tool execution."""
        with self._execution_lock:
            if self._turn_active:
                raise RuntimeError("External tool turn is already active")
            if self._executions:
                raise RuntimeError("External tool executions were not consumed")
            self._turn_active = True

    def finish_turn(self) -> list[ToolExecution]:
        """Close the active turn and return its calls in execution order."""
        with self._execution_lock:
            if not self._turn_active:
                raise RuntimeError("External tool turn is not active")
            self._turn_active = False
            executions = self._executions
            self._executions = []
        return executions

    def abort_turn(self) -> None:
        """Close the active turn and discard its calls."""
        with self._execution_lock:
            self._turn_active = False
            self._executions = []

    def stop(self) -> None:
        """Stop the loopback server. Repeated calls are safe."""
        self.abort_turn()
        if self._server is not None:
            self._server.should_exit = True
        if self._server_thread is not None:
            self._server_thread.join(timeout=5)
        if self._socket is not None:
            self._socket.close()
        self._server = None
        self._server_thread = None
        self._socket = None
        self._endpoint = None


class ExternalAgent(HalfDuplexAgent[ExternalAgentState]):
    """Half-duplex tau2 agent backed by an opaque external agent runtime."""

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        task: Task,
        tool_executor: ToolExecutor,
        driver_factory: ExternalAgentDriverFactory,
        startup_timeout: float = 30.0,
        turn_timeout: float = 900.0,
    ) -> None:
        super().__init__(tools=tools, domain_policy=domain_policy)
        self._task = task
        self._driver = driver_factory()
        self._gateway = ExternalToolGateway(tools, tool_executor)
        self._startup_timeout = startup_timeout
        self._turn_timeout = turn_timeout
        self._pending_executions: list[ToolExecution] = []
        self._started = False
        self._closed = False
        self._driver_stopped = False

    def _call_driver(
        self,
        operation: Callable[[], DriverCallResult],
        timeout: float,
        phase: str,
    ) -> DriverCallResult:
        outcomes: Queue[tuple[bool, object]] = Queue(maxsize=1)

        def invoke() -> None:
            try:
                outcomes.put((True, operation()))
            except BaseException as error:
                outcomes.put((False, error))

        thread = threading.Thread(
            target=invoke,
            name=f"tau2-external-agent-{phase}",
            daemon=True,
        )
        thread.start()
        thread.join(timeout=timeout)
        if thread.is_alive():
            raise AgentError(f"External agent {phase} timed out after {timeout}s")
        succeeded, outcome = outcomes.get_nowait()
        if not succeeded:
            if not isinstance(outcome, BaseException):
                raise RuntimeError("External driver returned an invalid error")
            raise outcome
        return cast(DriverCallResult, outcome)

    def _shutdown(self) -> None:
        try:
            if not self._driver_stopped:
                self._driver_stopped = True
                try:
                    self._driver.stop()
                except Exception as error:
                    logger.warning(f"Error stopping external agent driver: {error}")
        finally:
            try:
                self._gateway.stop()
            finally:
                self._started = False
                self._closed = True

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> ExternalAgentState:
        """Start the tool gateway and external agent session."""
        if self._closed:
            raise AgentError("External agent session is closed")
        if self._started:
            raise AgentError("External agent session is already started")
        endpoint = self._gateway.start()
        try:
            context = ExternalAgentContext(
                task_id=self._task.id,
                domain_policy=self.domain_policy,
                tool_endpoint=endpoint,
                message_history=tuple(message_history or []),
            )
            self._call_driver(
                lambda: self._driver.start(context),
                timeout=self._startup_timeout,
                phase="startup",
            )
        except Exception:
            self._shutdown()
            raise
        self._started = True
        return ExternalAgentState()

    def generate_next_message(
        self,
        message: ValidAgentInputMessage,
        state: ExternalAgentState,
    ) -> tuple[AssistantMessage, ExternalAgentState]:
        """Run an external turn while tau2 serves and records its tool calls."""
        if not self._started:
            raise AgentError("External agent session is not started")
        if not isinstance(message, UserMessage) or message.is_tool_call():
            raise AgentError("External agents accept user text messages only")
        if self._pending_executions:
            raise AgentError("Previous external tool executions were not consumed")

        self._gateway.begin_turn()
        try:
            response = self._call_driver(
                lambda: self._driver.respond(message.content or ""),
                timeout=self._turn_timeout,
                phase="turn",
            )
        except Exception:
            self._gateway.abort_turn()
            self._shutdown()
            raise
        if not isinstance(response, str) or not response.strip():
            self._gateway.abort_turn()
            self._shutdown()
            raise AgentError("External agent must return non-empty text")
        self._pending_executions = self._gateway.finish_turn()
        state.turns += 1
        return AssistantMessage(role="assistant", content=response), state

    def drain_tool_executions(self) -> list[ToolExecution]:
        """Return and clear tools completed during the latest external turn."""
        executions = self._pending_executions
        self._pending_executions = []
        return executions

    def stop(
        self,
        message: Optional[ValidAgentInputMessage] = None,
        state: Optional[ExternalAgentState] = None,
    ) -> None:
        """Stop the external agent and its tool gateway."""
        self._shutdown()


def make_external_agent_factory(
    driver_factory: ExternalAgentDriverFactory,
) -> Callable[..., ExternalAgent]:
    """Build a registry-compatible factory for one external agent adapter."""

    def factory(
        *,
        tools: list[Tool],
        domain_policy: str,
        task: Optional[Task] = None,
        tool_executor: Optional[ToolExecutor] = None,
        **_: object,
    ) -> ExternalAgent:
        if task is None:
            raise ValueError("External agents require a task")
        if tool_executor is None:
            raise ValueError("External agents require a tau2 tool executor")
        return ExternalAgent(
            tools=tools,
            domain_policy=domain_policy,
            task=task,
            tool_executor=tool_executor,
            driver_factory=driver_factory,
        )

    return factory


def _load_driver_factory(
    config: ExternalAgentConfig,
) -> ExternalAgentDriverFactory:
    module_name, attribute_name = config.driver.split(":", maxsplit=1)
    try:
        module = import_module(module_name)
    except ImportError as error:
        raise ValueError(
            f"Cannot import external agent driver module '{module_name}'"
        ) from error
    try:
        target = getattr(module, attribute_name)
    except AttributeError as error:
        raise ValueError(
            f"External agent driver '{config.driver}' does not exist"
        ) from error
    if not callable(target):
        raise ValueError(f"External agent driver '{config.driver}' is not callable")

    def factory() -> ExternalAgentDriver:
        driver = target(**config.driver_args)
        if not isinstance(driver, ExternalAgentDriver):
            raise TypeError(
                f"External agent driver '{config.driver}' must implement "
                "start(), respond(), and stop()"
            )
        return driver

    return factory


def create_external_agent(
    tools: list[Tool],
    domain_policy: str,
    **kwargs: object,
) -> ExternalAgent:
    """Create an external agent from serializable runner configuration."""
    raw_config = kwargs.get("external_agent")
    if raw_config is None:
        raise ValueError("external_agent configuration is required")
    config = ExternalAgentConfig.model_validate(raw_config)
    task = kwargs.get("task")
    tool_executor = kwargs.get("tool_executor")
    if not isinstance(task, Task):
        raise ValueError("External agents require a task")
    if not callable(tool_executor):
        raise ValueError("External agents require a tau2 tool executor")
    return ExternalAgent(
        tools=tools,
        domain_policy=domain_policy,
        task=task,
        tool_executor=tool_executor,
        driver_factory=_load_driver_factory(config),
        startup_timeout=config.startup_timeout,
        turn_timeout=config.turn_timeout,
    )
