# External Whole-Machine Agent Integration Guide

English | [中文](external-agent-integration.zh.md)

Use this guide when the evaluated agent runs outside the tau2 Python process, owns its model loop and tool system, or is available only as a binary, local application, container, or service. The agent does not need to expose its source code. It does need a controllable session interface and a way to invoke the task-scoped tools supplied by tau2.

The built-in `external_agent` is a text, half-duplex integration. Tau2 continues to own the user simulator, domain state, tool execution, standard trajectory, retries, checkpointing, and evaluation.

## Integration model

```text
tau2 run
  -> ExternalAgent
     -> your Python driver
        -> opaque agent process, application, container, or service
           -> the agent's own model loop and tool registry
        -> tau2 task-scoped HTTP tool gateway
     -> standard tau2 trajectory and evaluator
```

One driver instance is created for each simulation. Tau2 calls `start(context)` once, calls `respond(user_message)` for each user turn, and calls `stop()` during normal completion, failure, or timeout.

## What tau2 requires

The only required tau2-side artifact is an importable Python factory. The factory must accept JSON-serializable keyword arguments and return an object with these methods:

```python
class ExternalAgentDriver:
    def start(self, context: ExternalAgentContext) -> None: ...
    def respond(self, user_message: str) -> str: ...
    def stop(self) -> None: ...
```

The driver may translate this lifecycle into any interface the evaluated agent already supports. Source access is unnecessary if the agent can be launched or reached through a stable CLI, SDK, HTTP API, JSON-RPC API, MCP server, ACP server, or another documented protocol.

If the agent has no programmable session interface and no way to install or call dynamic tools, an adapter cannot make it evaluable without adding one of those capabilities around the agent.

## Recommended adapter package

Keep agent-specific code outside tau2 so the benchmark remains reusable. A small package can use this layout:

```text
my-agent-tau2/
├── pyproject.toml
├── README.md
├── external-agent.json
├── src/
│   └── my_agent_tau2/
│       ├── __init__.py
│       ├── driver.py
│       └── tau2_tools.py
└── tests/
    └── test_driver.py
```

The filenames are recommendations, not protocol requirements. `driver.py` and its importable factory are the required parts. `tau2_tools.py` is useful when the agent can register ordinary callable tools. `external-agent.json` keeps run configuration reviewable.

## File 1: `pyproject.toml`

This package installs the driver into the same Python environment as tau2. The example uses `httpx` to call the task-scoped tool gateway and pytest for adapter tests.

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-agent-tau2"
version = "0.1.0"
requires-python = ">=3.12,<3.14"
dependencies = ["httpx>=0.24"]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.hatch.build.targets.wheel]
packages = ["src/my_agent_tau2"]
```

Do not add model API keys or tau2 gateway tokens to this file.

## File 2: `src/my_agent_tau2/__init__.py`

Export the factory named by the run configuration:

```python
from my_agent_tau2.driver import create_driver

__all__ = ["create_driver"]
```

## File 3: `src/my_agent_tau2/tau2_tools.py`

This complete client lists the current task's tools as OpenAI-compatible function schemas and invokes a tool by its tau2 name. An agent-specific bridge can register each schema in its own tool system and route calls to `call()`.

```python
from __future__ import annotations

from typing import Any

import httpx


class Tau2Tools:
    def __init__(
        self,
        *,
        url: str,
        authorization_header: str,
        timeout_seconds: float = 120,
    ) -> None:
        self._client = httpx.Client(
            base_url=url,
            headers={"Authorization": authorization_header},
            timeout=timeout_seconds,
        )

    def health(self) -> None:
        response = self._client.get("/health")
        response.raise_for_status()

    def schemas(self) -> list[dict[str, Any]]:
        response = self._client.get("/tools")
        response.raise_for_status()
        body = response.json()
        return body["tools"]

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        response = self._client.post(f"/tools/{name}", json=arguments)
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        self._client.close()
```

`GET /health` and `GET /tools` are allowed during startup. `POST /tools/{name}` is accepted only while tau2 is inside the corresponding `respond()` call. The POST response is a serialized tau2 `ToolMessage`; a tool-level failure can therefore be a successful HTTP response whose JSON body reports the tool error.

If the agent natively accepts MCP, OpenAPI, plugins, or another tool protocol, keep the same tau2 endpoint in the Python driver and add a local protocol adapter. DeepSeek Harness, for example, uses an authenticated HTTP-to-MCP proxy. The conversion layer is agent-specific; tau2 does not require the evaluated agent to adopt a particular internal tool API.

## File 4: `src/my_agent_tau2/driver.py`

The following complete driver launches a local adapter process that speaks newline-delimited JSON on stdin and stdout. This is a useful default when the evaluated agent is a binary or has a proprietary SDK: the small adapter process wraps that existing interface, while tau2 imports only this Python module.

```python
from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from tau2.agent import ExternalAgentContext


class JsonLineDriver:
    def __init__(
        self,
        *,
        command: list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        tool_timeout_seconds: float = 120,
    ) -> None:
        if not command:
            raise ValueError("command must not be empty")
        self._command = command
        self._cwd = Path(cwd).resolve() if cwd is not None else None
        self._env = env or {}
        self._tool_timeout_seconds = tool_timeout_seconds
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._stopped = False

    def start(self, context: ExternalAgentContext) -> None:
        process = subprocess.Popen(
            self._command,
            cwd=self._cwd,
            env={**os.environ, **self._env},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        with self._lock:
            if self._process is not None or self._stopped:
                process.terminate()
                raise RuntimeError("driver is already started or stopped")
            self._process = process

        headers = {"Authorization": context.tool_endpoint.authorization_header}
        tools_response = httpx.get(
            f"{context.tool_endpoint.url}/tools",
            headers=headers,
            timeout=self._tool_timeout_seconds,
        )
        tools_response.raise_for_status()
        self._write(
            {
                "type": "start",
                "task_id": context.task_id,
                "domain_policy": context.domain_policy,
                "message_history": [
                    message.model_dump(mode="json")
                    for message in context.message_history
                ],
                "tools": tools_response.json()["tools"],
                "tool_endpoint": {
                    "url": context.tool_endpoint.url,
                    "authorization_header": (
                        context.tool_endpoint.authorization_header
                    ),
                },
            }
        )
        reply = self._read()
        if reply.get("type") != "ready":
            raise RuntimeError(f"expected ready response, received {reply!r}")

    def respond(self, user_message: str) -> str:
        self._write({"type": "turn", "content": user_message})
        reply = self._read()
        if reply.get("type") != "response":
            raise RuntimeError(f"expected response message, received {reply!r}")
        content = reply.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("agent returned empty response text")
        return content.strip()

    def stop(self) -> None:
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            process = self._process
            self._process = None
        if process is None:
            return
        try:
            if process.poll() is None and process.stdin is not None:
                try:
                    process.stdin.write('{"type":"stop"}\n')
                    process.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
        finally:
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()

    def _write(self, message: dict[str, Any]) -> None:
        with self._lock:
            process = self._process
            stopped = self._stopped
        if process is None or stopped or process.stdin is None:
            raise RuntimeError("agent process is not running")
        process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        process.stdin.flush()

    def _read(self) -> dict[str, Any]:
        with self._lock:
            process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("agent process is not running")
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(
                f"agent process exited before replying (code={process.poll()})"
            )
        message = json.loads(line)
        if not isinstance(message, dict):
            raise RuntimeError("agent process response must be a JSON object")
        return message


def create_driver(**kwargs: Any) -> JsonLineDriver:
    return JsonLineDriver(**kwargs)
```

The adapter process must reserve stdout for protocol messages. Send diagnostics to stderr. It must not log the startup message because that message contains the task-scoped authorization header.

Tau2 enforces the outer startup and turn timeouts. On timeout it calls `stop()`, so `stop()` must be able to terminate the process and unblock a pending `start()` or `respond()` read. Do not hold a lock across a blocking agent call if that would prevent `stop()` from acquiring it.

## The adapter process protocol

The JSON Lines example above expects these messages:

1. The driver sends one `start` object containing `task_id`, `domain_policy`, `message_history`, OpenAI-compatible `tools`, and `tool_endpoint`.
2. The adapter installs the supplied schemas in the evaluated agent's tool registry and answers `{"type":"ready"}`.
3. For every user turn, the driver sends `{"type":"turn","content":"..."}`.
4. The adapter runs the agent until its final customer-facing text is ready, waits for all tool HTTP requests to finish, and answers `{"type":"response","content":"..."}`.
5. The driver may send `{"type":"stop"}` and then terminate the process during cleanup.

Only the small process wrapper is agent-specific. Its implementation normally performs the following mapping:

```python
# Pseudocode inside your adapter process; replace these calls with the
# evaluated agent's documented CLI, SDK, RPC, MCP, or plugin API.
agent = AgentSdk.start_session(
    policy=start_message["domain_policy"],
    history=start_message["message_history"],
)
for schema in start_message["tools"]:
    agent.register_tool(
        schema=schema,
        callback=lambda name, args: tau2_tools.call(name, args),
    )
final_text = agent.run_turn(turn_message["content"])
```

If the black-box agent already exposes session and turn endpoints, the Python driver can call those endpoints directly and omit the JSON Lines process. Keep the same three tau2 lifecycle methods and the same tool-gateway rules.

## File 5: `external-agent.json`

The configuration names the importable factory and passes only JSON-compatible values:

```json
{
  "driver": "my_agent_tau2:create_driver",
  "driver_args": {
    "command": ["my-agent-tau2-bridge"],
    "cwd": "/absolute/path/to/my-agent",
    "env": {
      "MY_AGENT_PROFILE": "tau2"
    },
    "tool_timeout_seconds": 120
  },
  "startup_timeout": 60,
  "turn_timeout": 900
}
```

Use absolute paths for worker runs. Do not place API keys in this JSON because run configuration and process arguments may be saved or printed. Read secrets from the worker environment or a secret manager instead.

## File 6: `tests/test_driver.py`

At minimum, verify that tau2 can import the factory and that the returned object satisfies the runtime protocol:

```python
from tau2.agent import ExternalAgentDriver

from my_agent_tau2 import create_driver


def test_factory_returns_external_agent_driver() -> None:
    driver = create_driver(command=["my-agent-tau2-bridge"])
    assert isinstance(driver, ExternalAgentDriver)
    driver.stop()
    driver.stop()
```

Add adapter-owned tests for successful startup, one tool call, tool errors, agent-process exit, partial startup, repeated stop, and stopping while `start()` or `respond()` is blocked. Tau2's own interface tests are in [`tests/test_external_agent.py`](../tests/test_external_agent.py).

## Install the adapter

Install tau2 and the adapter into the same environment:

```bash
cd /path/to/tau2-bench
uv sync --extra dev
uv pip install -e /path/to/my-agent-tau2

uv run python -c \
  'from my_agent_tau2 import create_driver; print(create_driver)'
uv run pytest tests/test_external_agent.py -q
```

The import check must succeed before starting a paid evaluation. In controller mode, every worker must use an environment in which the same module and external executable are installed.

## Run the first evaluation

Because `--external-agent-config` accepts JSON, compact the configuration file when passing it on the command line:

```bash
cd /path/to/tau2-bench
uv run tau2 run \
  --domain mock \
  --task-ids create_task_1 \
  --agent external_agent \
  --external-agent-config "$(jq -c . /path/to/my-agent-tau2/external-agent.json)" \
  --user-llm openai/gpt-4.1 \
  --num-trials 1 \
  --max-concurrency 1 \
  --workers 0 \
  --timeout 1200 \
  --save-to my-agent-external-smoke \
  --log-level INFO
```

Replace the user model with any LiteLLM identifier for which the corresponding provider key is configured. The evaluated external agent's model and credentials are controlled by its own launcher or service, not by `--agent-llm`.

Inspect `data/simulations/my-agent-external-smoke/results.json` with `tau2 view`. A successful `create_task_1` trajectory contains the native `create_task` tool call followed by the external agent's text response.

## Tool gateway rules

The endpoint in `ExternalAgentContext.tool_endpoint` has these routes:

- `GET /health` checks availability and requires the Bearer token.
- `GET /tools` returns `{"tools": [...]}` with OpenAI-compatible function schemas and requires the Bearer token.
- `POST /tools/{tool_name}` accepts the tool arguments object as its JSON body and requires the Bearer token.

The gateway binds to `127.0.0.1` on an ephemeral port. A process on the same host can call it directly. A container must be configured so that its adapter can reach the host loopback endpoint, or the Python driver must provide a narrowly scoped local relay. A remote service cannot call the loopback URL directly; keep a local sidecar next to tau2 and relay only the current task's authenticated calls.

Tau2 allows POST calls only during `respond()`. Calls made in `start()`, after `respond()` returns, or from a detached background task receive HTTP 409. The driver must not return final text until every tool request caused by that turn has completed.

Every simulation receives a different endpoint and token. Never reuse them across tasks, store them in a global singleton, write them into a workspace, place them in result metadata, or send them to unrelated services.

## Lifecycle requirements

`start(context)` must create exactly one isolated agent session. Apply `context.domain_policy` as authoritative instructions, restore `context.message_history` when it is non-empty, configure the supplied tools, and return only when the session can accept a user turn.

`respond(user_message)` must pass only that user text into the existing session, let the agent perform zero or more tool calls, wait for all those calls, and return non-empty customer-facing text. Do not return internal reasoning, tool protocol messages, or a structured object.

`stop()` must be idempotent, safe after partial startup, and able to unblock a pending lifecycle call. Close the agent before closing any local tool adapter so in-flight agent work observes a controlled shutdown.

Do not share conversation state, endpoints, tokens, workspaces, or mutable clients between driver instances. Concurrency and controller workers rely on this isolation.

## Timeout ordering

The adapter's internal request timeout should be no greater than `turn_timeout`, and `turn_timeout` should be lower than the tau2 simulation `--timeout`:

```text
agent SDK or HTTP timeout <= external turn_timeout < tau2 --timeout
```

This order lets the nearest layer report the useful failure before an outer layer terminates the simulation. Startup has its own `startup_timeout`.

## Security checklist

- Bind protocol adapters and relays to loopback unless the deployment supplies an equivalent authenticated private channel.
- Require authentication on every locally translated tool request.
- Keep model keys in the agent environment or secret manager, not in `driver_args`.
- Never log the tau2 authorization header or the complete `start` protocol message.
- Do not expose the task-scoped gateway as a general-purpose proxy.
- Tear down the endpoint, relay, session, and temporary workspace after every task.

## Release-readiness checklist

Before publishing an adapter, verify all of the following:

- The factory is importable through the exact `module:attribute` string.
- Every `driver_args` value survives JSON serialization.
- One task creates one independent driver and one independent agent session.
- Domain policy and prior message history reach the agent exactly once.
- `GET /tools` schemas are registered without renaming the tau2 tool names.
- Successful and failed tool calls both return to the agent in its supported result format.
- `respond()` waits for all of its tool calls before returning final text.
- Startup failure, process exit, turn timeout, partial startup, and repeated `stop()` clean up resources.
- The adapter works with `--workers 0` before it is tested with controller workers.
- The mock task produces a standard tau2 tool trajectory and a readable `results.json`.
- No result, log, workspace, or config file contains a model key or gateway token.

After the mock run passes, test a small `base` sample in `airline`, `retail`, or `telecom`, then increase trials and concurrency. Record the adapter revision, agent version, model configuration, tau2 revision, domain split, seed, trial count, and user simulator model with published scores.

## Troubleshooting

`Cannot import external agent driver module` means the adapter is not installed in the Python environment running tau2 or a worker. Run the import check from that exact environment.

HTTP 401 means the gateway request omitted or changed `context.tool_endpoint.authorization_header`. Treat the value as opaque and send it unchanged.

HTTP 409 means a tool call happened outside an active `respond()` call. Wait for all agent tool activity before returning and stop detached background work.

An external-agent startup or turn timeout means tau2 called `stop()` after the configured limit. Confirm that `stop()` can interrupt the adapter's blocking operation, then align the inner and outer timeout values.

A fluent response with reward zero usually means the required environment state was not changed or required information was not communicated. Inspect the native tool calls, tool results, final database state, and evaluation assertions in `tau2 view`.

See [`src/tau2/agent/external.py`](../src/tau2/agent/external.py) for the authoritative Python interfaces and [`src/tau2/agent/README.md`](../src/tau2/agent/README.md) for the broader agent architecture.
