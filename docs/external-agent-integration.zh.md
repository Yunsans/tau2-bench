# 外部整机智能体接入指南

[English](external-agent-integration.md) | 中文

当被测智能体运行在 tau2 Python 进程之外、拥有自己的模型循环和工具系统，或者只能以二进制、本机应用、容器或服务形式访问时，使用本指南。Tau2 不要求访问智能体源码，但智能体必须具备可控制的会话接口，并能调用 tau2 提供的逐任务工具。

内置 `external_agent` 是文本半双工接口。用户模拟器、领域状态、工具执行、标准轨迹、重试、checkpoint 和评分仍由 tau2 负责。

## 接入结构

```text
tau2 run
  -> ExternalAgent
     -> 用户提供的 Python driver
        -> 不透明的智能体进程、应用、容器或服务
           -> 智能体自己的模型循环和工具注册表
        -> tau2 的 task-scoped HTTP 工具 gateway
     -> 标准 tau2 轨迹与 evaluator
```

每个 simulation 创建一个 driver 实例。Tau2 调用一次 `start(context)`，每个用户话轮调用一次 `respond(user_message)`，并在正常结束、失败或超时时调用 `stop()`。

## Tau2 的最低要求

Tau2 侧唯一必需的产物是可导入的 Python factory。Factory 必须接收可 JSON 序列化的关键字参数，并返回包含以下方法的对象：

```python
class ExternalAgentDriver:
    def start(self, context: ExternalAgentContext) -> None: ...
    def respond(self, user_message: str) -> str: ...
    def stop(self) -> None: ...
```

Driver 可以把该生命周期转换成被测智能体已有的任意接口。只要智能体可通过稳定的 CLI、SDK、HTTP API、JSON-RPC API、MCP server、ACP server 或其他已知协议启动或访问，就不需要源码。

如果智能体既没有可编程会话接口，也无法安装或调用动态工具，则必须先在智能体外围补充其中至少一项能力，否则 adapter 无法完成测评接入。

## 推荐的 adapter 包

智能体专用代码应放在 tau2 之外，使 benchmark 保持通用。建议使用以下小型包结构：

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

这些文件名是建议，不是协议要求。`driver.py` 及其可导入 factory 是必需部分。当智能体能注册普通 callable tools 时，`tau2_tools.py` 很有用。`external-agent.json` 让运行配置易于审查。

## 文件 1：`pyproject.toml`

该包会把 driver 安装到与 tau2 相同的 Python 环境。示例使用 `httpx` 调用逐任务工具 gateway，使用 pytest 测试 adapter。

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

不要把模型 API key 或 tau2 gateway token 写进此文件。

## 文件 2：`src/my_agent_tau2/__init__.py`

导出运行配置所引用的 factory：

```python
from my_agent_tau2.driver import create_driver

__all__ = ["create_driver"]
```

## 文件 3：`src/my_agent_tau2/tau2_tools.py`

下面的完整 client 会列出当前任务的 OpenAI-compatible function schemas，并按 tau2 原始名称调用工具。智能体专用 bridge 可以把每个 schema 注册进自己的工具系统，再把调用转发给 `call()`。

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

启动期间可以调用 `GET /health` 和 `GET /tools`。只有 tau2 正在执行相应 `respond()` 时，`POST /tools/{name}` 才会被接受。POST 响应是序列化后的 tau2 `ToolMessage`，因此工具级失败也可能使用成功的 HTTP 状态码，而 JSON body 会报告工具错误。

如果智能体原生支持 MCP、OpenAPI、plugin 或其他工具协议，应让 Python driver 保留同一 tau2 端点，并增加本机协议 adapter。例如 DeepSeek Harness 使用带鉴权的 HTTP-to-MCP proxy。转换层由智能体决定；tau2 不要求被测智能体采用特定内部工具 API。

## 文件 4：`src/my_agent_tau2/driver.py`

英文主文档给出了一个可直接复制的完整 `JsonLineDriver`，它启动本机 adapter 进程，并通过 stdin/stdout 交换 newline-delimited JSON。为避免中英文两份长代码发生偏差，请直接复制[英文文档中的 `driver.py`](external-agent-integration.md#file-4-srcmy_agent_tau2driverpy)。

这个方案适合只有二进制或专用 SDK 的智能体：小型 adapter 进程包装其既有接口，tau2 只导入 Python driver。Adapter 必须把 stdout 专用于协议消息，把诊断输出到 stderr；不得记录包含逐任务 authorization header 的启动消息。

Tau2 负责外层 startup 和 turn timeout。超时后 tau2 会调用 `stop()`，所以 `stop()` 必须能终止进程并解除正在等待的 `start()` 或 `respond()`。不要在阻塞的智能体调用期间持有会阻止 `stop()` 获取的锁。

## Adapter 进程协议

完整示例使用以下 JSON Lines 消息：

1. Driver 发送一个 `start` 对象，其中包含 `task_id`、`domain_policy`、`message_history`、OpenAI-compatible `tools` 和 `tool_endpoint`。
2. Adapter 把 schema 安装进被测智能体的工具注册表，再返回 `{"type":"ready"}`。
3. 每个用户话轮由 driver 发送 `{"type":"turn","content":"..."}`。
4. Adapter 运行智能体直到生成最终面向用户的文本，等待所有工具 HTTP 请求结束，再返回 `{"type":"response","content":"..."}`。
5. 清理时 driver 可以发送 `{"type":"stop"}`，随后终止进程。

只有这个小型进程 wrapper 与智能体有关。其实现通常完成如下映射：

```python
# Adapter 进程中的伪代码；把这些调用替换为被测智能体已有的
# CLI、SDK、RPC、MCP 或 plugin API。
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

如果黑盒智能体已经提供 session 与 turn API，Python driver 可以直接调用这些 API，并省略 JSON Lines 进程。仍需保持相同的三个 tau2 生命周期方法和工具 gateway 规则。

## 文件 5：`external-agent.json`

配置文件引用可导入 factory，并且只传递 JSON-compatible 值：

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

Worker 运行应使用绝对路径。不要把 API key 放进该 JSON，因为运行配置和进程参数可能被保存或打印。应让 worker 从环境变量或 secret manager 读取密钥。

## 文件 6：`tests/test_driver.py`

最低限度应验证 tau2 能导入 factory，且返回对象满足 runtime protocol：

```python
from tau2.agent import ExternalAgentDriver

from my_agent_tau2 import create_driver


def test_factory_returns_external_agent_driver() -> None:
    driver = create_driver(command=["my-agent-tau2-bridge"])
    assert isinstance(driver, ExternalAgentDriver)
    driver.stop()
    driver.stop()
```

Adapter 自己还应覆盖：成功启动、一次工具调用、工具错误、智能体进程退出、部分启动、重复停止，以及在 `start()` 或 `respond()` 阻塞时停止。Tau2 自身的接口测试位于 [`tests/test_external_agent.py`](../tests/test_external_agent.py)。

## 安装 adapter

把 tau2 与 adapter 安装进同一个环境：

```bash
cd /path/to/tau2-bench
uv sync --extra dev
uv pip install -e /path/to/my-agent-tau2

uv run python -c \
  'from my_agent_tau2 import create_driver; print(create_driver)'
uv run pytest tests/test_external_agent.py -q
```

开始付费测评前，import 检查必须成功。在 controller 模式下，每个 worker 都必须使用已安装相同 module 和外部 executable 的环境。

## 运行第一次测评

`--external-agent-config` 接收 JSON，因此在命令行传递配置文件时先压缩它：

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

用户模型可以换成任意已配置对应供应商 key 的 LiteLLM 标识。外部被测智能体的模型与密钥由它自己的 launcher 或 service 管理，不使用 `--agent-llm`。

通过 `tau2 view` 查看 `data/simulations/my-agent-external-smoke/results.json`。成功的 `create_task_1` 轨迹包含原生 `create_task` 工具调用，以及外部智能体返回的文本。

## 工具 gateway 规则

`ExternalAgentContext.tool_endpoint` 中的端点提供以下路由：

- `GET /health`：检查可用性，必须携带 Bearer token。
- `GET /tools`：返回 `{"tools": [...]}` 形式的 OpenAI-compatible function schemas，必须携带 Bearer token。
- `POST /tools/{tool_name}`：JSON body 为工具参数对象，必须携带 Bearer token。

Gateway 绑定在 `127.0.0.1` 的随机端口。同一主机上的进程可以直接访问。容器必须配置为能从 adapter 访问宿主 loopback 端点，或者由 Python driver 提供严格限域的本机 relay。远程服务无法直接访问该 loopback URL；应在 tau2 所在机器保留本机 sidecar，只中继当前任务的鉴权调用。

Tau2 只在 `respond()` 期间允许 POST。`start()` 内、`respond()` 返回后或分离后台任务发起的调用都会收到 HTTP 409。Driver 返回最终文本前必须等待该话轮引发的所有工具请求结束。

每个 simulation 都有不同的端点和 token。不得跨任务复用、放入全局 singleton、写进 workspace、放入结果 metadata 或发送给无关服务。

## 生命周期要求

`start(context)` 必须创建恰好一个隔离智能体 session，将 `context.domain_policy` 作为权威指令，在 `context.message_history` 非空时恢复历史，配置所提供工具，并且只在 session 已可接收用户话轮后返回。

`respond(user_message)` 只把该用户文本送进既有 session，让智能体执行零个或多个工具调用，等待所有调用结束，再返回非空且面向客户的文本。不得返回内部推理、工具协议消息或结构化对象。

`stop()` 必须幂等，在部分启动后仍安全，并能解除待处理的生命周期调用。应先关闭智能体，再关闭本机工具 adapter，使执行中的智能体工作能观察到受控停止。

不同 driver 实例不得共享对话状态、端点、token、workspace 或可变 client。并发和 controller worker 依赖这种隔离。

## 超时顺序

Adapter 内部请求超时不应大于 `turn_timeout`，而 `turn_timeout` 应小于 tau2 simulation 的 `--timeout`：

```text
agent SDK 或 HTTP timeout <= external turn_timeout < tau2 --timeout
```

此顺序可让最近的执行层先报告有用错误，外层再终止 simulation。启动有独立的 `startup_timeout`。

## 安全清单

- 协议 adapter 和 relay 应绑定 loopback；只有部署提供同等的带鉴权私有通道时才能例外。
- 每个本机转换后的工具请求都必须鉴权。
- 模型 key 保存在智能体环境或 secret manager，不放进 `driver_args`。
- 不得记录 tau2 authorization header 或完整 `start` 协议消息。
- 不得把 task-scoped gateway 暴露成通用 proxy。
- 每题结束后清理端点、relay、session 和临时 workspace。

## 开源前验收清单

- Factory 可通过准确的 `module:attribute` 字符串导入。
- 每个 `driver_args` 值都能完成 JSON 序列化。
- 每题创建独立 driver 与独立智能体 session。
- Domain policy 和既有 message history 恰好送入智能体一次。
- `GET /tools` schema 注册时不重命名 tau2 工具名。
- 成功和失败的工具调用都会用智能体支持的结果格式返回。
- `respond()` 返回最终文本前会等待该话轮的所有工具调用。
- 启动失败、进程退出、话轮超时、部分启动与重复 `stop()` 都会清理资源。
- Adapter 先在 `--workers 0` 下成功，再测试 controller worker。
- Mock 任务会生成标准 tau2 工具轨迹和可读取的 `results.json`。
- 结果、日志、workspace 和配置文件都不包含模型 key 或 gateway token。

Mock 通过后，先测试 `airline`、`retail` 或 `telecom` 的少量 `base` 样本，再增加 trial 和并发。发布分数时记录 adapter revision、智能体版本、模型配置、tau2 revision、domain split、seed、trial 数和用户模拟器模型。

## 故障排查

`Cannot import external agent driver module` 表示运行 tau2 或 worker 的 Python 环境没有安装 adapter。应从该准确环境运行 import 检查。

HTTP 401 表示 gateway 请求遗漏或改变了 `context.tool_endpoint.authorization_header`。该值应视为 opaque，并原样发送。

HTTP 409 表示工具调用发生在活跃 `respond()` 之外。返回前等待所有智能体工具活动，并停止分离的后台工作。

External-agent startup 或 turn timeout 表示 tau2 在配置时限后调用了 `stop()`。先确认 `stop()` 能打断 adapter 的阻塞操作，再对齐内外层超时。

回答流畅但 reward 为零，通常表示必要环境状态没有改变或必要信息没有传达。通过 `tau2 view` 检查原生工具调用、工具结果、最终数据库状态与 evaluation assertions。

权威 Python 接口见 [`src/tau2/agent/external.py`](../src/tau2/agent/external.py)，完整 agent 架构见 [`src/tau2/agent/README.md`](../src/tau2/agent/README.md)。
