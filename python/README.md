# clevr-py

Runtime governance for AI agents — Python. Every agent action passes through
a policy gate (`allow` / `block` / `step-up`) and is sealed in a
tamper-evident Ed25519 audit chain.

```bash
# Not on PyPI yet: install from this directory.
pip install /path/to/clevr/sdk/python
```

## Quick start (raw code)

```python
from clevr import Clevr

clevr = Clevr(
    api_key="clevr_sk_...",
    agent="oncall-agent",
    mode="shadow",  # observe-only until you trust the policies
)

result = clevr.guard(
    {"tool": "kb.search", "target": "kb/incidents", "action_type": "read",
     "action": "search for incidents related to login spike"},
    run=lambda _v: kb.search("login spike"),
)
```

In `enforce` mode (default), `guard()` raises `ClevrBlockedError` on block
and `ClevrEscalatedError` on escalation. In `shadow` mode, the inner function
always runs and the verdict is recorded silently — flip to `enforce` once
you've watched a few days of audit and you're confident.

## Framework adapters

| Framework | Import | Hook |
|-----------|--------|------|
| LangChain | `clevr_langchain` | `@guarded(...)` decorator or `ClevrCallbackHandler` |
| CrewAI | `clevr_crewai` | `guard(clevr, tool)` for `BaseTool`, `@guarded(...)` for `@tool` |
| Pydantic AI | `clevr_pydantic_ai` | `@guarded(...)` decorator or `guard_agent(clevr, agent)` |

Each adapter routes the framework's tool-execution point through
`clevr.guard()` — so block / escalate / allow happen at the smallest unit of
agent side-effect, and the model sees the verdict as a tool result it can
react to (rather than a crash).

### LangChain

```python
from langchain_core.tools import tool
from clevr import Clevr
from clevr_langchain import guarded

clevr = Clevr(agent="researcher")

@guarded(clevr, tool_name="kb_search", action_type="read")
@tool
def kb_search(query: str) -> str:
    return my_kb.search(query)
```

### CrewAI

```python
from crewai.tools import BaseTool
from clevr import Clevr
from clevr_crewai import guard

class KbSearch(BaseTool):
    name = "kb_search"
    description = "Search the knowledge base"
    def _run(self, query: str) -> str:
        return my_kb.search(query)

clevr = Clevr(agent="researcher")
agent = Agent(role="Researcher", tools=[guard(clevr, KbSearch())])
```

### Pydantic AI

```python
from pydantic_ai import Agent, RunContext
from clevr import Clevr
from clevr_pydantic_ai import guarded

clevr = Clevr(agent="researcher")
agent = Agent("openai:gpt-4o", system_prompt="You are a researcher.")

@agent.tool
@guarded(clevr, tool_name="kb_search", action_type="read")
async def kb_search(ctx: RunContext, query: str) -> str:
    return await my_kb.search(query)
```

## Actor chains

When a human triggers an agent that spawns sub-agents, every hop is sealed:

```python
root = Clevr(agent="oncall", api_key=key)
root.with_root_human("alice@acme.com", display="Alice · On-call lead")

investigator = root.child("investigator")
investigator.guard({...}, run=lambda _v: investigate(...))
```

## Platform

Every decision records the platform the agent runs on, so the console says
where an action came from. Each adapter names its own framework: `langchain`,
`crewai`, `autogen`, `llamaindex`, `pydantic-ai` or `bedrock`. When the agent
runs on something the adapter cannot see, name it once on the client:

```python
clevr = Clevr(agent="planner", runtime="langgraph")  # or CLEVR_RUNTIME=langgraph
```

The client's value wins over the adapter's, and a sub-agent made with
`child()` keeps it. The platform is recorded, never judged: no verdict
depends on it.

## Modes

| Mode | What happens |
|------|--------------|
| `enforce` (default) | Block → raises. Escalate → raises (or polls if `on_escalate='wait'`). Allow → runs. |
| `shadow` | Always runs the inner function. The verdict is recorded server-side and exposed at `clevr.last_verdict`. |

## License

Apache-2.0 © Clevr Security.
