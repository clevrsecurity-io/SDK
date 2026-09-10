# @clevr/sdk

Runtime governance for AI agents. Every agent action passes through a policy
gate (`allow` / `block` / `step-up`) and is sealed in a tamper-evident
Ed25519 audit chain.

```bash
npm install @clevr/sdk
```

## Quick start (raw code)

```js
import { Clevr } from '@clevr/sdk';

const clevr = new Clevr({
  apiKey: process.env.CLEVR_API_KEY,
  agent:  'oncall-agent',
  mode:   'shadow',          // observe-only until you trust the policies
});

const result = await clevr.guard(
  { tool: 'kb.search', target: 'kb/incidents', action_type: 'read',
    action: 'search for incidents related to login spike' },
  async () => kb.search('login spike'),
);
```

In `enforce` mode (default), `guard()` throws `ClevrBlockedError` if the
policy blocks the action, or `ClevrEscalatedError` if it requires human
approval. In `shadow` mode, the inner function always runs and the verdict
is recorded silently — flip to `enforce` once you've watched a few days of
audit and you're confident.

## Framework adapters

| Framework | Import | Hook |
|-----------|--------|------|
| Anthropic Messages | `@clevr/sdk/anthropic` | `withClevr(anthropic, clevr).runAgent({...})` — gates every `tool_use` block |
| OpenAI Chat Completions | `@clevr/sdk/openai` | `withClevr(openai, clevr).runAgent({...})` — gates every `tool_calls` entry |
| Vercel AI SDK | `@clevr/sdk/vercel-ai` | `guardTools(clevr, tools)` — wraps each tool's `execute` |
| LangChain.js | `@clevr/sdk/langchain` | `guard(clevr, tool)` or `ClevrCallbackHandler` |
| MCP server | `@clevr/sdk/mcp` | `wrapMcpHandler(handler, { clevr })` |

Each adapter routes the framework's tool-execution point through
`clevr.guard()` — so block / escalate / allow happen at the smallest unit
of agent side-effect, and the model sees the verdict as a tool result it
can react to (rather than a crash).

The Anthropic and OpenAI adapters additionally forward the running
conversation (system prompt + turns) with each tool evaluation, so the
engine's content floor scans the prompt itself (PII, secrets, prompt
injection), not just the tool name and arguments. The tool-level adapters
(Vercel AI, LangChain, MCP, Claude Agent SDK) gate at the tool boundary,
where the full conversation is not available; if you want every prompt
evaluated, wrap the model client with `wrapModel(client, clevr)`, which
records each prompt as a `chat` decision and runs the prompt guardrails
before the model is even called.

## Actor chains

When a human triggers an agent that spawns sub-agents, every hop is sealed:

```js
const root = new Clevr({ agent: 'oncall', apiKey });
root.withRootHuman('alice@acme.com', 'Alice · On-call lead');

const investigator = root.child('investigator');
await investigator.guard({...}, async () => {...});
// → actor_chain on the decision row:
//   [{type:'human', id:'alice@acme.com'}, {type:'agent', id:'oncall'},
//    {type:'agent', id:'investigator', on_behalf_of:'oncall'}]
```

The chain is verified hop-by-hop by the engine: an investigator agent
cannot take an action its parent isn't authorized for.

## Modes

| Mode | What happens |
|------|--------------|
| `enforce` (default) | Block → throws. Escalate → throws (or polls if `onEscalate: 'wait'`). Allow → runs. |
| `shadow` | Always runs the inner function. The verdict is recorded server-side and exposed at `clevr.lastVerdict`. Use this for the first 1-2 weeks of a deployment to find tuning gaps before flipping to enforce. |

## License

Apache-2.0 © Clevr Security.
