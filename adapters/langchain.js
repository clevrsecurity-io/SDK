// @clevr/sdk/langchain — drop-in LangChain.js integration for Clevr.
//
// LangChain.js exposes tools via two surfaces:
//   1. The `tool(fn, { name, schema })` factory (langchain/tools or
//      @langchain/core/tools) — produces a DynamicStructuredTool.
//   2. Custom subclasses of StructuredTool / BaseToolkit.
//
// We provide a single `guard(clevr, tool, meta?)` helper that wraps any
// tool with a Clevr policy gate. It works on:
//   • a DynamicStructuredTool returned by tool(...)
//   • a plain function (then you pass a name + schema)
//   • a class-based StructuredTool instance (we monkey-patch _call/.invoke)
//
// We also expose `ClevrCallbackHandler` for the cases where you cannot easily
// reach the tool definitions (LangGraph, AgentExecutor pulling from a
// registry). It duck-types LangChain's BaseCallbackHandler shape so it can
// be passed via `{ callbacks: [handler] }` without importing langchain.

import { ClevrBlockedError, ClevrEscalatedError } from '../index.js';

/**
 * Wrap a single LangChain tool with a Clevr policy gate.
 *
 * @param {object} clevr  — Clevr client
 * @param {object} tool   — DynamicStructuredTool or StructuredTool instance
 * @param {object} meta   — optional: { tool_name, action_type, target, environment }
 *                          target may be a string OR a fn(args) → string
 */
export function guard(clevr, tool, meta = {}) {
  const name = meta.tool_name || tool.name || tool.lc_namespace?.join('.') || 'unknown_tool';

  // LangChain v0.2+ uses .invoke({input}) → calls _call internally; we wrap
  // _call so both .invoke({...}) and direct .call({...}) end up gated.
  const originalCall = tool._call?.bind(tool);
  const originalInvoke = tool.invoke?.bind(tool);

  async function gated(args, runManager) {
    const target = typeof meta.target === 'function' ? meta.target(args) : meta.target;
    try {
      return await clevr.guard({
        tool: name,
        // The platform, unless the client names one (a LangGraph agent sets
        // runtime: 'langgraph' on its client: this adapter cannot tell).
        runtime: clevr.runtime || 'langchain',
        action_type: meta.action_type || 'tool_call',
        action: `${name}(${safeJson(args)})`,
        target: target || null,
        environment: meta.environment || null,
        metadata: { args },
      }, async () => {
        if (originalCall) return await originalCall(args, runManager);
        if (originalInvoke) return await originalInvoke(args, runManager);
        throw new Error(`Clevr langchain: tool "${name}" has no _call/invoke implementation.`);
      });
    } catch (e) {
      if (e instanceof ClevrBlockedError) {
        return `[Clevr blocked this action: ${e.verdict?.reason || e.message}]`;
      }
      if (e instanceof ClevrEscalatedError) {
        return `[Clevr requires human approval before this action runs: ${e.verdict?.reason || e.message}]`;
      }
      throw e;
    }
  }

  if (originalCall) tool._call = gated;
  // .invoke wraps _call → automatically picks up the gate, no need to touch it.
  return tool;
}

/**
 * Wrap an array of LangChain tools by name. metaByName is { name: meta }.
 */
export function guardAll(clevr, tools, metaByName = {}) {
  return tools.map((t) => guard(clevr, t, metaByName[t.name] || {}));
}

/**
 * LangChain BaseCallbackHandler-compatible interceptor. Use when tools come
 * from a registry (AgentExecutor, LangGraph) and you can't reach them to
 * wrap individually. Duck-types the callback API so we don't depend on
 * langchain at import time.
 *
 *   const handler = new ClevrCallbackHandler(clevr);
 *   await agentExecutor.invoke({ input }, { callbacks: [handler] });
 */
export class ClevrCallbackHandler {
  constructor(clevr, { actionType = 'tool_call' } = {}) {
    this.clevr = clevr;
    this.actionType = actionType;
    this._verdicts = new Map(); // runId → verdict
    // LangChain identifies callback handlers via these properties.
    this.name = 'ClevrCallbackHandler';
    this.ignoreLLM = false; // also evaluate the prompt at the model boundary
    this.ignoreChain = true;
    // LangChain swallows what a handler throws (a console.warn) unless it sets
    // raiseError, and runs it in the background unless it sets awaitHandlers:
    // without both, a Block or a Hold thrown here never stopped the tool.
    this.raiseError = true;
    this.awaitHandlers = true;
  }

  // Model boundary: evaluate the PROMPT / conversation BEFORE the model runs, so
  // prompt-level guardrails (PII / secrets / injection) work in LangGraph /
  // AgentExecutor too — not only the wrapped tools. Throwing aborts the call.
  async handleChatModelStart(_llm, messages, _runId) {
    const first = Array.isArray(messages?.[0]) ? messages[0] : (messages || []);
    await this._evalPrompt(lcMessagesToConversation(first));
  }

  async handleLLMStart(_llm, prompts, _runId) {
    const convo = (Array.isArray(prompts) ? prompts : []).map((p) => ({ role: 'user', content: String(p) }));
    await this._evalPrompt(convo);
  }

  async _evalPrompt(conversation) {
    if (!conversation.length) return;
    // Situate the session: capture the first user turn as the session goal so a
    // LangChain session reads with meaning, like the model-loop adapters. A
    // caller-provided goal is respected.
    if (!this.clevr.sessionId && !this.clevr.sessionGoal) {
      const u = conversation.find((m) => m.role === 'user');
      if (u && typeof u.content === 'string') this.clevr.sessionGoal = u.content.slice(0, 300);
    }
    const action = {
      tool: 'llm.messages',
      runtime: this.clevr.runtime || 'langchain',
      action_type: 'chat',
      action: conversation.map((m) => m.content).join('\n'),
      conversation,
    };
    const verdict = await this.clevr.evaluate(action);
    if (this.clevr.mode === 'enforce') {
      if (verdict.effect === 'block') throw new ClevrBlockedError(verdict);
      const held = verdict.effect === 'escalate' || verdict.effect === 'step_up';
      if (held && this.clevr.onEscalate === 'throw') throw new ClevrEscalatedError(verdict);
      // The model waits for a person as a tool does: a held prompt reached the
      // model at once, one a person refused too.
      if (held && this.clevr.onEscalate === 'wait') await this.clevr.waitForDecision(verdict, action);
    } else if (verdict.effect !== 'allow') {
      this.clevr.lastVerdict = verdict;
    }
  }

  // LangChain passes the tool serialized without its name (id ends with the
  // class, DynamicStructuredTool); the name comes as runName.
  async handleToolStart(tool, input, runId, _parentRunId, _tags, _metadata, runName) {
    const name = tool?.name || runName || tool?.id?.slice(-1)?.[0] || 'unknown_tool';
    const text = typeof input === 'string' ? input : safeJson(input, Infinity);
    const args = toolArgs(input);
    const action = {
      tool: name,
      runtime: this.clevr.runtime || 'langchain',
      action_type: this.actionType,
      action: `${name}(${truncate(text, 200)})`,
      // Whole, for a rule on an argument (target.amount > 1000); the string
      // under metadata is only for display.
      ...(args ? { target_attr: args } : {}),
      metadata: { input: truncate(text, 500) },
    };
    const verdict = await this.clevr.evaluate(action);
    this._verdicts.set(String(runId), verdict);

    if (this.clevr.mode === 'enforce') {
      if (verdict.effect === 'block') throw new ClevrBlockedError(verdict);
      // 'step_up' is the legacy synonym for a held/escalate verdict (the gateway,
      // hooks and other adapters all treat it as a hold); without it a step_up
      // tool call would run instead of being held.
      const held = verdict.effect === 'escalate' || verdict.effect === 'step_up';
      if (held && this.clevr.onEscalate === 'throw') throw new ClevrEscalatedError(verdict);
      // Waits for a person, as guard() does: approved, the approval is spent by
      // asking again with it and LangChain runs the tool; rejected, timed out or
      // already waiting (held_as), it throws and the tool does not run.
      if (held && this.clevr.onEscalate === 'wait') {
        this._verdicts.set(String(runId), await this.clevr.waitForDecision(verdict, action));
      }
    }
  }

  async handleToolEnd(_output, runId) {
    const v = this._verdicts.get(String(runId));
    this._verdicts.delete(String(runId));
    if (v && this.clevr.mode === 'shadow' && v.effect !== 'allow') {
      this.clevr.lastVerdict = v;
    }
  }

  async handleToolError(_err, runId) {
    this._verdicts.delete(String(runId));
  }
}

function safeJson(v, n = 200) {
  try { return String(JSON.stringify(v)).slice(0, n); } catch { return String(v).slice(0, n); }
}
function truncate(s, n) { return s.length <= n ? s : s.slice(0, n - 1) + '…'; }

// A tool input longer than this is not parsed for its arguments.
const MAX_INPUT_READ = 100_000;

// The tool's arguments from what handleToolStart receives: LangChain passes the
// JSON of the arguments, or of the whole tool call on 0.3's tool.call(toolCall).
// Anything else (a string tool's text) has no arguments to read.
function toolArgs(input) {
  let v = input;
  if (typeof v === 'string') {
    const s = v.trim();
    if (!s.startsWith('{') || s.length > MAX_INPUT_READ) return null;
    try { v = JSON.parse(s); } catch { return null; }
  }
  const isObject = (o) => o != null && typeof o === 'object' && !Array.isArray(o);
  if (!isObject(v)) return null;
  return v.type === 'tool_call' && isObject(v.args) ? v.args : v;
}

// Duck-type LangChain BaseMessage[] → a [{role, content}] conversation without
// importing langchain. _getType() returns human/ai/system/tool; content is a
// string or an array of content parts.
function lcMessagesToConversation(msgs) {
  return (msgs || []).map((m) => {
    const t = typeof m?._getType === 'function' ? m._getType() : (m?.type || m?.role || 'user');
    const role = t === 'human' ? 'user'
      : t === 'ai' ? 'assistant'
      : t === 'system' ? 'system'
      : (t === 'tool' || t === 'function') ? 'tool'
      : 'user';
    const c = m?.content;
    const content = typeof c === 'string' ? c
      : Array.isArray(c) ? c.map((p) => (typeof p === 'string' ? p : (p?.text || ''))).join('\n')
      : '';
    return { role, content };
  });
}
