// @clevr/sdk — minimal core. Provides a `Clevr` client that wraps any
// tool execution with a pre-flight call to /v1/evaluate.
//
// Usage:
//   import { Clevr } from '@clevr/sdk';
//   const clevr = new Clevr({ apiKey, base, agent: 'researcher' });
//
//   // Wrap any tool execution. The inner function only runs if effect === 'allow'.
//   const result = await clevr.guard({
//     tool: 'kb.search', target: 'kb/churn',
//     action: 'search churn docs', action_type: 'read',
//   }, async () => {
//     return await kbClient.search('churn');
//   });
//
// On effect === 'block'  → throws ClevrBlockedError
// On effect === 'escalate' → behavior controlled by `onEscalate`:
//   - 'throw' (default): throws ClevrEscalatedError immediately
//   - 'wait':            polls the resolution endpoint until human resolves
//   - 'allow':           runs anyway (dev mode — NOT recommended in prod)
//
// Identity, role, delegation, session — all handled by the engine.
//
// Verifiable identity (optional): pass `identitySeed` (a raw-32 Ed25519 seed,
// base64) or set CLEVR_IDENTITY_SEED and every evaluate call is SIGNED, so the
// engine can prove this workload is who it claims (asserted → verified). Provision
// one with `generateIdentitySeed()`. Unsigned clients keep working as 'asserted'.
import { signIdentityProof } from './identity.js';
export { generateIdentitySeed, signIdentityProof } from './identity.js';

export class ClevrBlockedError extends Error {
  constructor(verdict) {
    super(verdict.reason || 'Action blocked by Clevr policy.');
    this.name = 'ClevrBlockedError';
    this.verdict = verdict;
  }
}

export class ClevrEscalatedError extends Error {
  constructor(verdict) {
    super(verdict.reason || 'Action requires human approval.');
    this.name = 'ClevrEscalatedError';
    this.verdict = verdict;
  }
}

export class Clevr {
  constructor(opts = {}) {
    this.base = (opts.base || process.env.CLEVR_URL || 'http://localhost:8787').replace(/\/$/, '');
    this.apiKey = opts.apiKey || process.env.CLEVR_API_KEY;
    if (!this.apiKey) throw new Error('Clevr SDK: apiKey is required (or set CLEVR_API_KEY).');
    this.agent = opts.agent || 'unnamed-agent';
    this.onEscalate = opts.onEscalate || 'throw'; // 'throw' | 'wait' | 'allow'
    this.fetch = opts.fetch || globalThis.fetch;
    this.sessionId = opts.sessionId || null;
    this.sessionGoal = opts.sessionGoal || null;
    this.parentRequestId = opts.parentRequestId || null;
    this.parentAuthority = opts.parentAuthority || null;
    // mode: 'enforce' (default) actually blocks / escalates the action.
    // 'shadow' records the decision server-side but ALWAYS lets the action
    // run — so teams can ship the SDK to production with zero behavior change,
    // gather the audit trail, see what WOULD have been blocked, and only
    // then flip to enforce. This is the single biggest adoption unlock.
    this.mode = opts.mode || 'enforce';
    if (this.mode !== 'enforce' && this.mode !== 'shadow') {
      throw new Error(`Clevr SDK: mode must be 'enforce' or 'shadow' (got "${this.mode}")`);
    }
    // Optional verifiable identity: a raw-32 Ed25519 seed (base64). When set,
    // every evaluate call is signed so the engine verifies WHO is acting.
    this.identitySeed = opts.identitySeed || process.env.CLEVR_IDENTITY_SEED || null;
    // Actor chain — root-first. Either set explicitly (e.g. on entry from a
    // human-facing handler) or built incrementally via .child().
    this.actorChain = Array.isArray(opts.actorChain) ? [...opts.actorChain] : null;
  }

  /**
   * Start a new session (or attach to one). Returns the session id so callers
   * can persist it across calls.
   */
  startSession({ goal } = {}) {
    this.sessionId = 'sess_' + Math.random().toString(36).slice(2, 12);
    if (goal) this.sessionGoal = goal;
    return this.sessionId;
  }

  /**
   * Initialize the actor chain — typically called once on entry into the
   * agent, capturing the originating human. Returns the chain so the caller
   * can extend it further if desired.
   *
   *   clevr.withRootHuman('alice@acme.com', 'Alice · Support Lead')
   *     // → [{type:'human', id:'alice@acme.com', display:'Alice · Support Lead'},
   *     //    {type:'agent', id:'oncall-agent', on_behalf_of:'alice@acme.com'}]
   */
  withRootHuman(id, display) {
    const human = { type: 'human', id };
    if (display) human.display = display;
    this.actorChain = [human, { type: 'agent', id: this.agent, on_behalf_of: id }];
    return this.actorChain;
  }

  /**
   * Spawn a child Clevr client representing a sub-agent in the same actor
   * chain. The child inherits session, base, apiKey and mode; its actorChain
   * is the parent's plus one new agent hop. This is Uber's "paved path" —
   * the secure path becomes the easy path.
   *
   *   const investigator = clevr.child('investigation-agent');
   *   await investigator.guard({...});
   */
  child(childAgentId) {
    const parentChain = this.actorChain || [{ type: 'agent', id: this.agent }];
    const last = parentChain[parentChain.length - 1];
    const newChain = [
      ...parentChain,
      { type: 'agent', id: childAgentId, on_behalf_of: last?.id || this.agent },
    ];
    return new Clevr({
      base: this.base, apiKey: this.apiKey, fetch: this.fetch,
      agent: childAgentId,
      mode: this.mode, onEscalate: this.onEscalate,
      sessionId: this.sessionId, sessionGoal: this.sessionGoal,
      actorChain: newChain,
    });
  }

  /**
   * Wrap a single tool execution with the policy gate.
   *
   * In `enforce` mode (default): allow runs, block throws, escalate behaves
   * per `onEscalate`.
   * In `shadow` mode: ALWAYS runs the inner function and returns its result,
   * regardless of the verdict. The verdict is still recorded server-side
   * (full audit chain) and exposed via `clevr.lastVerdict` so the caller
   * can log what WOULD have happened.
   *
   * @param {object} action  { tool, target, action, action_type, environment, target_attr?, metadata? }
   * @param {function} run   async (verdict) => result  — the inner tool execution
   */
  async guard(action, run) {
    const verdict = await this.evaluate(action);
    this.lastVerdict = verdict;

    if (this.mode === 'shadow') {
      // Record the verdict, surface it on the returned object, but never
      // block or wait. Adoption-first stance — see CLAUDE.md & spec.
      const result = await run(verdict);
      return { result, verdict, shadow: true,
               wouldHaveBlocked: verdict.effect === 'block',
               wouldHaveEscalated: verdict.effect === 'escalate' };
    }

    if (verdict.effect === 'allow') {
      return await run(verdict);
    }
    if (verdict.effect === 'block') {
      throw new ClevrBlockedError(verdict);
    }
    // escalate
    if (this.onEscalate === 'wait')  return await this.#waitThenRun(verdict, run);
    if (this.onEscalate === 'allow') return await run(verdict);
    throw new ClevrEscalatedError(verdict);
  }

  /**
   * Call /v1/evaluate without executing. Useful when the caller wants to
   * react to the verdict directly (e.g. branch on `effect`).
   */
  async evaluate(action) {
    const body = {
      agent: this.agent,
      session_id: this.sessionId, session_goal: this.sessionGoal,
      parent_request_id: this.parentRequestId,
      // Phase 1 — pass the actor chain so the engine can verify authority
      // hop-by-hop and the audit log records the full lineage.
      ...(this.actorChain ? { actor_chain: this.actorChain } : {}),
      ...action,
    };
    // Surface tool-call arguments as target_attr so the engine's deterministic
    // Layer-4 parameter rules (target.<field>, e.g. amount > 1000) can read them.
    // Adapters stash the raw args under metadata.input / metadata.args; promote
    // them unless the caller set target_attr explicitly.
    if (body.target_attr == null && body.metadata) {
      const a = body.metadata.input ?? body.metadata.args;
      if (a && typeof a === 'object' && !Array.isArray(a)) body.target_attr = a;
    }
    if (this.parentAuthority) {
      body.delegation = {
        parent_request_id: this.parentRequestId,
        parent_authority: this.parentAuthority,
      };
    }
    // Sign the request with this workload's identity key (if provisioned) so the
    // engine verifies WHO is acting. Bound to the exact agent + action so a
    // captured proof cannot be lifted onto a different agent or tool.
    if (this.identitySeed) {
      try {
        body.identity = signIdentityProof(this.identitySeed, {
          agent: body.agent, action_type: body.action_type,
          action: body.action, tool: body.tool, session_id: body.session_id,
        });
      } catch { /* never let signing break the call — stays 'asserted' */ }
    }
    const r = await this.fetch(`${this.base}/v1/evaluate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${this.apiKey}` },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const text = await r.text();
      throw new Error(`Clevr evaluate failed (${r.status}): ${text}`);
    }
    const v = await r.json();
    // Auto-track session and delegation context for the next hop.
    if (!this.sessionId && v.session_id) this.sessionId = v.session_id;
    if (v.request_id) this.lastRequestId = v.request_id;
    return v;
  }

  /**
   * Poll the engine for resolution of an escalated decision. Backoff up to
   * `timeoutMs` (default 10 min). When resolved=approved → runs the inner fn.
   */
  async #waitThenRun(verdict, run, timeoutMs = 10 * 60_000) {
    const start = Date.now();
    let delay = 1000;
    while (Date.now() - start < timeoutMs) {
      await new Promise((r) => setTimeout(r, delay));
      delay = Math.min(delay * 1.5, 15_000);
      const status = await this.#getDecision(verdict.decision_id);
      // The engine resolves a step-up by writing 'approved' / 'rejected' on
      // the decision (override route + server-side expiry sweep). We also
      // accept the raw override effects 'allow' / 'block' defensively so a
      // direct effect write still unblocks the agent.
      const res = status?.resolution;
      if (res === 'approved' || res === 'allow') return await run(verdict);
      if (res === 'rejected' || res === 'block') {
        throw new ClevrBlockedError({ ...verdict, reason: `Rejected by ${status.resolved_by}: ${status.resolution_reason}` });
      }
    }
    throw new ClevrEscalatedError({ ...verdict, reason: `Escalation timed out after ${Math.round(timeoutMs / 1000)}s.` });
  }

  async #getDecision(decisionId) {
    const r = await this.fetch(`${this.base}/v1/decisions/${decisionId}`, {
      headers: { Authorization: `Bearer ${this.apiKey}` },
    });
    if (!r.ok) return null;
    return await r.json();
  }
}

// Convenience factory.
export function createClient(opts) { return new Clevr(opts); }

// Re-export chain helpers so callers can build chains without the class.
export { Clevr as default };

/**
 * Build a 2-element chain from a human + agent. Convenience for entry points.
 *   chain.fromHuman('alice@acme.com', 'oncall-agent')
 *   → [{type:'human', id:'alice@acme.com'}, {type:'agent', id:'oncall-agent', on_behalf_of:'alice@acme.com'}]
 */
export const chain = {
  fromHuman(humanId, agentId, humanDisplay) {
    const human = { type: 'human', id: humanId };
    if (humanDisplay) human.display = humanDisplay;
    return [human, { type: 'agent', id: agentId, on_behalf_of: humanId }];
  },
  child(parentChain, agentId, type = 'agent') {
    const last = parentChain[parentChain.length - 1];
    return [...parentChain, { type, id: agentId, on_behalf_of: last?.id }];
  },
};

/**
 * Wrap an LLM client so EVERY model call is evaluated by Clevr BEFORE it runs.
 * This is how the SDK transport gets full-session visibility (each prompt is
 * recorded as a `chat` decision) AND prompt guardrails (PII / keyword / secret /
 * injection detectors run on the prompt text). One transport, both outcomes.
 *
 * Supports the Anthropic Messages client (client.messages.create) and the
 * OpenAI client (client.chat.completions.create). On `block` the model is
 * NEVER called (ClevrBlockedError); on `allow` it forwards; in shadow mode it
 * always forwards and records what WOULD have happened.
 *
 *   import Anthropic from '@anthropic-ai/sdk';
 *   import { Clevr, wrapModel } from '@clevr/sdk';
 *   const clevr = new Clevr({ agent: 'researcher' });
 *   const model = wrapModel(new Anthropic(), clevr);
 *   await model.messages.create({ model: 'claude-...', messages: [...] }); // gated
 */
export function wrapModel(client, clevr, opts = {}) {
  const tool = opts.tool || 'llm.messages';
  const promptText = (args = {}) => {
    const msgs = Array.isArray(args.messages) ? args.messages
               : Array.isArray(args.input) ? args.input : [];
    if (msgs.length) {
      return msgs.map((m) =>
        typeof m.content === 'string' ? m.content
          : Array.isArray(m.content) ? m.content.map((c) => c.text || '').join('\n')
          : '').join('\n');
    }
    return typeof args.prompt === 'string' ? args.prompt : '';
  };
  const guardCreate = (realCreate) => (args = {}) => {
    const msgs = Array.isArray(args.messages) ? args.messages
               : Array.isArray(args.input) ? args.input : null;
    return clevr.guard(
      { tool, action_type: 'chat',
        action: String(promptText(args)).slice(0, 8000),
        conversation: msgs && msgs.length ? msgs : undefined,
        metadata: { model: args.model } },
      () => realCreate(args)
    );
  };
  return new Proxy(client, {
    get(target, prop, recv) {
      if (prop === 'messages' && target.messages?.create) {
        return new Proxy(target.messages, {
          get(m, p) { return p === 'create' ? guardCreate(m.create.bind(m)) : Reflect.get(m, p); }
        });
      }
      if (prop === 'chat' && target.chat?.completions?.create) {
        return new Proxy(target.chat, {
          get(c, p) {
            if (p !== 'completions') return Reflect.get(c, p);
            return new Proxy(c.completions, {
              get(cc, pp) { return pp === 'create' ? guardCreate(cc.create.bind(cc)) : Reflect.get(cc, pp); }
            });
          }
        });
      }
      return Reflect.get(target, prop, recv);
    }
  });
}
