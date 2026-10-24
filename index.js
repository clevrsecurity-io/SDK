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

// The policies the floor itself decides (brain lib/floor/floor.js), read when an
// engine older than `floor_enforced` answers.
const FLOOR_DECIDERS = new Set(['content-risk', 'verb-risk', 'identity', 'idp-deprovisioned', 'session-stopped', 'session-proof']);

/** The floor's own refusal, or the hold that took its place: applied in every mode. */
export function floorStands(verdict) {
  if (verdict?.floor_enforced === true) return true;
  return (verdict?.effect === 'block' && FLOOR_DECIDERS.has(verdict.matched_policy))
    || (verdict?.effect === 'escalate' && !!verdict?.floor_hold);
}

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
    // The platform the agent runs on (langgraph, crewai, n8n...), recorded on
    // every decision so the console says where an action came from. Each
    // adapter names its own framework when this is unset; set it when the
    // agent runs on something the adapter cannot see (LangGraph over the
    // LangChain adapter, say). Recorded, never judged: no verdict depends on it.
    this.runtime = opts.runtime || process.env.CLEVR_RUNTIME || null;
    this.onEscalate = opts.onEscalate || 'throw'; // 'throw' | 'wait' | 'allow'
    this.fetch = opts.fetch || globalThis.fetch;
    // How long one HTTP call to the brain may take (evaluate, card spend, the
    // poll of a held decision). Without it an unreachable brain held the
    // agent for as long as the platform's socket timeout, where the Python SDK
    // gave up at ten seconds. On the deadline the call fails like any other
    // transport error, so onEscalate / the failsafe answer as usual.
    this.timeoutMs = Number(opts.timeoutMs ?? process.env.CLEVR_TIMEOUT_MS) || 10_000;
    this.sessionId = opts.sessionId || null;
    this.sessionGoal = opts.sessionGoal || null;
    this.parentRequestId = opts.parentRequestId || null;
    this.parentAuthority = opts.parentAuthority || null;
    // mode: 'enforce' (default) actually blocks / escalates the action.
    // 'shadow' records the decision server-side and lets the action run, so
    // teams can ship the SDK, gather the audit trail and see what WOULD have
    // been blocked before flipping to enforce. Like the engine's Observe, it
    // shadows only what is discretionary: the safety floor's own refusal, or
    // the hold that took its place, is applied in both modes.
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
    this._reports = new Set();
  }

  /**
   * Start a new session on this client: one per run of the agent. Every later
   * call carries it until the next startSession. Returns the session id.
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
      agent: childAgentId, runtime: this.runtime,
      mode: this.mode, onEscalate: this.onEscalate,
      sessionId: this.sessionId, sessionGoal: this.sessionGoal,
      actorChain: newChain,
    });
  }

  /** Whether this client applies the verdict: always in enforce, only the floor's in shadow. */
  enforces(verdict) {
    return this.mode === 'enforce' || floorStands(verdict);
  }

  /**
   * Wrap a single tool execution with the policy gate.
   *
   * In `enforce` mode (default): allow runs, block throws, escalate behaves
   * per `onEscalate`.
   * In `shadow` mode: runs the inner function and returns
   * { result, verdict, shadow, wouldHaveBlocked, wouldHaveEscalated },
   * whatever a discretionary verdict says. The floor's refusal or hold is
   * applied as in enforce: a refusal throws, a hold follows `onEscalate`.
   *
   * @param {object} action  { tool, target, action, action_type, environment, target_attr?, metadata? }
   * @param {function} run   async (verdict) => result  — the inner tool execution
   */
  async guard(action, run) {
    const verdict = await this.evaluate(action);
    this.lastVerdict = verdict;
    if (this.mode !== 'shadow') return await this.#apply(verdict, run, action);
    const result = this.enforces(verdict)
      ? await this.#apply(verdict, run, action)
      : await this.runReported(verdict, run);
    return { result, verdict, shadow: true,
             wouldHaveBlocked: verdict.effect === 'block',
             wouldHaveEscalated: verdict.effect === 'escalate' };
  }

  async #apply(verdict, run, action) {
    if (verdict.effect === 'allow') {
      return await this.runReported(verdict, run);
    }
    if (verdict.effect === 'block') {
      this.reportEnforcement(verdict.decision_id, 'denied');
      throw new ClevrBlockedError(verdict);
    }
    // escalate
    if (this.onEscalate === 'wait')  return await this.#waitThenRun(verdict, run, action);
    if (this.onEscalate === 'allow') return await this.runReported(verdict, run);
    this.reportEnforcement(verdict.decision_id, 'denied');
    throw new ClevrEscalatedError(verdict);
  }

  // Report a successful tool answer as allowed, or a pre-execution refusal as denied.
  // An exception after launch leaves side effects unconfirmed; it never proves rollback.

  /**
   * Tell Clevr what this door did with a decision: 'allowed', 'denied' or
   * 'asked'. Never awaited by the caller, never throws: a report that cannot
   * be sent leaves the decision unconfirmed. `flush()` waits for the ones
   * still on their way.
   */
  reportEnforcement(decisionId, enforced) {
    if (!decisionId || !['allowed', 'asked', 'denied'].includes(enforced)) return;
    let p;
    try {
      p = Promise.resolve(this.fetch(`${this.base}/v1/decisions/${encodeURIComponent(decisionId)}/enforcement`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${this.apiKey}` },
        body: JSON.stringify({ enforced }),
        signal: AbortSignal.timeout(Math.min(this.timeoutMs, 2000)),
      }));
    } catch { return; }
    const done = p.then((r) => { try { r?.body?.cancel?.(); } catch { /* read nothing */ } }, () => {});
    this._reports.add(done);
    done.finally(() => this._reports.delete(done));
  }

  /** Wait for the reports still on their way. Call it before a short-lived process exits. */
  async flush() { await Promise.all([...this._reports]); }

  /** Run the tool on a verdict and report how it went. */
  async runReported(verdict, run) {
    // A thrown tool may already have committed side effects. Leave its outcome unknown.
    const result = await run(verdict);
    this.reportEnforcement(verdict?.decision_id, 'allowed');
    return result;
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
      // The action's own runtime, when it names one, wins (it comes after).
      ...(this.runtime ? { runtime: this.runtime } : {}),
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
      signal: AbortSignal.timeout(this.timeoutMs),
    });
    if (!r.ok) {
      const text = await r.text();
      throw new Error(`Clevr evaluate failed (${r.status}): ${text}`);
    }
    const v = await r.json();
    // Unknown protocol data must not fall through to onEscalate='allow'.
    if (!v || typeof v !== 'object' || Array.isArray(v) || !['allow', 'block', 'escalate', 'step_up'].includes(v.effect)) {
      throw new Error('Clevr evaluate returned an invalid verdict');
    }
    // Delegation context for the next hop. A session the engine opened is not
    // taken over: kept on a long-lived client, it would make every later run
    // one endless session.
    if (v.request_id) this.lastRequestId = v.request_id;
    return v;
  }

  /**
   * Ask to pay with this agent's card (POST /v1/card/spend). The payment is
   * judged like any action; on an allow Clevr opens a one-time permit, and the
   * next purchase on the card that fits it is approved when the card network
   * asks. The card number is never part of this exchange.
   *
   *   const p = await clevr.requestCardPayment({ amount: '49.00', currency: 'EUR', merchant: 'Acme', reason: 'licence' })
   *   if (p.granted) { ...pay before p.permit.expires_at... }
   *   else if (p.resume) { ...once a person approved: requestCardPayment({ ...same, resume: p.resume }) }
   *
   * Returns the brain's answer with `status` (200 granted, 202 held, 403 refused,
   * 409 allowed but no permit could be opened). Throws on transport and input errors.
   */
  async requestCardPayment({ amount, currency, merchant, reason, resume, ttlSeconds } = {}) {
    const r = await this.fetch(`${this.base}/v1/card/spend`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${this.apiKey}` },
      body: JSON.stringify({
        agent_id: this.agent, session_id: this.sessionId,
        amount, currency, merchant, reason,
        ...(resume ? { resume } : {}),
        ...(ttlSeconds ? { ttl_seconds: ttlSeconds } : {}),
      }),
      signal: AbortSignal.timeout(this.timeoutMs),
    });
    let body = {};
    try { body = await r.json(); } catch { body = {}; }
    if (![200, 202, 403, 409].includes(r.status)) {
      throw new Error(`Clevr card payment request failed (${r.status}): ${body.message || body.error || 'no answer'}`);
    }
    return { status: r.status, ...body };
  }

  /**
   * Poll the engine for resolution of an escalated decision. Backoff up to
   * `timeoutMs` (default 10 min). When resolved=approved → runs the inner fn.
   */
  async #waitThenRun(verdict, run, action, timeoutMs = 10 * 60_000) {
    return await this.runReported(await this.waitForDecision(verdict, action, timeoutMs), run);
  }

  /**
   * Wait for a person to decide a held verdict, as guard() does with onEscalate
   * 'wait'. Resolves with the verdict that spent the approval, for the caller
   * to run on; throws ClevrBlockedError on a rejection and ClevrEscalatedError
   * on a timeout or a hold the same action already waits on (held_as).
   */
  async waitForDecision(verdict, action, timeoutMs = 10 * 60_000) {
    // The same action already waits for a person (held_as): this attempt is
    // settled with that request and never runs on its own, so it does not wait.
    if (verdict.held_as) { this.reportEnforcement(verdict.decision_id, 'denied'); throw new ClevrEscalatedError(verdict); }
    const start = Date.now();
    let delay = 1000;
    let deadline = start + timeoutMs;
    while (Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, delay));
      delay = Math.min(delay * 1.5, 15_000);
      // A poll that fails (the engine restarting, a network blip, a proxy's
      // error page) is polled again, as in Python: one failed poll ended the
      // wait with the raw TypeError while the person could still approve.
      let status;
      try { status = await this.#getDecision(verdict.decision_id); } catch { continue; }
      // The wait ends when the decision ends. The workspace sets how long a held
      // action may wait before it is declined on its own, the decision carries
      // that deadline, and this reads it: a ten minute constant meant a tenant
      // allowing thirty had agents giving up at ten while the console said
      // otherwise. A minute and five seconds past it, so the sweep wins the race
      // and the agent reads the real answer instead of a timeout: the sweep runs
      // once a minute, and five seconds had both SDKs time out at 78 s on a one
      // minute window while the hold could still be approved (2026-10-14).
      const ends = status?.expires_at ? Date.parse(status.expires_at) : NaN;
      if (Number.isFinite(ends)) deadline = ends + 65_000;
      // The engine resolves a step-up by writing 'approved' / 'rejected' on
      // the decision (override route + server-side expiry sweep). We also
      // accept the raw override effects 'allow' / 'block' defensively so a
      // direct effect write still unblocks the agent.
      const res = status?.resolution;
      // Approved: the action runs on the approval, spent by asking again with
      // it, as every other door does. Run on the poll alone, it was never spent:
      // the operation read "approved, not used" for an action that had run.
      if (res === 'approved' || res === 'allow') return await this.#spend(verdict, action);
      if (res === 'rejected' || res === 'block') {
        this.reportEnforcement(verdict.decision_id, 'denied');
        throw new ClevrBlockedError({ ...verdict, reason: `Rejected by ${status.resolved_by}: ${status.resolution_reason}` });
      }
    }
    this.reportEnforcement(verdict.decision_id, 'denied');
    throw new ClevrEscalatedError({ ...verdict, reason: `Escalation timed out after ${Math.round((Date.now() - start) / 1000)}s.` });
  }

  // The approval, spent by asking again with it: the verdict the action runs on.
  async #spend(verdict, action) {
    const spent = await this.evaluate({ ...action, resume: verdict.decision_id });
    this.lastVerdict = spent;
    if (spent.effect === 'allow') return spent;
    this.reportEnforcement(spent.decision_id, 'denied');
    if (spent.effect === 'block') throw new ClevrBlockedError(spent);
    throw new ClevrEscalatedError(spent);
  }

  // One poll of a held decision. A transport error (the deadline included) is
  // one missed poll, never the end of the wait: the loop asks again.
  async #getDecision(decisionId) {
    try {
      const r = await this.fetch(`${this.base}/v1/decisions/${decisionId}`, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
        signal: AbortSignal.timeout(this.timeoutMs),
      });
      if (!r.ok) return null;
      return await r.json();
    } catch { return null; }
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

// The name a prompt is recorded under, kept under the model's door `llm.`.
// A caller's own name outside it had "Pay Acme the 5000" refused as a
// financial operation (2026-10-14).
export function modelDoor(name, fallback) {
  const n = typeof name === 'string' && name.trim() ? name.trim() : fallback;
  return /^llm\./i.test(n) ? n : `llm.${n}`;
}

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
 *
 * `opts.tool` names the prompts in the console (default `llm.messages`). It is
 * kept under the model's door: `{ tool: 'billing' }` records `llm.billing`. The
 * engine reads a chat as what a person asks the model only under `llm.`; under
 * any other name, the prompt is judged as the operation that name says.
 */
export function wrapModel(client, clevr, opts = {}) {
  const tool = modelDoor(opts.tool, 'llm.messages');
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
  // Which SDK the wrapped client is, read from the method it was called by:
  // messages.create is Anthropic's, chat.completions.create OpenAI's.
  const guardCreate = (realCreate, framework) => (args = {}) => {
    const msgs = Array.isArray(args.messages) ? args.messages
               : Array.isArray(args.input) ? args.input : null;
    return clevr.guard(
      { tool, action_type: 'chat', runtime: clevr.runtime || framework,
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
          get(m, p) { return p === 'create' ? guardCreate(m.create.bind(m), 'anthropic-sdk') : Reflect.get(m, p); }
        });
      }
      if (prop === 'chat' && target.chat?.completions?.create) {
        return new Proxy(target.chat, {
          get(c, p) {
            if (p !== 'completions') return Reflect.get(c, p);
            return new Proxy(c.completions, {
              get(cc, pp) { return pp === 'create' ? guardCreate(cc.create.bind(cc), 'openai-sdk') : Reflect.get(cc, pp); }
            });
          }
        });
      }
      return Reflect.get(target, prop, recv);
    }
  });
}
