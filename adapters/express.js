// @clevr/sdk/express — gate any Express / Connect-style HTTP handler through Clevr.
//
// Drops a decision in front of a route: the middleware calls /v1/evaluate BEFORE
// your handler runs. On `allow` it calls next(); on `block` it responds 403 and
// your handler never runs; on `escalate` it responds 403 by default (a human must
// approve) or calls next() if you set onEscalate:'allow'. The verdict is attached
// to `req.clevr` for downstream use. Works with Express, Connect, and any
// Connect-style stack (e.g. next-connect); for raw Node http, pass (req, res, next).
//
// Usage (Express):
//   import express from 'express'
//   import { Clevr } from '@clevr/sdk'
//   import { clevrGate } from '@clevr/sdk/express'
//
//   const app = express()
//   app.use(express.json())                    // so req.body is parsed BEFORE the gate
//   const clevr = new Clevr({ agent: 'billing-api' })
//
//   // Gate a whole router:
//   app.use('/refunds', clevrGate(clevr))
//
//   // Or one route, with an explicit action mapping:
//   app.post('/refunds/:id',
//     clevrGate(clevr, {
//       actionType: 'write',
//       toAction: (req) => ({ tool: 'payment.refund', target: `order/${req.params.id}` }),
//     }),
//     refundHandler)
//
// Options:
//   toAction(req)  → partial action merged over the defaults (tool/target/action/metadata)
//   actionType     → force the action_type; otherwise GET/HEAD → 'read', else 'write'
//   onEscalate     → 'block' (403, default) | 'allow' (call next())
//   failClosed     → on an engine error: true = 503, false = fail-open/next().
//                    Default follows the client's intent: fail closed when the
//                    Clevr client mode is 'enforce', fail open when 'shadow'.
//                    Set it explicitly to override.
//
// This is the GUARD path for an HTTP surface. It calls clevr.evaluate() (not
// guard()) because next() is the continuation — there is no inner function to
// wrap. It honors the client's shadow mode: in shadow the verdict is recorded
// but the request always proceeds.

export function clevrGate (clevr, opts = {}) {
  const {
    toAction,
    actionType,
    onEscalate = 'block',
  } = opts
  // Fail policy on an engine error. Default follows the client's enforcement
  // intent: an `enforce` client asked for governance, so a brain outage refuses
  // (503) rather than silently serving the route ungoverned; a `shadow` client
  // is not enforcing anyway, so it proceeds. This aligns the HTTP gate with every
  // other SDK surface (guard(), the model-loop and tool adapters all fail closed
  // on an engine error) and with the platform's fail-closed floor. An explicit
  // failClosed always wins, so availability-first teams keep opting out.
  const failClosed = opts.failClosed ?? (clevr.mode === 'enforce')

  const verbFor = (m) => (m === 'GET' || m === 'HEAD' || m === 'OPTIONS' ? 'read' : 'write')

  return async function clevrMiddleware (req, res, next) {
    const routePath = req.route?.path || req.baseUrl || req.path || req.url || ''
    const url = req.originalUrl || req.url || routePath

    // Default mapping of an HTTP request to a Clevr action. A caller-supplied
    // toAction() overrides any field. Body + query travel under metadata.input
    // so the engine's Layer-4 parameter rules (target.<field>) can read them —
    // the SDK promotes metadata.input to target_attr automatically.
    let action = {
      tool: `${req.method} ${routePath}`.trim(),
      action_type: actionType || verbFor(req.method),
      target: url,
      action: `${req.method} ${url}`,
      metadata: { input: { ...(req.query || {}), ...(req.body || {}) } },
    }
    if (typeof toAction === 'function') {
      const extra = toAction(req) || {}
      action = {
        ...action,
        ...extra,
        metadata: { ...action.metadata, ...(extra.metadata || {}) },
      }
    }

    let verdict
    try {
      verdict = await clevr.evaluate(action)
    } catch (err) {
      // Engine unreachable or errored. Do not take the app down because
      // governance is momentarily unavailable: fail open by default, or fail
      // closed (503) when the caller wants a hard dependency.
      const msg = String(err?.message || err)
      if (failClosed) {
        return res.status(503).json({ error: 'governance_unavailable', message: msg })
      }
      req.clevr = { error: msg }
      return next()
    }

    req.clevr = verdict

    // Shadow mode records the verdict server-side but never enforces.
    if (clevr.mode === 'shadow') return next()

    if (verdict.effect === 'allow') return next()
    if (verdict.effect === 'escalate' && onEscalate === 'allow') return next()

    // block, or escalate under the default policy → refuse the request.
    return res.status(403).json({
      error: verdict.effect === 'escalate' ? 'step_up_required' : 'blocked_by_policy',
      effect: verdict.effect,
      reason: verdict.reason || null,
      decision_id: verdict.decision_id || null,
    })
  }
}

export default clevrGate
