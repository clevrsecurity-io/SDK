// mint-on-permit.mjs — reference agent flow for credential brokering.
//
// The agent holds NO long-lived secret. Before each consequential action it
// asks the Clevr broker for a credential. The broker runs the decision engine
// and only mints a short-lived, scoped credential when the verdict is `allow`.
// A blocked action returns NO credential — so even an agent that ignores the
// verdict has nothing to act with. Enforcement is physical, not advisory.
//
//   node sdk/examples/mint-on-permit.mjs
//
// Env: CLEVR_URL (default http://localhost:8081)

const CLEVR = process.env.CLEVR_URL || 'http://localhost:8081'

async function getCredential (action) {
  const r = await fetch(`${CLEVR}/brain/api/broker/token`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(action)
  })
  return { status: r.status, body: await r.json() }
}

// What the agent would do AFTER getting a credential: call the real target
// service with the short-lived token in the Authorization header. (Here we just
// show that the agent only proceeds when it actually holds a credential.)
async function callTarget (action, credential) {
  console.log(`  → executing ${action.action} on ${action.target} with ephemeral token (scope=${credential.scope}, expires in ${credential.ttl_seconds}s)`)
  // await fetch(targetUrl, { headers: { Authorization: `Bearer ${credential.token}` } })
}

async function act (action) {
  const { status, body } = await getCredential(action)
  if (!body.granted) {
    console.log(`✗ ${action.action} → DENIED (${body.effect}). No credential. decision=${body.decision_id}`)
    return
  }
  console.log(`✓ ${action.action} → PERMITTED. decision=${body.decision_id}`)
  await callTarget(action, body.credential)
}

const agent_id = 'demo-agent'
await act({ agent_id, action: 'dataset_query', tool: 'dataset_query', target: 'staging-logs' })   // allowed → token
await act({ agent_id, action: 'volumeDelete', tool: 'railway_cli', target: 'prod-db-volume' })     // blocked → no key
