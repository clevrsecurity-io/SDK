// examples/github-action.mjs — Clevr guard for a CI/CD step.
//
// Coding agents in GitHub Actions / GitLab CI run with broad credentials and
// can touch production. This script wraps a single "agent step" with Clevr:
// the proposed action is evaluated BEFORE it runs, and the actor chain
// captures the originating human (the user who pushed the commit), the
// runner identity, and the coding agent.
//
// Use as a CI step:
//   - name: Guarded deploy
//     env:
//       CLEVR_API_KEY: ${{ secrets.CLEVR_API_KEY }}
//     run: node scripts/clevr-guard.mjs deploy
//
// In shadow mode the workflow keeps running and reports what WOULD have
// been blocked — perfect for the "ship to prod with zero risk" rollout.

import { Clevr, chain } from '../index.js';

const action = process.argv[2] || 'unknown';

// In GitHub Actions, these env vars are populated automatically.
const human = process.env.GITHUB_ACTOR;          // who pushed / dispatched
const repo  = process.env.GITHUB_REPOSITORY;     // owner/name
const sha   = process.env.GITHUB_SHA;            // commit being acted on
const runId = process.env.GITHUB_RUN_ID;
const agent = process.env.CLEVR_AGENT || 'ci-coding-agent';
const mode  = (process.env.CLEVR_MODE || 'shadow'); // start shadow, flip later

const clevr = new Clevr({ agent, mode });
clevr.actorChain = human
  ? chain.fromHuman(`${human}@github`, agent, human)
  : [{ type: 'agent', id: agent }];

const VERBS = {
  deploy:      { action_type: 'exec',   tool: 'ci.deploy',   target: `${repo}@prod`, environment: 'prod' },
  rollback:    { action_type: 'exec',   tool: 'ci.rollback', target: `${repo}@prod`, environment: 'prod' },
  'lint-fix':  { action_type: 'write',  tool: 'ci.commit',   target: `${repo}@${sha}` },
  scan:        { action_type: 'read',   tool: 'ci.scan',     target: `${repo}@${sha}` },
};
const verb = VERBS[action];
if (!verb) {
  console.error(`Unknown action "${action}". Known: ${Object.keys(VERBS).join(', ')}`);
  process.exit(2);
}

try {
  const r = await clevr.guard({
    ...verb,
    action: `[CI run ${runId || 'local'}] ${action} on ${repo || 'unknown-repo'}@${sha?.slice(0,7) || 'unknown'}`,
    metadata: { run_id: runId, sha, repo, mode },
  }, async (verdict) => {
    if (clevr.mode === 'shadow' && verdict.effect !== 'allow') {
      console.warn(`[clevr:shadow] would have been ${verdict.effect}: ${verdict.reason}`);
      console.warn(`[clevr:shadow] decision_id=${verdict.decision_id}`);
    } else {
      console.log(`[clevr:enforce] ${verdict.effect.toUpperCase()} — proceeding`);
    }
    // Actual CI work happens here — e.g. spawn a child process to do the deploy.
    return { ok: true, action };
  });
  console.log('CI step result:', JSON.stringify(r));
  process.exit(0);
} catch (e) {
  // In enforce mode a block throws ClevrBlockedError — fail the CI step.
  console.error(`[clevr] ${e.name}: ${e.message}`);
  if (e.verdict) console.error('decision_id:', e.verdict.decision_id);
  process.exit(1);
}
