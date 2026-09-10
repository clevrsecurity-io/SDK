// @clevr/sdk/anthropic — adapter for the Anthropic Messages API tool_use loop.
//
// Wraps Anthropic's client.messages.create({ tools, ... }) so that EVERY
// tool_use block returned by the model gets routed through /v1/evaluate
// before the tool implementation actually runs. The adapter handles the
// full agent loop: ask the model → get tool_use → guard → execute → feed
// tool_result back → repeat until the model returns a final answer.
//
// Usage:
//   import Anthropic from '@anthropic-ai/sdk';
//   import { Clevr } from '@clevr/sdk';
//   import { withClevr } from '@clevr/sdk/anthropic';
//
//   const anthropic = new Anthropic();
//   const clevr = new Clevr({ agent: 'researcher' });
//   const guarded = withClevr(anthropic, clevr);
//
//   const result = await guarded.runAgent({
//     model: 'claude-sonnet-4-6',
//     system: 'You are a research assistant.',
//     messages: [{ role: 'user', content: 'Find churn risks in Q1.' }],
//     tools: [
//       { name: 'kb_search', description: '...', input_schema: { ... } },
//     ],
//     toolImplementations: {
//       kb_search: async ({ query }) => await myKb.search(query),
//     },
//     // Optional: classify each tool by action shape for the engine
//     toolMeta: {
//       kb_search: { action_type: 'read', target: ({ query }) => `kb/${query}` },
//     },
//   });

export function withClevr(anthropic, clevr) {
  return {
    /**
     * Run a full Anthropic agent loop with Clevr policy gating on every tool call.
     * Returns { final, messages, evaluations }.
     */
    async runAgent({
      model, system, messages, tools, toolImplementations, toolMeta = {},
      maxIterations = 10, ...rest
    }) {
      const history = [...messages];
      const evaluations = [];

      // First-iteration session bootstrap — pull user's first message as the goal.
      if (!clevr.sessionId) {
        const firstUser = history.find((m) => m.role === 'user');
        const goalText = typeof firstUser?.content === 'string'
          ? firstUser.content : JSON.stringify(firstUser?.content).slice(0, 200);
        clevr.startSession({ goal: goalText });
      }

      for (let i = 0; i < maxIterations; i++) {
        const resp = await anthropic.messages.create({
          model, system, messages: history, tools, ...rest,
        });

        if (resp.stop_reason === 'tool_use') {
          const toolUses = resp.content.filter((b) => b.type === 'tool_use');
          const toolResults = [];

          // Forward the running conversation (system prompt + prior turns + the
          // assistant turn that requested this tool) so the engine's content
          // floor can scan the PROMPT itself — PII, secrets, prompt-injection —
          // not just the tool name and arguments. The engine excludes
          // tool_result blocks from its content scan, so passing the full
          // history is verdict-safe. Without this, GUARD mode would only see
          // the action; with it, prompt-level detection works out of the box.
          const conversation = [
            ...(system ? [{ role: 'system', content: system }] : []),
            ...history,
            { role: 'assistant', content: resp.content },
          ];

          for (const tu of toolUses) {
            const meta = toolMeta[tu.name] || {};
            const target = typeof meta.target === 'function' ? meta.target(tu.input) : meta.target;
            const action = {
              tool: tu.name,
              action_type: meta.action_type || 'tool_call',
              action: `${tu.name}(${JSON.stringify(tu.input).slice(0, 200)})`,
              target: target || null,
              environment: meta.environment || null,
              conversation,
              metadata: { tool_use_id: tu.id, input: tu.input },
            };

            let toolResult;
            let isError = false;
            const fn = toolImplementations[tu.name];
            if (!fn) {
              // Still evaluate for the audit trail; nothing to run.
              try { await clevr.evaluate(action); } catch {}
              toolResult = `[No implementation for tool ${tu.name}]`;
              isError = true;
            } else {
              try {
                // ONE guard() per tool: evaluates exactly once (no double-seal),
                // respects shadow mode (always runs, records the would-be
                // verdict), throws on block, and honors onEscalate (wait polls
                // the SAME decision). Previously the adapter evaluate()'d then
                // re-guard()'d on escalate — a second decision + re-seal — and
                // ignored shadow mode entirely (suppressing the tool).
                const out = await clevr.guard(action, async () => fn(tu.input));
                const r = clevr.mode === 'shadow' ? out?.result : out;
                toolResult = typeof r === 'string' ? r : JSON.stringify(r);
              } catch (e) {
                // Governance refusal (block / escalate timeout / rejection) or a
                // tool error — surfaced to the model as an error tool_result.
                toolResult = `[Clevr: ${e.message}]`;
                isError = true;
              }
            }
            evaluations.push({ tool_use_id: tu.id, verdict: clevr.lastVerdict });
            toolResults.push({
              type: 'tool_result', tool_use_id: tu.id, content: toolResult, is_error: isError,
            });
          }

          history.push({ role: 'assistant', content: resp.content });
          history.push({ role: 'user', content: toolResults });
          continue; // ask the model again with the tool results
        }

        // No more tool_use — final answer.
        history.push({ role: 'assistant', content: resp.content });
        return { final: resp, messages: history, evaluations };
      }
      throw new Error(`Agent loop did not converge after ${maxIterations} iterations.`);
    },
  };
}
