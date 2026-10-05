// @clevr/sdk/mcp — Model Context Protocol tool execution guarded by Clevr.
//
// MCP exposes tools to LLMs (search, write file, shell, etc.). This wrapper
// intercepts every MCP tool call before the underlying handler runs, so the
// agent's policy gate is applied to every real-world side-effect.
//
// Drop it into an existing MCP server with one line:
//   import { Clevr } from '@clevr/sdk';
//   import { wrapMcpHandler } from '@clevr/sdk/mcp';
//
//   const clevr = new Clevr({ agent: 'mcp-gateway', mode: 'shadow' });
//   export const handleToolCall = wrapMcpHandler(originalHandler, { clevr });
//
// In shadow mode the wrapper logs what WOULD have been blocked but lets the
// call through — zero-friction adoption while you build trust in the policies.

import { Clevr, chain } from '../index.js';

/**
 * Wrap an MCP tool handler. `originalHandler({ name, arguments })` is the
 * MCP-standard tool invocation. We compute an actor chain (human caller →
 * MCP gateway agent → tool), call clevr.guard, then run the original.
 *
 * @param {function} originalHandler — async (req) => result
 * @param {object}   opts            — { clevr, mcpAgentId? }
 * @returns a guarded handler with the same (req) => result signature.
 */
export function wrapMcpHandler(originalHandler, { clevr, mcpAgentId = 'mcp-gateway' }) {
  return async function guardedHandler(req) {
    const toolName = req.name;
    const args = req.arguments || {};

    // The human (if known) typically comes from MCP's session metadata —
    // most servers expose it via req.session.user. Fall back to anonymous.
    const human = req.session?.user || req.user || null;
    const actorChain = human
      ? chain.fromHuman(human, mcpAgentId)
      : [{ type: 'agent', id: mcpAgentId }];

    // A one-shot Clevr scoped to this MCP call, carrying the actor chain.
    const scoped = new Clevr({
      base: clevr.base, apiKey: clevr.apiKey, fetch: clevr.fetch,
      agent: mcpAgentId, mode: clevr.mode || 'enforce',
      // The platform the server's own client names; an MCP server cannot see
      // which application called the tool, so it names none of its own.
      runtime: clevr.runtime || null,
      sessionId: req.session?.id || null,
      actorChain,
    });

    return await scoped.guard({
      tool: toolName,
      target: args.target || args.path || args.url || null,
      action_type: 'mcp_tool_call',
      action: `MCP ${toolName}(${JSON.stringify(args).slice(0, 200)})`,
      metadata: { mcp_request: req.id, args },
    }, async (verdict) => {
      // In shadow mode verdict.effect may be 'block' but we still run — the
      // caller can log what would have happened.
      if (scoped.mode === 'shadow' && verdict.effect !== 'allow') {
        console.warn(`[clevr:shadow] ${toolName} would have been ${verdict.effect}: ${verdict.reason}`);
      }
      return await originalHandler(req);
    });
  };
}
