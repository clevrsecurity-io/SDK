// examples/mcp-wrapper.mjs — runnable demo of the MCP adapter.
//
// The real adapter now lives at adapters/mcp.js (exported as `@clevr/sdk/mcp`).
// This file re-exports it for backward compatibility and runs a small
// simulation when executed directly.
//
//   import { wrapMcpHandler } from '@clevr/sdk/mcp';
//   const guarded = wrapMcpHandler(myHandler, { clevr });

import { Clevr } from '../index.js';
import { wrapMcpHandler } from '../adapters/mcp.js';

export { wrapMcpHandler };

// Demo when run directly: simulate 2 MCP calls.
if (import.meta.url === `file://${process.argv[1]}`) {
  const clevr = new Clevr({ agent: 'mcp-gateway', mode: 'shadow' });
  const handler = wrapMcpHandler(
    async (req) => ({ result: `did ${req.name}` }),
    { clevr },
  );

  console.log('\n--- Shadow mode: kb.search (safe) ---');
  console.log(await handler({
    id: 'req_1', name: 'kb.search', arguments: { query: 'churn' },
    session: { id: 'sess_mcp_1', user: 'alice@acme.com' },
  }));

  console.log('\n--- Shadow mode: fs.delete (destructive — would be blocked) ---');
  console.log(await handler({
    id: 'req_2', name: 'fs.delete', arguments: { path: '/etc/passwd' },
    session: { id: 'sess_mcp_1', user: 'alice@acme.com' },
  }));
}
