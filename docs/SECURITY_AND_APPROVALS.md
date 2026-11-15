# Security, Workspace Scope and Agent Approval Notes

## Workspace and document authorization

Every retrieval call verifies workspace membership before query expansion, embedding or document access. A selected-document scope is an explicit allow-list; an empty selected list must never be converted into workspace-wide retrieval. Document filters also enforce workspace ownership, indexed status and current-version status.

Keep authorization fields (user ID, workspace ID, session ID and selected document IDs) in trusted server-side request/session state. Never accept an LLM-generated workspace or user identifier as authorization context.

## Web evidence

Tavily is optional and user-controlled per chat. It is invoked only in Agentic mode when the user enables web search and the route selects hybrid evidence. Web content is untrusted, bounded, normalized and cited separately. The integration does not fetch arbitrary URLs from the document corpus.

## Approval lifecycle

Endpoints are workspace scoped:

- `POST /api/v1/workspaces/{workspace_id}/agent-approvals`
- `GET /api/v1/workspaces/{workspace_id}/agent-approvals`
- `POST /api/v1/workspaces/{workspace_id}/agent-approvals/{approval_id}/decision`

Only owners/admins may list and decide. The requester cannot approve their own request. Consumption verifies the action, requester, exact canonical JSON arguments, current membership, approver role and expiry, then marks the approval consumed.

No current agent graph node invokes a mutating tool. The approval data/API/gate is infrastructure for future write-capable tools; do not claim that agent-driven document deletion or workspace changes are currently enabled.

## Secrets and logs

- Never include `.env`, API keys, JWT secrets, database passwords, user files or model caches in source ZIPs.
- Use `.env.example` only as a placeholder template.
- Do not log prompt text, document contents, authorization headers, secrets or raw provider exceptions.
- Rotate any key/password that was ever included in an uploaded archive.
