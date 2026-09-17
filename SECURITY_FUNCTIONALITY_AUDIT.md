# Security and Functionality Audit

**Project:** deep-agents-langchain-blog  
**Audit date:** 2026-09-17  
**Branch reviewed:** `feature/mcp-a2a-integrations`  
**Scope:** Current implementation, not just the pending diff  
**Disposition:** Findings and recommendations only; no application source changes

## Executive summary

The implementation has substantial orchestration and integration functionality, but several correctness and security boundaries do not meet the guarantees described in the documentation.

The highest-priority findings are:

1. The delegation tool's provider-visible name does not match its dispatcher.
2. Filesystem validation does not reliably enforce workspace containment.
3. URL fetching has no network-destination policy and its advertised limits do not bound total work.
4. Conversation compression can separate tool results from their corresponding tool calls.
5. Audit records, token reporting, integration recovery, and UI state propagation have correctness gaps.

**All 148 existing tests passed.** This does not invalidate the findings: several tests manually construct the expected internal inputs and do not exercise the boundary where the implementation diverges. Type checking reported six errors and two unrecognized configuration options.

The project should be treated as a trusted, single-user application until its containment, resource-control, and isolation guarantees are strengthened. The current evidence does not support broad production-grade governance claims.

## Methodology and limitations

- Read `PHASES.md` and `README.md` before inspecting code; used `PHASES.md` as the authoritative implementation record.
- Reviewed core orchestration, routing, state, tools, guardrails, subagents, summarization, MCP/A2A integrations, the async bridge, CLI, Streamlit UI, configuration, and relevant tests.
- Consulted installed dependency source for tool naming, HTTP timeout behavior, and MCP discovery behavior.
- Findings are based on source inspection unless a runtime check is explicitly identified.
- No security exploits, live provider calls, remote-service probes, or malicious-input reproductions were performed.
- No claim is made that credentials were exposed, that browser JavaScript execution was achieved, or that every possible vulnerability was found.
- This was not a dependency-CVE or comprehensive Git-history secret audit.
- References use repository-relative paths and line numbers from the reviewed tree; they may move after subsequent changes.

### Verification results

| Check | Observed result |
|---|---|
| `uv run pytest` | 148 passed, 1 warning, 7.65 seconds |
| `uv run mypy src` | 6 errors in 4 source files; 2 unrecognized options |
| Ruff | Not run during this audit |
| Live model / remote integration tests | Not run |
| Application source changes | None |

## Prioritization

- **High:** Broken core behavior or a significant security boundary requiring early remediation.
- **Medium:** Incorrect results, unreliable recovery, misleading accounting, or deployment-dependent exposure.
- **Hardening / limitation:** A trust assumption, accepted limitation, or concern requiring additional verification rather than a demonstrated exploit.

These are qualitative remediation priorities, not CVSS scores. Deployment assumptions affect security severity.

---

## High-priority findings

### F-01 — Delegation tool name does not match dispatch

**Category:** Functionality  
**Priority:** High  
**Confidence:** High; confirmed from application and installed dependency source

**References:** `src/core/tools.py:205`, `src/core/tools.py:411`, `src/core/agent_factory.py:240`

The tool is constructed using `task = tool(_task_impl)`. LangChain derives the tool's name from the function, producing `_task_impl`. The registry key and special delegation dispatcher instead expect `task`.

**Impact:** A model following the advertised schema and calling `_task_impl` receives a tool-not-found result instead of delegating to a research, writer, graph, or A2A subagent. This is a schema-to-dispatch mismatch, not a live-model reproduction.

**Coverage gap:** Existing graph tests manually construct calls named `task`; schema tests inspect arguments and descriptions without asserting the tool's name.

**Recommendation:** Explicitly name the tool `task`. Preserve the direct-invocation refusal and tools-node recursion/accounting checks.

**Acceptance criteria:** The bound schema name, registry key, and dispatcher agree. A scripted model derives its emitted call name from the actual bound tool and completes delegation through the graph.

### S-01 — Filesystem validation does not enforce reliable containment

**Category:** Security  
**Priority:** High  
**Confidence:** High; symlink consequences require relevant filesystem links to exist

**References:** `src/core/guardrails.py:33`, `src/core/guardrails.py:63`, `src/core/guardrails.py:93`, `src/core/tools.py:75`, `src/core/tools.py:124`

Write validation uses a string-prefix comparison rather than directory-component containment. Both read and write validation normalize strings without resolving filesystem links. Subsequent file operations follow ordinary filesystem semantics. `search_files` also opens enumerated files without applying the read validator.

**Impact:** Writes are not strictly confined to the workspace directory. Existing symlinks can direct reads or writes outside their intended boundaries. No successful escape was executed during the audit.

**Recommendation:** Establish canonical trusted roots and enforce filesystem-aware containment consistently across file tools and UI access. Account for symlinks, regular-file handling, and concurrent path changes. A separate resolved-path check followed by an unrestricted open is not sufficient to eliminate time-of-check/time-of-use races.

**Acceptance criteria:** File access remains within its authorized root; read allowlist semantics remain intact; navigation and UI access use the same policy. Validate through isolated defensive tests without reading real secrets or touching external files.

### S-02 — URL fetching has no network-destination policy

**Category:** Security  
**Priority:** High where the process can reach sensitive HTTP services  
**Confidence:** High; actual exposure depends on deployment

**Reference:** `src/core/tools.py:155`

Agent-controlled URLs are passed directly to HTTPX with automatic redirects enabled. The application does not restrict sensitive address ranges or validate redirect destinations.

**Impact:** Fetching is not limited to public research pages; it uses the host process's available network access.

**Recommendation:** Define an outbound policy for research tools, validate destinations and redirects, and use network-level egress restrictions alongside application checks. Validation must cover the destination actually connected to, not merely the original hostname string.

**Qualification:** This is not evidence of local-file loading through `file://`; the inspected HTTP transport rejects unsupported schemes. Explicitly configured trusted integrations may need a different network policy from arbitrary research URLs.

**Acceptance criteria:** Disallowed destinations cannot be reached through direct requests or redirects. Verify policy decisions using mocks or controlled fixtures, not sensitive live endpoints.

### R-01 — Fetch limits bound retained output rather than total work

**Category:** Reliability and resource control  
**Priority:** High  
**Confidence:** High

**References:** `src/core/tools.py:155`, `src/core/tools.py:160`

After reaching the nominal one-megabyte cap, the streaming loop stops retaining chunks but continues consuming the response. The HTTPX scalar timeout bounds individual network operations, not total elapsed time. Decoded chunks are produced before the application's retained-size check.

**Impact:** Large or continuously streaming responses can occupy workers longer than the advertised limits suggest. The retained-output cap also does not bound all decompression work.

**Recommendation:** Stop consumption at the limit, enforce a total deadline, and explicitly bound response processing. Handle malformed URL errors consistently.

**Acceptance criteria:** A mocked response stops being consumed at the configured limit; processing terminates at the total deadline; truncation is reported accurately.

### F-02 — Compression can produce invalid tool-message history

**Category:** Functionality  
**Priority:** High  
**Confidence:** High; provider rejection was not exercised live

**References:** `src/core/summarization.py:64`, `src/core/summarization.py:70`, `src/nodes/plan.py:38`

Compression retains a fixed-length suffix without preserving assistant tool calls and their corresponding tool results as a unit.

**Impact:** A retained `ToolMessage` can lack the preceding assistant call it answers. Providers that validate tool-message pairing can reject the compressed conversation.

A separate boundary issue occurs when `keep_count` is zero: Python's `messages[-0:]` retains the entire history.

**Recommendation:** Compress complete conversational/tool-execution groups and define small-history-limit behavior explicitly.

**Acceptance criteria:** Every retained tool result has its corresponding retained assistant call, including multi-tool responses. Small configured history limits behave as documented.

---

## Functional correctness and integration findings

### F-03 — Write audit records are lost and failures appear successful

**Priority:** Medium  
**References:** `src/core/agent_factory.py:262`, `src/core/agent_factory.py:275`, `src/core/agent_factory.py:380`, `src/core/subagents.py:193`

Each regular write rebuilds `pending_writes` from the original state, so later writes replace earlier same-turn entries. Child-write aggregation can likewise discard same-turn parent records. Error strings are treated as executed outcomes, and tool-loop subagents record writes before execution.

**Impact:** The audit trail is incomplete and can misrepresent failed operations. This affects reporting; it does not roll back actual writes.

**Recommendation:** Accumulate records once, preserve tool-call IDs, and record explicit success/failure outcomes after execution.

**Acceptance criteria:** Multiple parent writes and mixed parent/child writes retain all records; failed operations are distinguishable from successful writes.

### F-04 — UI plan updates consume the wrong state key

**Priority:** Medium  
**References:** `app.py:357`, `src/core/agent_factory.py:264`

The UI expects `todos`, while the tools node emits `current_plan`.

**Impact:** Plan updates do not reach the sidebar through the normal stream handler.

**Recommendation:** Consume `current_plan` and test UI event-to-state propagation.

### F-05 — Rejected drafts disappear from revision context

**Priority:** Medium  
**References:** `src/nodes/review.py:82`, `src/nodes/review.py:170`, `src/nodes/plan.py:41`

Reviewer rejection replaces the staged draft with critique. Because the draft was not appended to conversation history, the next orchestrator turn receives feedback without necessarily receiving the response being critiqued.

**Recommendation:** Preserve the rejected draft alongside revision feedback without duplicating final conversation history. Keep reflection guidance available as well; reflection deliberately clears `review_verdict`, so gating all feedback solely on a nonempty verdict would introduce a regression.

**Acceptance criteria:** Revision input includes both the relevant draft and critique. Reflection guidance still reaches the orchestrator.

### F-06 — Last-budget answers bypass review

**Priority:** Medium  
**Reference:** `src/core/routing.py:22`

At the iteration limit, every no-tool answer routes to the responder, not only the explicit budget-stop message.

**Impact:** A normal answer generated on the final allowed turn skips critic and plan-compliance checks.

**Recommendation:** Distinguish a budget-stop outcome from an ordinary draft using an explicit state field. Preserve the repository's prohibition against routing by message text. Alternatively, document the bypass as an intentional policy.

**Acceptance criteria:** Tests distinguish a normal final-budget draft from a budget-stop outcome and assert the chosen review policy.

### F-07 — Token accounting is incomplete and no total-token ceiling is enforced

**Priority:** Medium  
**References:** `src/nodes/review.py:71`, `src/nodes/review.py:128`, `src/nodes/review.py:163`, `src/core/agent_factory.py:335`, `src/nodes/plan.py:65`, `src/state.py:14`

Reviewer model calls do not aggregate usage. Failed or timed-out children return empty accounting, losing partial consumption. Orchestrator accounting suppresses exceptions. Usage is tracked, but no enforced total-token ceiling was identified.

**Impact:** Displayed totals undercount actual consumption, and README's token-budget claim overstates implemented controls. Per-agent iteration limits do not constitute a total-work budget.

**Recommendation:** Aggregate usage across model paths and preserve available partial accounting. Either implement a total-token budget or correct the documentation.

**Acceptance criteria:** Deterministic reviewer and child usage contributes to reported totals; unavailable remote usage is explicitly labeled rather than implied to be zero spend.

### F-08 — Tool-loop exhaustion can return an empty answer

**Priority:** Medium  
**Reference:** `src/core/subagents.py:217`

The exhaustion fallback returns the last assistant message's text even when that message only issued tool calls.

**Recommendation:** Return an explicit exhaustion outcome and preserve completed-work accounting.

**Acceptance criteria:** A tool-call-only final turn produces a nonempty, accurate exhaustion result.

### F-09 — Custom workspace configuration is inconsistent

**Priority:** Medium  
**References:** `src/core/memory.py:43`, `src/core/guardrails.py:63`, `src/core/agent_factory.py:411`, `app.py:244`

Core filesystem tools and enumeration use literal `workspace`, while graph initialization and UI operations honor `WORKSPACE_ROOT`.

**Impact:** The agent and UI can operate on different directories. Display, download, and clear operations may not correspond to files produced by the agent.

**Recommendation:** Use one workspace-root resolver across initialization, tools, containment checks, enumeration, and UI operations. This should be coordinated with S-01.

**Acceptance criteria:** A custom root consistently controls all workspace operations.

### F-10 — MCP discovery failures can disable healthy integrations persistently

**Priority:** Medium  
**Reference:** `src/core/mcp_client.py:61`

The adapter aggregates discovery across servers. One discovery exception can discard otherwise healthy servers' tools. The empty result is cached under the unchanged configuration. Client construction is outside the discovery exception handler.

**Impact:** A transient failure can leave integrations unavailable until reset or configuration change.

**Recommendation:** Isolate discovery failures per server, avoid indefinite negative caching, and handle initialization failures consistently.

**Acceptance criteria:** A failed server does not remove a healthy server's tools, and recovery is detected without a process restart.

### F-11 — A2A deadlines and completion semantics are incomplete

**Priority:** Medium  
**References:** `src/core/a2a_client.py:45`, `src/core/a2a_client.py:89`, `src/core/async_bridge.py:54`

HTTP timeouts do not bound the complete coroutine. Poll exhaustion can return existing artifact text while a task remains nonterminal. The bridge waits without its own timeout.

**Impact:** Partial work can be presented as a finished task. A parent timeout stops waiting but does not cancel underlying remote work.

**Recommendation:** Apply an overall coroutine deadline and explicitly distinguish completion, partial results, interruption, failure, and poll exhaustion. Preserve valid direct-message responses, which need not have task-state transitions.

**Acceptance criteria:** Mocked nonterminal tasks cannot silently become completed results after poll exhaustion; deadlines terminate local waiting and perform appropriate cleanup.

---

## Deployment risks and intentional trust boundaries

### D-01 — Browser sessions share workspace files

**Priority:** Deployment-dependent  
**Reference:** `app.py:236`

Session UUIDs isolate checkpoints but not filesystem contents. Browsing, downloads, and clearing operate on one shared workspace.

**Impact:** Mutually untrusted users would not have file confidentiality or ownership isolation. This can be acceptable for a trusted single-user local application.

**Recommendation:** Document the single-user assumption or implement authenticated workspace ownership before multi-user deployment.

### D-02 — Unescaped HTML interpolation

**Priority:** Hardening  
**References:** `app.py:196`, `app.py:212`

Skill metadata and audit display fields are inserted into `unsafe_allow_html=True` blocks.

**Qualification:** Current audit display producers use fixed or empty values. A remote-output-to-HTML path and arbitrary JavaScript execution were not demonstrated.

**Recommendation:** Escape interpolated fields or use native Streamlit rendering. Do not describe this as confirmed remote XSS without further evidence.

### D-03 — Plugins and integrations are trusted code/configuration

**Priority:** Trust-model clarification  
**References:** `src/core/tools.py:363`, `src/core/mcp_client.py:32`, `src/core/subagents.py:92`

Dynamic tools execute Python during discovery. MCP tools are not constrained by built-in filesystem guards. A2A task descriptions are sent externally by design. Skills and shared conventions influence system prompts.

These are intentional capabilities, not evidence of unauthorized execution.

**Concrete correctness concerns:**

- Dynamic registry keys use Python attribute names, which may differ from provider-visible tool names.
- Configured A2A names can silently replace built-in subagent names.

**Recommendation:** Validate name consistency and reserved-name collisions. Document trusted ownership of plugins, skills, shared conventions, and integration configuration.

### D-04 — Parent timeout does not cancel worker execution

**Priority:** Accepted limitation with additional scheduling concerns  
**References:** `src/core/agent_factory.py:355`, `src/core/agent_factory.py:358`

Running workers survive parent timeouts. `PHASES.md` already accepts this limitation; it is not a newly discovered flaw.

However, sequential tasks are submitted even after the shared deadline expires. They can overlap previously timed-out work. Later pools can start while older workers remain active.

**Recommendation:** Stop submitting new work after deadline expiry; propagate cooperative cancellation where practical. Do not describe a per-pool worker count as a process-wide concurrency ceiling.

### D-05 — Sensitive arguments and error details are retained

**Priority:** Deployment-dependent hardening  
**References:** `src/core/agent_factory.py:251`, `src/core/agent_factory.py:268`, `src/core/tools.py:166`

Audit records contain tool arguments, potentially including complete file contents. Errors can include paths and full URLs.

**Qualification:** No actual credential leak was demonstrated.

**Recommendation:** Define retention and access policies. Apply redaction where the deployment's data sensitivity requires it rather than assuming all local debugging information is safe to expose broadly.

---

## Additional observations

| Observation | Reference | Recommendation / qualification |
|---|---|---|
| Messages and audit history grow indefinitely; compression does not compact checkpoints and reviewers still consume full history. | `src/state.py:6`, `src/state.py:13`, `src/nodes/review.py:67` | Define retention and context budgets across all model paths. |
| CLI invocation errors escape after retries are exhausted. | `main.py:32` | Catch expected invocation failures and preserve the REPL with an accurate error message. |
| UI opens download files outside its preceding read exception handler. | `app.py:251` | Handle missing/unreadable files consistently. |
| Numeric configuration accepts invalid ranges; float timeouts can be nonfinite. | `src/core/config.py:45`–`71` | Validate ranges and finiteness. Depth zero may intentionally disable delegation; preserve meaningful zero semantics. |
| Name validation uses `match` with an end anchor rather than whole-string matching. | `src/core/config.py:19` | Use full-string validation for the documented name alphabet. |
| Model-only callbacks do not establish graph/tool observability. Subsequent tool binding may drop callback configuration. | `src/core/agent_factory.py:140`, `src/nodes/plan.py:56` | Test actual tool-bound model paths and graph/tool events. Callback loss remains a source-backed concern, not a runtime-confirmed result. |
| Skill frontmatter parsing does not implement folded YAML descriptions. | `src/core/memory.py:55`, `app.py:145` | Parse the supported format correctly or constrain/document it. |
| UI status fields are not consistently updated; errors can disappear after unconditional rerun. | `app.py:222`, `app.py:377`–`387` | Add event-handler tests and persistent error/status state. |
| Sample tool uses `pytz` without a direct project dependency. | `tools/sample_tool.py:3`–`16` | Declare the dependency or use a suitable existing standard-library facility. |

## Type-check baseline

Mypy reported these six errors:

| Location | Reported issue |
|---|---|
| `src/core/rag.py:23` | Tavily import has no stubs or `py.typed` marker. |
| `src/core/utils.py:41` | Returning `Any` from a function declared to return `str`. |
| `src/core/subagents.py:181` | Missing annotation for `write_ops`. |
| `src/core/agent_factory.py:238` | `BaseMessage` has no attribute `tool_calls`. |
| `src/core/agent_factory.py:251` | Inferred `Sequence[str]` has no `append`. |
| `src/core/agent_factory.py:288` | Inferred `Sequence[str]` has no `append`. |

It also rejected these configuration options:

- `check_unsemetic_casts`
- `explicit_error_codes`

These are existing baseline results, not regressions from audit changes.

## Separate working-tree observation: `tf_idf.py`

**Reference:** `tf_idf.py:163`–`192`

The working-tree version contains a module-level remote API call. Importing it requires `HF_TOKEN` and attempts a request. The committed version contains the original TF-IDF/window-attention implementation.

This file is unrelated to the active deep-agent execution path. Its existing modifications were left untouched under the report-only decision. Earlier interactive option labels and descriptions were inconsistent; no restoration authorization is inferred from that selection.

**Recommendation:** Make an explicit separate decision about keeping the demo or restoring the original implementation. If retained as a demo, avoid network calls at import time.

---

## Recommended remediation sequence

### Phase 1 — Core contracts and containment

1. Correct delegation schema/registry/dispatch naming (F-01).
2. Establish consistent filesystem containment and workspace configuration (S-01, F-09).
3. Restrict research-tool outbound destinations and bound fetch work (S-02, R-01).
4. Preserve protocol-valid message groups during compression (F-02).

**Verification:** Focused deterministic tests of each boundary, then the full test suite. Security checks should use mocks and isolated fixtures, not live sensitive destinations or real secrets.

### Phase 2 — State integrity and review correctness

1. Repair write-audit accumulation and outcome reporting (F-03).
2. Correct UI plan-state propagation (F-04).
3. Preserve rejected drafts and reflection guidance (F-05).
4. Make iteration-boundary review policy explicit (F-06).
5. Correct usage reporting and exhaustion outcomes (F-07, F-08).

**Verification:** Scripted graph tests covering multiple writes, mixed parent/child accounting, rejection/revision, reflection, final-budget answers, and loop exhaustion.

### Phase 3 — Integration resilience

1. Isolate MCP discovery failures and support recovery (F-10).
2. Correct A2A deadlines and completion states (F-11).
3. Stop scheduling new tasks after deadline expiry (D-04).
4. Verify callback coverage and name consistency across integrations.

**Verification:** Fake clients and deterministic deadline/state fixtures; preserve the persistent async bridge and valid A2A direct-message responses.

### Phase 4 — Deployment clarity and maintenance

1. Document shared-workspace and trusted-input assumptions.
2. Escape UI HTML interpolation and improve error handling.
3. Define retention/context policies and address type-check failures.
4. Add accurate implementation notes to `PHASES.md` and revise overstated README guarantees after fixes are verified.
5. Handle `tf_idf.py` only under a separate explicit decision.

## Deliberate behavior to preserve

- Tools execute writes immediately; `pending_writes` is an audit record, not an approval gate.
- Direct invocation of the exposed `task` tool remains refused; delegation executes through the tools node.
- Routing remains based on explicit state, never response-text keywords.
- Checkpointer-owned conversation history and distinct child thread IDs remain intact.
- The persistent async bridge remains; do not replace it with per-call `asyncio.run`.
- A2A import-time registration and MCP call-time configuration retain their documented semantics.

## Final assessment

The test suite establishes many intended flows, but not all advertised security and integration contracts. Prioritize boundary-focused regression tests and the concrete defects above before expanding capabilities or describing the system as production-ready for untrusted or multi-user workloads.
