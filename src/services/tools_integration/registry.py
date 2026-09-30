"""Tool registry with risk metadata, role-based visibility, and schema validation."""

import json
import logging
from typing import Any, Callable

from src.services.tools_integration.decorator import ToolSpecMetadata
from src.services.tools_integration.validation import validate_args
from src.types import ToolKind

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central registry of all tools with risk, role, and approval metadata.

    Maps tool names to their :class:`ToolSpecMetadata` and the ``BaseTool``
    object that can actually be called. Provides role-based filtering,
    schema validation, and the single dispatch entry point that picks a
    transport from the tool's declared :class:`ToolKind`.
    """

    def __init__(self):
        self._specs: dict[str, ToolSpecMetadata] = {}
        self._callables: dict[str, Callable] = {}
        # The real tool object, kept alongside the spec. A `BaseTool` carries
        # `args_schema` and `kind`; a bare callable carries neither, so
        # validation and transport selection both need the object itself.
        self._tools: dict[str, Any] = {}
        # Agent URL per A2A tool, from config. Kept beside the spec rather than
        # on it because `ToolSpecMetadata` is frozen and shared with the
        # decorator's output.
        self._a2a_urls: dict[str, str] = {}

    def register_a2a(self, name: str, url: str, description: str = "") -> None:
        """Register a remote A2A agent as a callable tool.

        A2A has no introspection, so the payload contract is whatever the
        remote agent documents — there is no schema to validate against, and
        this registers it as such rather than pretending otherwise.
        """
        self._specs[name] = ToolSpecMetadata(
            name=name,
            description=description or f"A2A agent at {url}",
            kind=ToolKind.A2A,
        )
        self._a2a_urls[name] = url

    def select_for_query(
        self,
        allowed_names: list[str],
        query: str,
        max_tools: int = 5,
        model: Any = None,
    ) -> list[Any]:
        """Return the real tools a department may use for *this* query.

        The registry's half of the department filter: the allowlist resolves
        against registered tools, names that do not resolve are dropped, and
        the survivors are narrowed to *max_tools* by query relevance.

        ``subagents.select_department_tools`` calls this and remains the
        single place a department's tools are decided — it owns the 20-tool
        visibility cap and the ``prefix*`` namespace matching, which are about
        how a department declares its tools rather than what the registry
        knows. Keeping the split means one rule is not enforced in two places.

        Returns ``BaseTool`` objects, so the caller can bind them directly.
        """
        from src.services.tools_integration.relevance import sort_tools

        resolved: list[Any] = []
        seen: set[str] = set()
        for name in allowed_names:
            tool = self._tools.get(name)
            if tool is None:
                logger.warning("Allowlisted tool '%s' is not registered; skipping.", name)
                continue
            if name in seen:
                continue
            seen.add(name)
            resolved.append(tool)

        if len(resolved) <= max_tools:
            return resolved

        defs = [
            {
                "name": t.name,
                "description": getattr(t, "description", "") or "",
                "args_schema": getattr(t, "args_schema", None),
            }
            for t in resolved
        ]
        try:
            if model is None:
                from src.services.agent_orchestrator.agent_factory import get_model

                model = get_model()
            chosen = sort_tools(model, query, defs, max_tools)
        except Exception as e:
            logger.warning("Tool selection failed (%s); using the full allowlist.", e)
            return resolved

        by_name = {t.name: t for t in resolved}
        # The cap is enforced here, not just inside `sort_tools`: this is the
        # function's contract, and trusting the sorter's length would let a
        # bad reply hand the department its whole allowlist.
        shortlist = [
            by_name[c["name"]] for c in chosen if c.get("name") in by_name
        ][:max_tools]
        # Two failures, two fallbacks, both deliberate. A sorter that answered
        # but named nothing usable means it had no better idea, so fall back
        # to a bounded slice rather than the whole set. A sorter that could
        # not be reached at all is a different problem: the allowlist is all
        # the department has, and silently shrinking it would look like a
        # correct relevance decision that never happened.
        return shortlist or resolved[:max_tools]

    def register(self, spec: ToolSpecMetadata, callable_: Any | None = None) -> None:
        """Register a tool spec and optionally the tool object itself.

        ``callable_`` is normally a ``BaseTool``: it carries the ``args_schema``
        that validates a payload and the ``kind`` that selects a transport,
        neither of which survives being replaced by the plain function behind
        it. A bare function is still accepted and stored, but a tool
        registered that way cannot be schema-checked.
        """
        self._specs[spec.name] = spec
        if callable_ is not None:
            self._callables[spec.name] = callable_
            # Stored unconditionally. `args_schema` and `kind` are read with
            # getattr, so a bare function is usable — it just has no schema to
            # validate against and takes its declared kind. Gating on
            # `hasattr(args_schema)` would make such a tool unselectable,
            # which is worse than selecting something that validates nothing.
            self._tools[spec.name] = callable_

    def get_spec(self, tool_name: str) -> ToolSpecMetadata | None:
        """Return the spec for a tool, or None if unregistered."""
        return self._specs.get(tool_name)

    def get_tool(self, tool_name: str) -> Any | None:
        """Return the registered object for a tool, or None if unregistered.

        Usually a ``BaseTool``, which is what carries the ``args_schema`` that
        validates a payload and the ``kind`` that selects a transport. May be a
        bare function, in which case there is no schema and no declared kind.
        """
        return self._tools.get(tool_name)

    def register_builtin(self, name: str, callable_: Any, risk_level: str = "low",
                         requires_approval: bool = False, allowed_roles: tuple = (),
                         kind: ToolKind = ToolKind.LOCAL) -> None:
        """Register a built-in tool with metadata."""
        self.register(
            ToolSpecMetadata(
                name=name,
                description=getattr(callable_, "description", None)
                or getattr(callable_, "__doc__", "") or "",
                risk_level=risk_level,
                requires_approval=requires_approval,
                allowed_roles=allowed_roles,
                kind=kind,
            ),
            callable_,
        )

    # `get_visible_tools(subagent_name)` used to sit here: a sub-agent's tools
    # resolved by role, capped at 20, with a `general-purpose` special case
    # that saw everything. It has no production caller. A sub-agent's tools are
    # decided by its SKILL.md `allowed-tools` allowlist in
    # `subagents.select_department_tools`, which returns real callable
    # `BaseTool` objects rather than this registry's metadata — the two
    # disagreed, and the allowlist is the one the architecture specifies.
    # The 20-tool visibility cap still applies, there, in
    # `select_department_tools`.

    def _validate_args(self, tool_name: str, args: dict) -> dict:
        """Validate *args* against the tool's declared schema.

        Returns the validated arguments, with omitted optionals pruned. Raises
        ``ValueError`` naming the offending field if the payload does not match
        the spec.

        A tool registered as a bare function declares no schema, so there is
        nothing to check it against and its own signature is the contract.
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            return args
        return validate_args(getattr(tool, "args_schema", None), args)

    def kind_of(self, tool_name: str) -> ToolKind:
        """Return the transport for a tool, defaulting to a local call.

        The tool object is consulted first: an MCP tool is constructed with
        ``kind="mcp"`` at load time, which is authoritative. The spec's
        ``kind`` is the fallback for a tool that declared one without being
        constructed with the attribute.
        """
        tool = self._tools.get(tool_name)
        raw = getattr(tool, "kind", None) if tool is not None else None
        if raw is None:
            spec = self._specs.get(tool_name)
            raw = spec.kind if spec is not None else ToolKind.LOCAL
        try:
            return ToolKind(raw)
        except ValueError:
            logger.warning("Tool '%s' declares unknown kind %r; treating as local.",
                           tool_name, raw)
            return ToolKind.LOCAL

    def dispatch(
        self,
        tool_name: str,
        args: dict,
        subagent_name: str = "",
    ) -> Any:
        """Validate a payload and route the call to the tool's transport.

        The single execution entry point. The transport is decided by the
        tool's declared :class:`ToolKind`, not by the call site, and every kind
        validates against the same declared schema first — so a bad payload is
        refused identically whichever way the tool is reached.

        Raises:
            ValueError: If the tool is unknown, its payload does not match its
                schema, or it has no callable registered.
            RequiresApprovalError: If the tool requires approval.
            PermissionError: If the sub-agent lacks authorization.
        """
        spec = self._specs.get(tool_name)
        if spec is None:
            raise ValueError(f"Tool '{tool_name}' is not registered")

        if spec.requires_approval:
            raise RequiresApprovalError(
                f"Tool '{tool_name}' requires approval (risk={spec.risk_level})"
            )

        if spec.allowed_roles and subagent_name:
            role = subagent_name.split("_")[0] if "_" in subagent_name else subagent_name
            if role not in spec.allowed_roles and "*" not in spec.allowed_roles:
                raise PermissionError(
                    f"Sub-agent '{subagent_name}' not authorized for '{tool_name}'"
                )

        # Validate before choosing a transport: a malformed payload is the
        # same mistake whichever protocol would have carried it.
        validated = self._validate_args(tool_name, args or {})

        kind = self.kind_of(tool_name)
        logger.info("Dispatching tool '%s' (kind=%s, risk=%s)", tool_name, kind,
                    spec.risk_level)

        if kind == ToolKind.A2A:
            return self._dispatch_a2a(tool_name, validated)

        # LOCAL and MCP both end at the tool's own call: an MCP tool is
        # already wrapped as a sync StructuredTool at load time, so the
        # transport is already inside the object being called.
        tool = self._tools.get(tool_name)
        if tool is not None and hasattr(tool, "invoke"):
            return tool.invoke(validated)
        if tool_name in self._callables:
            return self._callables[tool_name](**validated)
        raise ValueError(f"Tool '{tool_name}' has no registered callable")

    def _dispatch_a2a(self, tool_name: str, args: dict) -> Any:
        """Call a remote A2A agent with the payload serialized as JSON.

        A2A carries free text, not a structured payload, so the validated
        arguments are serialized rather than splatted — that keeps the
        receiver's contract stable regardless of which transport carried them,
        and the remote agent parses the same JSON a local call would have
        received.
        """
        from src.services.tools_integration.a2a_client import call_a2a_agent

        spec = self._specs[tool_name]
        url = getattr(spec, "url", None) or self._a2a_urls.get(tool_name)
        if not url:
            raise ValueError(f"A2A tool '{tool_name}' has no configured agent URL")

        message = json.dumps(args, sort_keys=True, default=str)
        return call_a2a_agent(url, message)

    def list_tools(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._specs.keys())


class RequiresApprovalError(Exception):
    """Raised when a high-risk tool requires approval before execution."""
    pass
