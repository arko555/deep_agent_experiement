"""Shared type definitions for the three-service architecture."""

from enum import StrEnum


class ToolKind(StrEnum):
    """How a tool is reached, which decides its payload contract.

    One discriminator, read off the tool itself, so a call site never has to
    infer the transport. Each kind differs in what a correct payload even
    looks like:

    ``LOCAL``
        A Python callable in this process. Validated against the tool's
        ``args_schema``.
    ``MCP``
        A tool on a configured MCP server, reached over the MCP transport.
        MCP declares its parameters as a raw JSON Schema dict, which
        langchain-core does not validate, so the schema is normalized to a
        model first — see ``tools_integration.validation``.
    ``A2A``
        A remote agent reached over the A2A protocol. A2A carries free text
        rather than a structured payload, so validated arguments are
        serialized into the message body as JSON.

    ``http`` is deliberately absent: no such tool type exists, and adding one
    is an SSRF surface that needs scheme checks and private-IP rejection
    before it can be a member here.
    """

    LOCAL = "local"
    MCP = "mcp"
    A2A = "a2a"

