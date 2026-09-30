"""A real MCP server, used as a test fixture.

The payload contract is only worth anything if it holds against a genuine MCP
server rather than a hand-built mock. `langchain-mcp-adapters` hands us a raw
JSON Schema dict for each of these tools, and langchain-core does not validate
against a dict — so this is exactly the case that was silently unvalidated
before `validation.normalize_schema` existed.

Run as a stdio server: tools arrive prefixed `<server>_`, which is how
`MCPTool` gets tagged with its kind.
"""

import sys

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("fixture")

# Records what the server actually received, so a test can assert a rejected
# payload never got here. Written to a file the test reads back, since the
# server runs in a separate process.
RECEIPT = sys.argv[1] if len(sys.argv) > 1 else None


def _record(name: str, args: dict) -> None:
    if not RECEIPT:
        return
    import json

    with open(RECEIPT, "a") as f:
        f.write(json.dumps({"tool": name, "args": args}) + "\n")


@mcp.tool()
def search_employee(employee_id: str, include_terminated: bool = False) -> str:
    """Search Workday for an employee by their worker ID."""
    _record("search_employee", {
        "employee_id": employee_id,
        "include_terminated": include_terminated,
    })
    return f"employee:{employee_id}"


@mcp.tool()
def ping() -> str:
    """A tool that takes no arguments."""
    _record("ping", {})
    return "pong"


if __name__ == "__main__":
    mcp.run(transport="stdio")
