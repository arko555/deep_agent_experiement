"""A mock Workday HR system, served over MCP stdio.

The point of this server is a realistic end-to-end run: the router picks the
`hr` department, the department's SKILL.md allowlist (`workday_*`) claims these
tools, and the sub-agent's ReAct loop calls them over a genuine MCP transport.
So the tools return actual records — an echo server would prove the plumbing
but not the answer-with-exact-information part.

Run via `MCP_SERVERS` (read at call time, no restart needed):

    MCP_SERVERS='{"workday": {"transport": "stdio",
                              "command": "uv", "args": ["run", "python",
                                                        "tools/mock_workday_server.py"]}}'

Or directly, to smoke-test the server itself:

    uv run python tools/mock_workday_server.py --selftest
"""

import argparse
import json
import sys

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("workday")

# ---------------------------------------------------------------------------
# Mock data — a small in-memory HR record set
# ---------------------------------------------------------------------------

EMPLOYEES: dict[str, dict] = {
    "WD-001": {
        "name": "Priya Sharma",
        "designation": "Senior Software Engineer",
        "department": "Engineering",
        "manager": "WD-020",
        "manager_name": "Kieran O'Donnell",
        "tenure_years": 3.5,
        "leave_balance_days": 12,
        "location": "Bengaluru",
    },
    "WD-002": {
        "name": "Marcus Webb",
        "designation": "Marketing Manager",
        "department": "Marketing",
        "manager": "WD-021",
        "manager_name": "Sofia Reyes",
        "tenure_years": 1.2,
        "leave_balance_days": 5,
        "location": "Manchester",
    },
    "WD-020": {
        "name": "Kieran O'Donnell",
        "designation": "Engineering Director",
        "department": "Engineering",
        "manager": None,
        "manager_name": None,
        "tenure_years": 7.0,
        "leave_balance_days": 20,
        "location": "Dublin",
    },
    "WD-021": {
        "name": "Sofia Reyes",
        "designation": "VP Marketing",
        "department": "Marketing",
        "manager": None,
        "manager_name": None,
        "tenure_years": 5.5,
        "leave_balance_days": 15,
        "location": "Madrid",
    },
}

# Records keyed by employee id, mirroring a real HR calendar.
LEAVE_REQUESTS: dict[str, dict] = {
    "LV-1001": {
        "employee_id": "WD-001",
        "type": "annual",
        "start": "2026-10-12",
        "end": "2026-10-16",
        "days": 5,
        "status": "approved",
    },
    "LV-1002": {
        "employee_id": "WD-002",
        "type": "sick",
        "start": "2026-09-28",
        "end": "2026-09-29",
        "days": 2,
        "status": "approved",
    },
}

# ---------------------------------------------------------------------------
# Tools — the surface the `workday_*` allowlist claim picks up
# ---------------------------------------------------------------------------


@mcp.tool()
def search_employee(name: str) -> str:
    """Search the HR system for an employee by (partial) full name.

    Returns a JSON object with `matches`, each carrying the employee's
    `employee_id`, `name`, `designation`, and `department`. Use the
    `employee_id` from the best match in every other tool here.
    """
    query = name.strip().lower()
    matches = [
        {k: e[k] for k in ("employee_id", "name", "designation", "department")}
        for e in ({"employee_id": eid, **rec} for eid, rec in EMPLOYEES.items())
        if query in e["name"].lower()
    ]
    if not matches:
        return json.dumps({"matches": [], "note": "no employee record matched"})
    return json.dumps({"matches": matches})


@mcp.tool()
def get_employee_details(employee_id: str) -> str:
    """Return the full HR record for one employee by their worker ID.

    The record includes designation, department, manager, tenure, leave
    balance in days, and location. Returns an error string if the ID is
    unknown — report that rather than guessing.
    """
    rec = EMPLOYEES.get(employee_id.strip())
    if rec is None:
        return f"Error: no employee with worker ID '{employee_id}'."
    return json.dumps({"employee_id": employee_id.strip(), **rec})


@mcp.tool()
def get_leave_balance(employee_id: str) -> str:
    """Return an employee's remaining leave balance in days.

    Use after `search_employee` when the question is specifically
    about time off. The result is a point-in-time figure from the HR system.
    """
    rec = EMPLOYEES.get(employee_id.strip())
    if rec is None:
        return f"Error: no employee with worker ID '{employee_id}'."
    return json.dumps({
        "employee_id": employee_id.strip(),
        "name": rec["name"],
        "leave_balance_days": rec["leave_balance_days"],
    })


@mcp.tool()
def list_leave_requests(employee_id: str) -> str:
    """List an employee's leave requests with type, dates, days, and status."""
    eid = employee_id.strip()
    if eid not in EMPLOYEES:
        return f"Error: no employee with worker ID '{employee_id}'."
    reqs = [
        {"request_id": rid, **r}
        for rid, r in LEAVE_REQUESTS.items()
        if r["employee_id"] == eid
    ]
    return json.dumps({"employee_id": eid, "requests": reqs})


@mcp.tool()
def get_reporting_line(employee_id: str) -> str:
    """Return who an employee reports to, resolved to a name and worker ID."""
    rec = EMPLOYEES.get(employee_id.strip())
    if rec is None:
        return f"Error: no employee with worker ID '{employee_id}'."
    if rec["manager"] is None:
        return json.dumps({
            "employee_id": employee_id.strip(),
            "name": rec["name"],
            "manager": None,
            "note": "reports to no one in the system; likely an executive",
        })
    return json.dumps({
        "employee_id": employee_id.strip(),
        "name": rec["name"],
        "manager": {"employee_id": rec["manager"], "name": rec["manager_name"]},
    })


# ---------------------------------------------------------------------------
# Self-test: exercise every tool through the real MCP client layer
# ---------------------------------------------------------------------------


def _selftest() -> int:
    """Call each tool directly and print the results, then exit.

    This bypasses MCP (it imports the functions, it does not open a session),
    so it proves the tool bodies and the data, not the transport. The
    transport is proven by pointing `MCP_SERVERS` at this file and running
    the agent.
    """
    checks = [
        ("search: Priya", search_employee(name="Priya")),
        ("search: nobody", search_employee(name="Zaphod")),
        ("details: WD-001", get_employee_details(employee_id="WD-001")),
        ("details: bad id", get_employee_details(employee_id="WD-999")),
        ("leave: WD-002", get_leave_balance(employee_id="WD-002")),
        ("requests: WD-001", list_leave_requests(employee_id="WD-001")),
        ("reporting: WD-001", get_reporting_line(employee_id="WD-001")),
        ("reporting: WD-020", get_reporting_line(employee_id="WD-020")),
    ]
    for label, out in checks:
        print(f"--- {label}\n{out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selftest", action="store_true",
        help="call each tool directly, print results, and exit",
    )
    args = parser.parse_args()
    if args.selftest:
        return _selftest()
    mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
