"""Command-line entry point for the deep agent.

Runs the full application: workspace/skills bootstrap, tool-registry
discovery, the compiled LangGraph, and the session-memory checkpointer.
The Streamlit app (app.py) drives the same graph through the same entry
point, so behaviour here matches the UI.

Usage:
    uv run python main.py                    # interactive REPL
    uv run python main.py "count the words"  # one-shot query, then exit
    uv run python main.py --tools            # list the tool registry and exit
    uv run python main.py --debug "query"    # verbose per-message logging

Every user turn sends only the new message and resets the iteration budget;
history is held by the checkpointer under a per-process thread id, so a
multi-turn REPL session carries context forward.
"""

import logging
import sys
import uuid

from langchain_core.messages import AIMessage, HumanMessage

# The graph module compiles the workflow; agent_factory's wrapper adds the
# workspace and skills bootstrap that the `task` tool and the Streamlit app
# depend on. Import the wrapper so the CLI boots exactly what the UI boots.
from src.services.agent_orchestrator.agent_factory import (
    get_deep_agent as _get_deep_agent,
)
from src.services.tools_integration.tools import create_tool_registry

logger = logging.getLogger("deep_agent.cli")


def setup_logging(verbose: bool = False) -> None:
    """Log to stdout and to deep_agent.log so a run can be inspected after."""
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    try:
        handlers.append(logging.FileHandler("deep_agent.log", mode="a"))
    except OSError:
        pass
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)-30s %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    # Third-party HTTP chatter is noise for a CLI user.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def show_tools() -> None:
    """Print the tool registry: what the model can actually call."""
    registry = create_tool_registry()
    print(f"\nTool registry — {len(registry.list_tools())} tools available\n")
    for name in sorted(registry.list_tools()):
        spec = registry.get_spec(name)
        roles = spec.allowed_roles or ("all",)
        print(f"  {name:<20} risk={spec.risk_level:<7} roles={','.join(roles)}")
        if spec.description:
            first_line = spec.description.strip().splitlines()[0]
            print(f"  {'':<20} {first_line[:70]}")
    print()


def log_turn(result: dict) -> None:
    """Log every intermediate message produced during a turn."""
    for message in result.get("messages", []):
        kind = getattr(message, "type", type(message).__name__)
        text = str(getattr(message, "content", "")).replace("\n", " ")
        if kind == "tool":
            logger.info("  [tool result] %s", text[:160])
        elif kind == "ai":
            calls = getattr(message, "tool_calls", None)
            if calls:
                for call in calls:
                    logger.info(
                        "  [tool call] %s(%s)", call["name"], call.get("args", {})
                    )
            else:
                logger.info("  [ai] %s", text[:160])
        else:
            logger.info("  [%s] %s", kind, text[:160])


def run_turn(agent, config: dict, prompt: str) -> None:
    """Send one user turn and print the answer."""
    logger.info("USER >>> %s", prompt)
    result = agent.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config=config,
    )
    log_turn(result)

    final = next(
        (m for m in reversed(result.get("messages", [])) if isinstance(m, AIMessage)),
        None,
    )
    print(f"\nAgent: {final.content if final is not None else '(no response)'}")

    usage = result.get("token_usage") or {}
    if usage:
        print(f"[tokens: {usage.get('total', 0):,} total]")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    verbose = "--debug" in argv
    argv = [a for a in argv if a != "--debug"]

    setup_logging(verbose)

    if "--tools" in argv:
        show_tools()
        return 0

    # Compile the graph: this bootstraps the workspace and skills directories
    # the `task` tool reads from, so it must be the factory's wrapper.
    logger.info("Compiling deep agent graph...")
    agent = _get_deep_agent()
    config = {"configurable": {"thread_id": f"cli-{uuid.uuid4()}"}}

    show_tools()

    # One-shot mode: run a single query from argv and exit.
    if argv:
        run_turn(agent, config, " ".join(argv))
        return 0

    print('Deep Agent CLI — type "exit" to quit.')
    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit"):
            break
        run_turn(agent, config, prompt)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
