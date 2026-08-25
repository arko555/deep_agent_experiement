"""Command-line entry point for the deep agent.

Interactive REPL over the same compiled graph the Streamlit app uses.
Conversation history is held by the checkpointer under a per-process
thread id; each user turn sends only the new message and resets the
iteration budget.
"""

import uuid

from langchain_core.messages import AIMessage, HumanMessage

from agent import get_deep_agent


def main():
    agent = get_deep_agent()
    config = {"configurable": {"thread_id": f"cli-{uuid.uuid4()}"}}
    print("Deep Agent CLI — type 'exit' to quit.")

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

        result = agent.invoke(
            {"messages": [HumanMessage(content=prompt)], "iteration_count": 0},
            config=config,
        )
        messages = result.get("messages", [])
        final = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        print(f"Agent: {final.content if final is not None else '(no response)'}")
        usage = result.get("token_usage") or {}
        if usage:
            print(
                f"[tokens: {usage.get('total', 0):,} total | "
                f"iterations this turn: {result.get('iteration_count', 0)}]"
            )


if __name__ == "__main__":
    main()
