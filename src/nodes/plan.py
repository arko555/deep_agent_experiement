from langchain_core.messages import SystemMessage

from src.state import AgentState
from src.core.memory import get_system_prompt
from src.core.utils import invoke_with_retry


def call_orchestrator(state: AgentState, model, tools: list, max_history_messages: int = 20) -> dict:
    """
    Executes the main orchestrator agent step.

    Args:
        state: The current AgentState.
        model: The LLM instance to invoke.
        tools: The list of tools bound to the LLM.
        max_history_messages: Maximum number of conversation messages to keep.

    Returns:
        A dict updating `next_message`.
    """
    # Enforce max_iterations guard
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 10)
    if iteration_count >= max_iterations:
        from langchain_core.messages import AIMessage
        error_msg = AIMessage(content="Maximum iterations reached. Stopping to prevent runaway execution.")
        return {"next_message": error_msg}

    system_prompt = get_system_prompt()
    messages = state["messages"]
    # Truncate old messages to prevent exceeding context limits
    if len(messages) > max_history_messages:
        messages = messages[-max_history_messages:]
    formatted_messages = [SystemMessage(content=system_prompt)] + messages
    model_with_tools = model.bind_tools(tools)
    response = invoke_with_retry(model_with_tools, formatted_messages)

    # Token usage tracking from response metadata
    updates = {"next_message": response}
    try:
        usage = getattr(response, "usage_metadata", None)
        if usage:
            # LangChain usage_metadata contains input_tokens, output_tokens
            input_tokens = getattr(usage, "input_tokens", 0) or usage.get("input_tokens", 0)
            output_tokens = getattr(usage, "output_tokens", 0) or usage.get("output_tokens", 0)
            current_usage = state.get("token_usage", {})
            total_in = current_usage.get("input", 0) + input_tokens
            total_out = current_usage.get("output", 0) + output_tokens
            updates["token_usage"] = {"input": total_in, "output": total_out, "total": total_in + total_out}
    except Exception:
        pass

    return updates
