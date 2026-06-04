# Deep Agent Demo with Dynamic Skills

This project demonstrates the **Deep Agent** architecture using LangChain, LangGraph, and the [Agent Skills](https://agentskills.io) specification. It features a purely generic orchestrator that loads specialized workflows on-demand.

## Core Architecture

1.  **Generic Orchestrator**: A universal agent that maps out complex tasks using a `write_todos` planning tool.
2.  **General-Purpose Subagents**: Uses isolated sub-contexts for execution, preventing context window saturation and ensuring focus.
3.  **Dynamic Skill Loading**: Specialized behaviors (e.g., `research`, `writer`) are defined as `SKILL.md` files in the `skills/` directory. The agents discover these skills via metadata and load full instructions only when relevant.
4.  **Strict Persistence**: All intermediate work, notes, and final reports are stored exclusively in the `./workspace` directory.

## Getting Started

### Prerequisites

- [uv](https://github.com/astral-sh/uv) package manager installed.
- Anthropic API Key (Claude 3.5 Sonnet) or OpenAI API Key.
- Tavily API Key (for web research).

### Setup

1.  Clone this repository.
2.  Create a `.env` file:
    ```bash
    ANTHROPIC_API_KEY=your_key
    TAVILY_API_KEY=your_key
    WORKSPACE_ROOT=./workspace
    ```

### Running the App

Start the Streamlit interface:
```bash
uv run streamlit run app.py
```

## How to Test

1.  **Submit a Request**: Enter a complex prompt in the chat, such as:
    > "Research the impact of multi-agent systems on software engineering and draft a 500-word blog post."
2.  **Observe Planning**: Watch the agent decompose the request in the **Current Plan** sidebar.
3.  **Monitor Skill Loading**: The agent will identify the `research` and `writer` skills, loading their `SKILL.md` files dynamically.
4.  **Check Output**: View the final artifacts and intermediate notes in the **Workspace Files** section of the sidebar. All files are written to the `./workspace/` directory.

## Skill Specification

Each skill folder in `skills/` must follow the [Agent Skills spec](https://agentskills.io/specification):
- `SKILL.md`: Contains YAML frontmatter (name, description) and instructions.
- Folder name must match the `name` field in the frontmatter.
