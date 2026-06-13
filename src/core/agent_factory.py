import os
from typing import Literal
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

load_dotenv()

def get_deep_agent():
    """Create and return a generic Deep Agent orchestrator.

    This agent is designed to be purely generic, relying on dynamically loaded
    skills from the './skills/' directory to handle specialized tasks.
    """
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if anthropic_key:
        model = ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620"),
            temperature=0,
        )
    elif openai_key:
        model = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            temperature=0,
        )
    else:
        model = ChatOllama(
          model="gemma4:12b-mlx",
          temperature=0.0,
    )

    workspace_root = os.getenv("WORKSPACE_ROOT", "./workspace")
    if not os.path.exists(workspace_root):
        os.makedirs(workspace_root)

    skills_root = "./skills"
    if not os.path.exists(skills_root):
        os.makedirs(skills_root)

    return create_deep_agent(
        model=model,
        system_prompt="""You are a generic Deep Agent, an expert orchestrator designed to perform any task.
Your primary goal is to use the provided skill library to handle specialized requirements on-demand.

1. **Strategic Planning**: Use `write_todos` to map out your approach for complex requests.
2. **On-Demand Skills**: You have access to a library of skills in the `skills/` directory.
   - You ONLY see names and descriptions of skills in your system prompt initially.
   - For any specialized task (e.g. research, writing, coding), you MUST look for matching skills and use `read_file` to load the `SKILL.md` before executing.
3. **Generic Subagents**: Use the `task` tool with the 'general-purpose' subagent to handle independent, complex, or context-heavy sub-tasks.
   - The 'general-purpose' subagent is also generic and can load the same skills.
4. **NO /tmp/ FOLDER**: NEVER save files to the `/tmp/` directory. This is a critical requirement.
5. **STRICT Workspace Usage**: ALL file outputs, intermediate notes, and final reports MUST be written to the `./workspace/` directory exclusively. Use the `write_file` and `edit_file` tools to manage files within this directory.
6. **Isolated Context**: Use subagents to keep the main conversation thread clean and focused on high-level orchestration.
7. **Shared Data**: Refer to `AGENTS.md` for project conventions and mission statements.

Follow the instructions in the loaded SKILL.md exactly once they are retrieved. If no skill exists for a task, proceed using your general knowledge and reasoning.""",
        tools=[internet_search],
        skills=[skills_root],
        memory=["./AGENTS.md"],
        backend=FilesystemBackend(root_dir="."),
        name="generic_deep_agent",
    )
