import os
from typing import Dict, List, Optional


# --- Context and Skills Loading Helpers ---

def get_workspace_files() -> List[str]:
    workspace_dir = "./workspace"
    if not os.path.exists(workspace_dir):
        return []
    files = []
    for root, dirs, filenames in os.walk(workspace_dir):
        for f in filenames:
            rel_path = os.path.relpath(os.path.join(root, f), workspace_dir)
            files.append(rel_path)
    return sorted(files)

def get_skill_info(skill_path: str) -> Optional[dict]:
    skill_md = os.path.join(skill_path, "SKILL.md")
    if os.path.exists(skill_md):
        try:
            with open(skill_md, "r") as f:
                content = f.read()
                if content.startswith("---"):
                    parts = content.split("---")
                    if len(parts) >= 3:
                        header = parts[1]
                        info = {}
                        for line in header.split("\n"):
                            if ":" in line:
                                k, v = line.split(":", 1)
                                info[k.strip()] = v.strip()
                        return info
        except Exception:
            pass
    return None

def get_skills_summary() -> str:
    skills_dir = "./skills"
    summary = []
    if os.path.exists(skills_dir):
        for skill_name in os.listdir(skills_dir):
            skill_path = os.path.join(skills_dir, skill_name)
            if os.path.isdir(skill_path):
                info = get_skill_info(skill_path)
                if info:
                    summary.append(f"- **{info.get('name', skill_name)}**: {info.get('description', '').strip()}")
    if not summary:
        return "No specialized skills available."
    return "\n".join(summary)

def get_tools_summary(tools_dict: Dict) -> str:
    """Generates a summary of available tools for the system prompt."""
    if not tools_dict:
        return "No tools available."

    summary = []
    for name, tool_obj in tools_dict.items():
        # LangChain tools have a 'description' attribute
        description = getattr(tool_obj, 'description', 'No description provided.')
        summary.append(f"- **{name}**: {description}")

    return "\n".join(summary)

def get_memory_content() -> str:
    path = "./AGENTS.md"
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return f.read()
        except Exception:
            pass
    return ""

def get_system_prompt(role: Optional[str] = None, tools_dict: Optional[Dict] = None) -> str:
    if role in ("research", "researcher"):
        from src.nodes.research import get_researcher_system_prompt
        return get_researcher_system_prompt()
    elif role in ("write", "writer"):
        from src.nodes.write import get_writer_system_prompt
        return get_writer_system_prompt()

    skills_summary = get_skills_summary()
    tools_summary = get_tools_summary(tools_dict) if tools_dict else "No tools available."

    prompt = f"""You are a generic Deep Agent, an expert orchestrator designed to perform any task.

**Conversational Guidance:**
- If the user sends a simple greeting (e.g., "hello", "hi", "good morning"), respond warmly and briefly, then ask how you can help.
- If the user asks a straightforward question that doesn't require tools, answer directly.
- For complex, multi-step tasks, follow the planning workflow below.

1. **Strategic Planning**: Use `write_todos` to map out your approach for complex requests.
2. **On-Demand Skills**: You have access to a library of skills in the `skills/` directory.
   - You ONLY see names and descriptions of skills in your system prompt initially.
   - For any specialized task (e.g. research, writing, coding), you MUST look for matching skills and use `read_file` to load the `SKILL.md` before executing.

Available Skills:
{skills_summary}

Available Tools:
{tools_summary}

3. **Generic Subagents**: Use the `task` tool with the 'general-purpose' subagent to handle independent, complex, or context-heavy sub-tasks.
   - The 'general-purpose' subagent is also generic and can load the same skills.
4. **NO /tmp/ FOLDER**: NEVER save files to the `/tmp/` directory. This is a critical requirement.
5. **STRICT Workspace Usage**: ALL file outputs, intermediate notes, and final reports MUST be written to the `./workspace/` directory exclusively. Use the `write_file` and `edit_file` tools to manage files within this directory.
6. **Isolated Context**: Use subagents to keep the main conversation thread clean and focused on high-level orchestration.
7. **Shared Data**: Refer to `AGENTS.md` for project conventions and mission statements.

Follow the instructions in the loaded SKILL.md exactly once they are retrieved."""

    agents_md = get_memory_content()
    if agents_md:
        prompt += f"\n\n=== Shared Data / AGENTS.md ===\n{agents_md}"

    return prompt
