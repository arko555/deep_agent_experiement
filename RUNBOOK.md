# Deep Agent Demo — RunBook

This document provides step-by-step instructions for setting up, running, and testing the Deep Agent Demo.

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have the following installed:
- **Python 3.12+**
- **[uv](https://github.com/astral-sh/uv)** (Recommended package manager)
- **API Keys**:
    - `ANTHROPIC_API_KEY` (Recommended) or `OPENAI_API_KEY`
    - `TAVILY_API_KEY` (Required for web search)

### 2. Installation
Clone the repository and install dependencies using `uv`:

```bash
# Clone the repository
git clone https://github.com/arko555/deep_agent_experiement.git
cd deep_agent_experiement

# Install dependencies
uv sync
```

### 3. Configuration
Create a `.env` file in the root directory to store your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your actual keys:
```env
ANTHROPIC_API_KEY=sk-ant-xxx...
TAVILY_API_KEY=tvly-xxx...
# Optional:
# OPENAI_API_KEY=sk-proj-xxx...
# WORKSPACE_ROOT=./workspace
```

### 4. Running the Application

#### Option A: Streamlit Web UI (Recommended)
The Web UI provides a visual dashboard for the agent's thinking process, audit logs, and human-in-the-loop approvals.

```bash
uv run streamlit run app.py
```
*Once running, open `http://localhost:8501` in your browser.*

#### Option B: Python CLI
For programmatic access or automated testing.

```python
from agent import get_deep_agent
from langchain_core.messages import HumanMessage

# Initialize the agent
agent = get_deep_agent()

# Define the task
task = "Research the impact of multi-agent systems on software engineering and draft a 500-word blog post."

# Execute
result = agent.invoke({
    "messages": [HumanMessage(content=task)],
    "current_plan": [],
    "workspace_files": [],
    "audit_log": [],
    "token_usage": {},
    "iteration_count": 0,
    "max_iterations": 10,
    "max_tokens": 5000
})

# Print the final response
print(result["messages"][-1].content)
```

---

## 🧪 Testing & Verification

### 1. Verifying Governance (Human-in-the-Loop)
To test the safety controls:
1. Start the Streamlit app.
2. Ask the agent to: *"Create a file named `test.txt` in the workspace with the content 'Hello World'."*
3. **Observe:** The agent should trigger a `write_file` tool call.
4. **Verify:** In the Streamlit UI, look for a warning: `⚠️ Action Required: Approval needed for write_file`.
5. **Action:** Click the **Approve write_file** button.
6. **Result:** The file should appear in the "Workspace Files" section in the sidebar.

### 2. Verifying Accuracy (Critic & Plan Checker)
To test the reasoning loop:
1. Ask the agent a complex, multi-step question: *"Research the history of LangChain and write a summary, then check if the summary is accurate."*
2. **Observe:** Watch the "Agent Thought" in the chat.
3. **Verify:** Check the **Audit Log** in the sidebar to see the `critic` and `plan_checker` nodes executing.
4. **Verify:** Check the **Current Plan** in the sidebar to see if the agent is checking off tasks as it completes them.

### 3. Verifying Skills
To test dynamic skill loading:
1. Ask the agent: *"Use the research skill to find the latest news on LLM quantization."*
2. **Observe:** The agent should log `📖 Loading Skill: research` in the chat.
3. **Verify:** The agent should use the `internet_search` tool to gather data.

---

## 🛠️ Troubleshooting

| Issue | Possible Cause | Solution |
|---|---|---|
| `ModuleNotFoundError` | Dependencies not installed | Run `uv sync` |
| `AuthenticationError` | Invalid API Key | Check your `.env` file |
| `Tavily API Error` | Missing or invalid Tavily Key | Ensure `TAVILY_API_KEY` is set in `.env` |
| `Streamlit Error` | Port 8501 is in use | Run `streamlit run app.py --port 8502` |
| `Agent hangs/loops` | Model is stuck in a reasoning loop | Check the **Audit Log** for repeated tool calls; increase `max_iterations` in your script. |
