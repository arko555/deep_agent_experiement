import streamlit as st
import os
import time
import base64
from datetime import datetime
from agent import get_deep_agent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="Deep Agent Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure workspace exists
workspace_dir = os.getenv("WORKSPACE_ROOT", "./workspace")
if not os.path.exists(workspace_dir):
    os.makedirs(workspace_dir)

# --- Custom Styling ---
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stChatMessage[data-testid="stChatMessageUser"] {
        background-color: #1a1c24;
        border: 1px solid #30363d;
    }
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background-color: #161b22;
        border: 1px solid #238636;
    }
    .sidebar .sidebar-content {
        background-color: #0d1117;
    }
    .skill-card {
        background-color: #1a1c24;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .skill-name {
        color: #58a6ff;
        font-weight: bold;
        font-size: 1.1em;
    }
    .skill-desc {
        color: #8b949e;
        font-size: 0.9em;
    }
    .thinking-process {
        border-left: 2px solid #238636;
        padding-left: 15px;
        margin-bottom: 20px;
        color: #8b949e;
        font-style: italic;
    }
    .workspace-file {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 5px 0;
        border-bottom: 1px solid #21262d;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    with st.spinner("Initializing Deep Agent..."):
        st.session_state.agent = get_deep_agent()
if "current_plan" not in st.session_state:
    st.session_state.current_plan = []
if "workspace_files" not in st.session_state:
    st.session_state.workspace_files = []

# --- Helper Functions ---
def update_workspace_files():
    workspace_dir = os.getenv("WORKSPACE_ROOT", "./workspace")
    if os.path.exists(workspace_dir):
        files = []
        for root, dirs, filenames in os.walk(workspace_dir):
            for f in filenames:
                rel_path = os.path.relpath(os.path.join(root, f), workspace_dir)
                files.append(rel_path)
        st.session_state.workspace_files = sorted(files)

def get_skill_info(skill_path):
    skill_md = os.path.join(skill_path, "SKILL.md")
    if os.path.exists(skill_md):
        with open(skill_md, "r") as f:
            content = f.read()
            # Simple YAML-ish header parser
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
    return None

# --- Sidebar ---
with st.sidebar:
    st.image("https://github.com/deepagents/deepagents/raw/main/docs/logo.png", width=200) # Placeholder or actual logo
    st.title("Deep Agent Context")
    
    st.divider()
    
    # Dynamic Skills
    st.subheader("🛠️ Active Skills")
    skills_dir = "./skills"
    if os.path.exists(skills_dir):
        for skill_name in os.listdir(skills_dir):
            skill_path = os.path.join(skills_dir, skill_name)
            if os.path.isdir(skill_path):
                info = get_skill_info(skill_path)
                if info:
                    st.markdown(f"""
                    <div class="skill-card">
                        <div class="skill-name">{info.get('name', skill_name)}</div>
                        <div class="skill-desc">{info.get('description', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    st.divider()

    # Memory / Conventions
    st.subheader("🧠 Shared Memory")
    memory_file = "./AGENTS.md"
    if os.path.exists(memory_file):
        with open(memory_file, "r") as f:
            st.caption("Context from AGENTS.md")
            st.markdown(f.read())

    st.divider()

    # Workspace Files
    st.subheader("📂 Workspace Files")
    update_workspace_files()
    if not st.session_state.workspace_files:
        st.info("No files in workspace yet.")
    else:
        for f in st.session_state.workspace_files:
            with st.expander(f"📄 {f}"):
                file_path = os.path.join(os.getenv("WORKSPACE_ROOT", "./workspace"), f)
                try:
                    with open(file_path, "r") as file_text:
                        st.code(file_text.read(), language="markdown")
                except Exception as e:
                    st.error(f"Could not read file: {e}")
                
                # Download button
                with open(file_path, "rb") as file_bytes:
                    st.download_button(
                        label="Download",
                        data=file_bytes,
                        file_name=f,
                        mime="text/markdown"
                    )
    
    if st.button("Clear Workspace"):
        workspace_dir = os.getenv("WORKSPACE_ROOT", "./workspace")
        for f in os.listdir(workspace_dir):
            file_path = os.path.join(workspace_dir, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
        update_workspace_files()
        st.rerun()

    # Plan Placeholder in Sidebar
    plan_section = st.empty()
    if st.session_state.current_plan:
        with plan_section.container():
            st.divider()
            st.subheader("📋 Current Plan")
            for i, task in enumerate(st.session_state.current_plan):
                st.checkbox(str(task), key=f"plan_init_{i}", value=False, disabled=True)

# --- Main Interface ---
st.title("🚀 Deep Agent Orchestrator")
st.markdown("*Demonstrating hierarchical planning, specialist subagents, and dynamic skill loading.*")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like me to do?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process agent response
    with st.chat_message("assistant"):
        thinking_container = st.container()
        status_placeholder = st.empty()
        
        # --- Thinking Process Visualization ---
        # Note: We use st.status to show the "Thinking" state and internal steps
        with st.status("Agent is thinking...", expanded=True) as status:
            full_response = ""
            
            # DeepAgent streaming
            try:
                # We expect the stream to yield state updates or events
                # LangGraph stream("values") yields the full state at each step
                # stream("updates") yields only the changes
                for event in st.session_state.agent.stream(
                    {"messages": [("user", prompt)]}, 
                    stream_mode="updates"
                ):
                    # event is usually a dict: { 'node_name': { 'key': 'value' } }
                    for node_name, data in event.items():
                        if not isinstance(data, dict):
                            continue
                            
                        # --- Handle Thinking / Reasoning ---
                        if "messages" in data:
                            messages = data["messages"]
                            # LangGraph may return an Overwrite object which is not iterable
                            if not isinstance(messages, list):
                                # If it has a .value property (like from langgraph.constants.Overwrite)
                                if hasattr(messages, "value") and isinstance(messages.value, list):
                                    messages = messages.value
                                else:
                                    messages = []

                            for msg in messages:
                                # If it's an AI Message with content, it might be a thought or final answer
                                if hasattr(msg, "content") and msg.content:
                                    if node_name == "agent":
                                        st.markdown(f"**Agent Thought:** {msg.content}")
                                    else:
                                        full_response = msg.content
                                
                                # --- Handle Tool Calls ---
                                if hasattr(msg, "tool_calls") and msg.tool_calls:
                                    for tc in msg.tool_calls:
                                        tool_name = tc.get("name")
                                        tool_args = tc.get("args", {})
                                        
                                        if tool_name == "task":
                                            subagent = tool_args.get("subagent_type", "unknown")
                                            st.write(f"👥 **Delegating to Subagent:** `{subagent}`")
                                            st.caption(f"Reason: {tool_args.get('description', '')}")
                                        elif tool_name == "write_todos":
                                            st.session_state.current_plan = tool_args.get("todos", [])
                                            st.success("📍 **Plan Updated**")
                                        elif tool_name == "read_file" and "SKILL.md" in str(tool_args.get("path", "")):
                                            skill_name = os.path.basename(os.path.dirname(tool_args["path"]))
                                            st.info(f"📖 **Loading Skill:** `{skill_name}`")
                                        else:
                                            st.write(f"🛠️ **Tool Call:** `{tool_name}`")
                                            if tool_args:
                                                st.caption(f"Args: `{tool_args}`")

                        # --- Handle Todo Updates directly ---
                        if "todos" in data:
                            st.session_state.current_plan = data["todos"]
                            with plan_section.container():
                                st.divider()
                                st.subheader("📋 Current Plan")
                                for i, t in enumerate(st.session_state.current_plan):
                                    st.checkbox(str(t), key=f"plan_update_{i}_{time.time()}", value=False, disabled=True)

                status.update(label="Process Complete!", state="complete", expanded=False)
            except Exception as e:
                import traceback
                st.error(f"Error during agent execution: {e}")
                st.code(traceback.format_exc())
                status.update(label="error occurred", state="error")
                full_response = "I encountered an error while processing your request."

        st.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        update_workspace_files()

