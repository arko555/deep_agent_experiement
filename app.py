import streamlit as st
import os
import time
import base64
import logging
import uuid
from datetime import datetime
from langchain_core.messages import HumanMessage
from agent import get_deep_agent
from dotenv import load_dotenv
from src.core.guardrails import clear_workspace, get_workspace_root, validate_read_path
from src.core.memory import get_workspace_files, get_memory_content, get_skill_info

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app_new.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DeepAgentApp")
logger.info("App started")

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
get_workspace_root().mkdir(parents=True, exist_ok=True)

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
    .audit-log-entry {
        font-family: monospace;
        font-size: 0.85em;
        padding: 4px;
        border-bottom: 1px solid #21262d;
    }
    .audit-timestamp {
        color: #8b949e;
        margin-right: 8px;
    }
    .audit-action {
        color: #58a6ff;
        font-weight: bold;
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
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []
if "token_usage" not in st.session_state:
    st.session_state.token_usage = {"total": 0}
if "last_action" not in st.session_state:
    st.session_state.last_action = "None"
if "current_node" not in st.session_state:
    st.session_state.current_node = "Idle"
if "iteration_count" not in st.session_state:
    st.session_state.iteration_count = 0
# One checkpoint thread per browser session; the checkpointer holds the
# conversation history so each turn only sends the new user message.
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# --- Helper Functions ---
def update_workspace_files():
    st.session_state.workspace_files = get_workspace_files()


def read_workspace_bytes(relative_path):
    path = validate_read_path(str(get_workspace_root() / relative_path))
    with open(path, "rb") as file_bytes:
        return file_bytes.read()

def add_audit_entry(action: str, details: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.audit_log.append({
        "timestamp": timestamp,
        "action": action,
        "details": details
    })
    st.session_state.last_action = f"{action}: {details}"
    logger.info(f"[{timestamp}] {action}: {details}")

# --- Sidebar ---
with st.sidebar:
    st.image("https://github.com/deepagents/deepagents/raw/main/docs/logo.png", width=200) # Placeholder
    st.title("Deep Agent Context")

    st.divider()

    # Governance Dashboard
    st.subheader("🛡️ Governance")
    col1, col2 = st.columns(2)
    col1.metric("Iterations", st.session_state.iteration_count)
    col2.metric("Tokens", f"{st.session_state.token_usage.get('total', 0):,}")

    st.divider()

    # Dynamic Skills
    st.subheader("🛠️ Active Skills")
    skills_dir = "./skills"
    if os.path.exists(skills_dir):
        for skill_name in os.listdir(skills_dir):
            skill_path = os.path.join(skills_dir, skill_name)
            if os.path.isdir(skill_path) and not os.path.islink(skill_path):
                info = get_skill_info(skill_path)
                if info:
                    st.markdown(f"""
                    <div class="skill-card">
                        <div class="skill-name">{info.get('name', skill_name)}</div >
                        <div class="skill-desc">{info.get('description', '')}</div >
                    </div >
                    """, unsafe_allow_html=True)

    st.divider()

    # Audit Log
    st.subheader("📜 Audit Log")
    if not st.session_state.audit_log:
        st.caption("No actions recorded yet.")
    else:
        audit_container = st.container(height=200)
        with audit_container:
            for entry in reversed(st.session_state.audit_log):
                st.markdown(f"""
                <div class="audit-log-entry">
                    <span class="audit-timestamp">[{entry['timestamp']}]</span >
                    <span class="audit-action">{entry['action']}</span >: {entry['details']}
                </div >
                """, unsafe_allow_html=True)

    st.divider()
    st.subheader("📡 System Status")
    st.info(f"**Current Node:** {st.session_state.current_node}")
    st.info(f"**Last Action:** {st.session_state.last_action}")
    st.divider()

    # Memory / Conventions
    st.subheader("🧠 Shared Memory")
    agents_md = get_memory_content()
    if agents_md:
        st.caption("Context from AGENTS.md")
        st.markdown(agents_md)

    st.divider()

    # Workspace Files
    st.subheader("📂 Workspace Files")
    update_workspace_files()
    if not st.session_state.workspace_files:
        st.info("No files in workspace yet.")
    else:
        for f in st.session_state.workspace_files:
            with st.expander(f"📄 {f}"):
                try:
                    data = read_workspace_bytes(f)
                    st.code(data.decode("utf-8", errors="replace"), language="markdown")
                except Exception as e:
                    st.error(f"Could not read file: {e}")
                    data = None

                if data is not None:
                    st.download_button(
                        label="Download",
                        data=data,
                        file_name=f,
                        mime="text/markdown"
                    )

        if st.button("Clear Workspace"):
            clear_workspace()
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
    prompt = prompt.strip()
    if not prompt:
        st.warning("Please enter a valid message.")
        st.rerun()

    if len(prompt) < 3:
        st.warning("Please provide a more detailed request (at least 3 characters).")
        st.rerun()

    # 1. Persist user message to session state (displayed by chat history loop below)
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Render user message will happen on rerun after agent completion
    # 2. Process agent response
    with st.spinner("🤖 Agent is thinking..."):
        full_response = ""
        turn_messages = []  # Collect all intermediate messages for chat history

        try:
            st.write("🚀 Initializing agent execution...")
            # Only the new user message goes to the graph — the checkpointer
            # holds conversation history under this session's thread. The
            # iteration budget resets each turn so multi-turn chats don't
            # exhaust it.
            for event in st.session_state.agent.stream(
                {"messages": [HumanMessage(content=prompt)], "iteration_count": 0},
                config={"configurable": {"thread_id": st.session_state.thread_id}},
                stream_mode="updates"
            ):
                for node_name, data in event.items():
                    if not isinstance(data, dict):
                        continue

                    st.write(f"🔹 **Node `{node_name}`** is active")

                    # --- Handle Thinking / Reasoning ---
                    if "messages" in data:
                        messages = data["messages"]
                        if not isinstance(messages, list):
                            if hasattr(messages, "value") and isinstance(messages.value, list):
                                messages = messages.value
                            else:
                                messages = []

                        for msg in messages:
                            if hasattr(msg, "content") and msg.content:
                                # Map LangChain role names to Streamlit role names
                                role = getattr(msg, "type", "assistant")
                                if role == "ai":
                                    role = "assistant"
                                elif role == "human":
                                    role = "user"
                                elif role == "system":
                                    continue  # Skip system messages in chat UI

                                # Display in thinking container
                                if node_name == "agent":
                                    st.markdown(f"**Agent Thought:** {msg.content}")
                                elif node_name == "responder":
                                    full_response = msg.content
                                    st.success("✅ Final response generated.")

                                # Only capture the final responder message in chat history
                                if node_name == "responder":
                                    turn_messages.append({
                                        "role": role,
                                        "content": msg.content,
                                    })

                    # --- Handle Todo Updates directly ---
                    if "todos" in data:
                        st.session_state.current_plan = data["todos"]
                        with plan_section.container():
                            st.divider()
                            st.subheader("📋 Current Plan")
                            for i, t in enumerate(st.session_state.current_plan):
                                st.checkbox(str(t), key=f"plan_update_{i}_{time.time()}", value=False, disabled=True)

                    # --- Handle Audit Log updates ---
                    if "audit_log" in data:
                        st.session_state.audit_log.extend(data["audit_log"])

                    # --- Handle Token Usage ---
                    if "token_usage" in data:
                        st.session_state.token_usage.update(data["token_usage"])

                    # --- Handle Iteration Count ---
                    if "iteration_count" in data:
                        st.session_state.iteration_count = data["iteration_count"]

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            st.error(f"❌ **Agent Execution Error**")
            st.info(f"**Error Details:** {e}")

    # 3. Persist assistant messages and always refresh UI
    for msg in turn_messages:
        st.session_state.messages.append(msg)
    update_workspace_files()
    st.rerun()  # Always rerun so the full chat history (user + assistant) renders correctly
