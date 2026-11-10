from __future__ import annotations

import os
import time
from typing import Any

import streamlit as st

from api_client import APIError, RAGNexusAPI


# ---------------------------------------------------------------------
# Page configuration and dark visual system
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="RAGNexus | Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
    }

    .stApp {
        background:
            radial-gradient(circle at 50% -15%, rgba(76, 110, 245, .16), transparent 38%),
            #0b0f19;
        color: #e8edf7;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 15, 25, .82);
    }

    .block-container {
        max-width: 1480px;
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }

    [data-testid="stSidebar"] {
        background: #0e1422;
        border-right: 1px solid rgba(148, 163, 184, .14);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.4rem;
    }

    .rn-brand {
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -.04em;
        color: #f8fafc;
    }

    .rn-muted {
        color: #94a3b8;
        font-size: .9rem;
    }

    [data-testid="stVerticalBlockBorderWrapper"]:has(.rn-auth-marker) {
        max-width: 540px;
        margin: 5vh auto 0 auto;
        padding: 1.8rem 2rem 1.5rem 2rem;
        background: linear-gradient(145deg, rgba(23, 32, 52, .98), rgba(15, 22, 37, .98));
        border: 1px solid rgba(148, 163, 184, .20);
        border-radius: 24px;
        box-shadow: 0 24px 80px rgba(0, 0, 0, .38);
    }

    .rn-auth-marker { display: none; }

    .rn-auth-logo {
        width: 58px;
        height: 58px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1rem auto;
        border-radius: 18px;
        background: linear-gradient(135deg, #6366f1, #06b6d4);
        font-size: 1.8rem;
        box-shadow: 0 10px 30px rgba(79, 70, 229, .28);
    }

    .rn-auth-title {
        text-align: center;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -.045em;
        color: #f8fafc;
        margin-bottom: .2rem;
    }

    .rn-auth-subtitle {
        text-align: center;
        color: #9caec5;
        margin-bottom: 1.5rem;
    }

    .rn-section-title {
        font-size: 1.35rem;
        font-weight: 750;
        letter-spacing: -.025em;
        color: #f1f5f9;
    }

    .rn-chat-welcome {
        text-align: center;
        padding: 12vh 1rem 5vh 1rem;
    }

    .rn-chat-welcome h1 {
        font-size: clamp(2rem, 4vw, 3.2rem);
        letter-spacing: -.055em;
        font-weight: 800;
        color: #f8fafc;
    }

    .rn-chat-welcome p {
        color: #94a3b8;
        font-size: 1.05rem;
    }

    .rn-panel {
        background: rgba(20, 29, 47, .72);
        border: 1px solid rgba(148, 163, 184, .16);
        border-radius: 16px;
        padding: 1rem 1.15rem;
    }

    .rn-doc-row {
        padding: .85rem 1rem;
        border: 1px solid rgba(148, 163, 184, .14);
        border-radius: 12px;
        background: rgba(15, 23, 42, .58);
        margin-bottom: .55rem;
    }

    div[data-testid="stChatMessage"] {
        background: rgba(20, 29, 47, .55);
        border: 1px solid rgba(148, 163, 184, .13);
        border-radius: 16px;
        padding: .9rem 1rem;
    }

    .stButton > button {
        border-radius: 10px;
        min-height: 2.65rem;
        font-weight: 650;
    }

    .stTextInput input, .stTextArea textarea {
        border-radius: 10px;
    }

    [data-testid="stMetric"] {
        background: rgba(20, 29, 47, .72);
        border: 1px solid rgba(148, 163, 184, .14);
        border-radius: 13px;
        padding: .8rem 1rem;
    }

    #MainMenu, footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Session state and API client
# ---------------------------------------------------------------------
def init_state() -> None:
    defaults = {
        "api_base": os.getenv("RAGNEXUS_API_URL", "http://127.0.0.1:8000/api/v1"),
        "token": None,
        "user": None,
        "workspaces": [],
        "workspace_id": None,
        "documents": [],
        "session_id": None,
        "messages": [],
        "last_chat_meta": None,
        "page": "Chat",
        "auth_mode": "Login",
        "chat_visibility": "private",
        "document_scope": "workspace",
        "selected_document_ids": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


def api() -> RAGNexusAPI:
    return RAGNexusAPI(st.session_state.api_base, st.session_state.token)


def reset_chat() -> None:
    st.session_state.session_id = None
    st.session_state.messages = []
    st.session_state.last_chat_meta = None


def clear_auth() -> None:
    st.session_state.token = None
    st.session_state.user = None
    st.session_state.workspaces = []
    st.session_state.workspace_id = None
    st.session_state.documents = []
    reset_chat()


def show_api_error(exc: APIError) -> None:
    if exc.status_code == 401:
        clear_auth()
        st.error("Your session has expired. Please sign in again.")
        st.rerun()
    st.error(
        f"Request failed{f' ({exc.status_code})' if exc.status_code else ''}: {exc}"
    )


def refresh_documents() -> None:
    workspace_id = st.session_state.workspace_id
    st.session_state.documents = (
        api().documents(workspace_id) if workspace_id else []
    ) or []


def refresh_workspaces() -> None:
    st.session_state.workspaces = api().workspaces() or []
    valid_ids = {str(w["id"]) for w in st.session_state.workspaces}
    if st.session_state.workspace_id not in valid_ids:
        st.session_state.workspace_id = (
            str(st.session_state.workspaces[0]["id"])
            if st.session_state.workspaces else None
        )
    refresh_documents()


# ---------------------------------------------------------------------
# Authentication: centered in the MAIN page, never in the sidebar
# ---------------------------------------------------------------------
def render_auth_page() -> None:
    _, center, _ = st.columns([1, 1.2, 1])
    with center:
        with st.container(border=True):
            st.markdown(
                """
                <span class="rn-auth-marker"></span>
                <div class="rn-auth-logo">🧠</div>
                <div class="rn-auth-title">RAGNexus</div>
                <div class="rn-auth-subtitle">
                    Your private knowledge assistant
                    <br/>Sign in to access your documents and conversations.
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.session_state.auth_mode = st.radio(
                "Account",
                ["Login", "Register"],
                horizontal=True,
                label_visibility="collapsed",
                key="auth_mode_selector",
            )

            with st.form("main_auth_form", clear_on_submit=False):
                email = st.text_input("Email address", placeholder="you@example.com")
                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Enter your password",
                )
                if st.session_state.auth_mode == "Register":
                    st.caption("Use a strong password of at least 12 characters.")
                submitted = st.form_submit_button(
                    "Sign in" if st.session_state.auth_mode == "Login"
                    else "Create account",
                    type="primary",
                    use_container_width=True,
                )

            if submitted:
                email = email.strip()
                if not email or not password:
                    st.error("Enter both email address and password.")
                    return
                if st.session_state.auth_mode == "Register" and len(password) < 12:
                    st.error("Use a password with at least 12 characters.")
                    return
                try:
                    if st.session_state.auth_mode == "Register":
                        api().register(email, password)
                        st.success("Account created. Sign in with your new credentials.")
                        st.session_state.auth_mode = "Login"
                        st.rerun()
                    else:
                        result = api().login(email, password)
                        st.session_state.token = result["access_token"]
                        st.session_state.user = result.get("user") or {"email": email}
                        refresh_workspaces()
                        st.session_state.page = "Chat"
                        st.rerun()
                except APIError as exc:
                    st.error(str(exc))


# Do not render the sidebar or any backend/health-check control before login.
if not st.session_state.token:
    render_auth_page()
    st.stop()


# ---------------------------------------------------------------------
# Authenticated sidebar: navigation + profile
# ---------------------------------------------------------------------
try:
    if not st.session_state.workspaces:
        refresh_workspaces()
except APIError as exc:
    show_api_error(exc)

with st.sidebar:
    st.markdown('<div class="rn-brand">🧠 RAGNexus</div>', unsafe_allow_html=True)
    st.caption("Knowledge workspace")
    st.divider()

    if st.button("＋  New chat", type="primary", use_container_width=True):
        reset_chat()
        st.session_state.page = "Chat"
        st.rerun()

    st.session_state.page = st.radio(
        "Navigation",
        ["Chat", "Documents"],
        index=0 if st.session_state.page == "Chat" else 1,
        label_visibility="collapsed",
    )

    st.divider()
    workspaces = st.session_state.workspaces or []
    if workspaces:
        workspace_labels = {
            str(w["id"]): f"{w.get('name', 'Workspace')} · {w.get('workspace_type', 'workspace')}"
            for w in workspaces
        }
        workspace_ids = list(workspace_labels)
        current_id = (
            st.session_state.workspace_id
            if st.session_state.workspace_id in workspace_ids
            else workspace_ids[0]
        )
        chosen_id = st.selectbox(
            "Workspace",
            workspace_ids,
            index=workspace_ids.index(current_id),
            format_func=lambda value: workspace_labels[value],
        )
        if chosen_id != st.session_state.workspace_id:
            st.session_state.workspace_id = chosen_id
            reset_chat()
            try:
                refresh_documents()
            except APIError as exc:
                show_api_error(exc)
            st.rerun()
    else:
        st.warning("No workspace is available for this account.")

    with st.expander("＋ Create team workspace"):
        team_name = st.text_input("Workspace name", key="new_team_workspace_name")
        if st.button("Create workspace", use_container_width=True):
            if not team_name.strip():
                st.warning("Enter a workspace name.")
            else:
                try:
                    api().create_workspace(team_name.strip())
                    refresh_workspaces()
                    st.success("Workspace created.")
                    st.rerun()
                except APIError as exc:
                    show_api_error(exc)

    st.divider()
    user = st.session_state.user or {}
    email = user.get("email") or "Signed-in user"
    st.caption("PROFILE")
    st.markdown(f"**{email}**")
    if st.button("Sign out", use_container_width=True):
        clear_auth()
        st.rerun()


# ---------------------------------------------------------------------
# Workspace context
# ---------------------------------------------------------------------
workspace_id = st.session_state.workspace_id
if not workspace_id:
    st.title("RAGNexus")
    st.warning("No workspace is available. Check the account's workspace provisioning.")
    st.stop()

workspace = next(
    (w for w in st.session_state.workspaces if str(w["id"]) == str(workspace_id)),
    {},
)
documents = st.session_state.documents or []
indexed_docs = [
    d for d in documents
    if d.get("status") in ("indexed", "completed") and d.get("is_current", True)
]


# ---------------------------------------------------------------------
# Documents page
# ---------------------------------------------------------------------
def render_documents_page() -> None:
    st.markdown('<div class="rn-section-title">Document library</div>', unsafe_allow_html=True)
    st.caption(
        f"{workspace.get('name', 'Workspace')} · Manage files and monitor ingestion"
    )

    total = len(documents)
    indexed = len(indexed_docs)
    failed = sum(1 for d in documents if d.get("status") == "failed")
    processing = sum(
        1 for d in documents if d.get("status") in ("pending", "processing", "running")
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total documents", total)
    m2.metric("Indexed", indexed)
    m3.metric("Processing", processing)
    m4.metric("Failed", failed)

    st.divider()
    st.markdown("### Add documents")
    st.caption("Supported formats: PDF, TXT, CSV, DOCX and HTML.")
    uploaded = st.file_uploader(
        "Drop a file here or browse",
        type=["pdf", "txt", "csv", "docx", "html", "htm"],
        key=f"document_upload_{workspace_id}",
    )

    if uploaded and st.button("Upload and ingest", type="primary"):
        try:
            with st.spinner(f"Uploading {uploaded.name}..."):
                result = api().upload(
                    workspace_id,
                    uploaded.name,
                    uploaded.getvalue(),
                    uploaded.type,
                )
            st.success(
                f"Uploaded {result.get('filename', uploaded.name)}. "
                f"Initial status: {result.get('status', 'queued')}."
            )

            job_id = result.get("ingestion_job_id")
            if job_id:
                progress = st.empty()
                latest: dict[str, Any] | None = None
                for attempt in range(45):
                    latest = api().ingestion_job(str(job_id))
                    job_status = str(latest.get("status", "unknown")).lower()
                    progress.info(
                        f"Ingestion: **{job_status}** · "
                        f"Checking progress ({attempt + 1}/45)"
                    )
                    if job_status in ("completed", "indexed", "failed"):
                        break
                    time.sleep(2)

                if latest:
                    final_status = str(latest.get("status", "unknown")).lower()
                    if final_status in ("completed", "indexed"):
                        progress.success("Document ingestion completed.")
                    elif final_status == "failed":
                        reason = (
                            latest.get("error_message")
                            or latest.get("error_code")
                            or "No failure details were returned by the API."
                        )
                        progress.error(f"Ingestion failed: {reason}")
                    else:
                        progress.warning(
                            "The job is still running. Refresh the document list shortly."
                        )
            else:
                st.warning(
                    "Upload was accepted, but the API response did not include an "
                    "ingestion job ID. Check the document status after refreshing."
                )

            refresh_documents()
            st.rerun()
        except APIError as exc:
            show_api_error(exc)

    st.divider()
    header_left, header_right = st.columns([4, 1])
    with header_left:
        st.markdown("### Workspace documents")
    with header_right:
        if st.button("↻ Refresh", use_container_width=True):
            try:
                refresh_documents()
                st.rerun()
            except APIError as exc:
                show_api_error(exc)

    if not documents:
        st.info("No documents yet. Upload a file to start building this workspace's knowledge base.")
        return

    for item in documents:
        status = str(item.get("status", "unknown")).lower()
        if status in ("indexed", "completed"):
            status_label = "🟢 Indexed"
        elif status == "failed":
            status_label = "🔴 Failed"
        else:
            status_label = f"🟡 {status.title()}"

        with st.container(border=True):
            top_l, top_r = st.columns([4, 1])
            with top_l:
                st.markdown(f"**📄 {item.get('filename', 'Unnamed document')}**")
                st.caption(
                    f"{status_label} · Version {item.get('version_number', 1)}"
                )
            with top_r:
                st.caption(
                    f"{item.get('file_size_bytes', item.get('size_bytes', ''))}"
                )
            error_message = item.get("error_message") or item.get("failure_reason")
            if status == "failed" and error_message:
                st.error(str(error_message))


# ---------------------------------------------------------------------
# Chat page
# ---------------------------------------------------------------------
def create_session_if_needed() -> str:
    if st.session_state.session_id:
        return str(st.session_state.session_id)

    scope = st.session_state.document_scope
    selected_ids = st.session_state.selected_document_ids if scope == "selected" else []

    created = api().create_chat_session(
        workspace_id=workspace_id,
        visibility=st.session_state.chat_visibility,
        document_scope=scope,
        document_ids=selected_ids,
    )
    st.session_state.session_id = str(created["id"])
    return st.session_state.session_id


def load_chat_history() -> None:
    if st.session_state.session_id and not st.session_state.messages:
        history = api().chat_messages(st.session_state.session_id) or []
        st.session_state.messages = [
            {
                "role": message["role"],
                "content": message["content"],
                "metadata": message.get("message_metadata") or {},
            }
            for message in history
            if message.get("role") in ("user", "assistant")
        ]


def render_citations(citations: list[dict[str, Any]]) -> None:
    if not citations:
        return
    with st.expander(f"Sources · {len(citations)}"):
        for cite in citations:
            source_no = cite.get("source_number", "?")
            filename = cite.get("filename") or "Document"
            st.markdown(f"**[{source_no}] {filename}**")
            st.caption(
                f"Document: {cite.get('document_id', 'N/A')} · "
                f"Chunk: {cite.get('chunk_id', 'N/A')}"
            )


def render_chat_page() -> None:
    title_col, context_col = st.columns([3, 2])
    with title_col:
        st.markdown('<div class="rn-section-title">Knowledge Assistant</div>', unsafe_allow_html=True)
        st.caption(f"Workspace · {workspace.get('name', 'Unknown')}")
    with context_col:
        st.caption(
            f"{len(indexed_docs)} indexed document(s) available"
        )

    with st.expander("Conversation settings", expanded=not bool(st.session_state.session_id)):
        setting_a, setting_b = st.columns(2)
        with setting_a:
            st.session_state.chat_visibility = st.selectbox(
                "Conversation visibility",
                ["private", "workspace"],
                index=0 if st.session_state.chat_visibility == "private" else 1,
                disabled=bool(st.session_state.session_id),
            )
        with setting_b:
            st.session_state.document_scope = st.selectbox(
                "Knowledge scope",
                ["workspace", "selected"],
                index=0 if st.session_state.document_scope == "workspace" else 1,
                disabled=bool(st.session_state.session_id),
                help="Workspace searches all eligible indexed documents. Selected searches only the documents you choose.",
            )

        if st.session_state.document_scope == "selected":
            options = [str(d["id"]) for d in indexed_docs]
            labels = {
                str(d["id"]): d.get("filename", str(d["id"]))
                for d in indexed_docs
            }
            st.session_state.selected_document_ids = st.multiselect(
                "Select indexed documents",
                options=options,
                default=[
                    value for value in st.session_state.selected_document_ids
                    if value in options
                ],
                format_func=lambda value: labels.get(value, value),
                disabled=bool(st.session_state.session_id),
            )
            if not indexed_docs:
                st.warning("There are no indexed documents yet. Upload and index documents first.")
        else:
            st.caption("The conversation can retrieve from eligible indexed documents in this workspace.")

    if not indexed_docs:
        st.warning(
            "No indexed documents are available yet. You can explore the chat interface, "
            "but document-grounded answers require successful ingestion."
        )

    try:
        load_chat_history()
    except APIError as exc:
        show_api_error(exc)

    if not st.session_state.messages:
        st.markdown(
            """
            <div class="rn-chat-welcome">
                <h1>What would you like to know?</h1>
                <p>Ask questions across your documents. RAGNexus will retrieve relevant evidence and cite its sources.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        suggestion_cols = st.columns(3)
        suggestions = [
            "Summarize the key points in my documents",
            "What are the most important facts?",
            "Compare the information across my files",
        ]
        for col, suggestion in zip(suggestion_cols, suggestions):
            with col:
                st.markdown(
                    f'<div class="rn-panel"><span class="rn-muted">{suggestion}</span></div>',
                    unsafe_allow_html=True,
                )
        st.write("")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_citations((message.get("metadata") or {}).get("citations") or [])

    query = st.chat_input("Message RAGNexus...")
    if query:
        try:
            session_id = create_session_if_needed()
            st.session_state.messages.append(
                {"role": "user", "content": query, "metadata": {}}
            )
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("Searching your documents and preparing an answer..."):
                    result = api().send_message(session_id, query)
                answer = result.get("answer") or "The API returned an empty answer."
                st.markdown(answer)
                citations = result.get("citations") or []
                render_citations(citations)
                if result.get("abstained"):
                    st.info("I couldn't find sufficient supporting evidence in the selected documents.")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "metadata": {"citations": citations},
                }
            )
            st.session_state.last_chat_meta = result
            st.rerun()
        except APIError as exc:
            show_api_error(exc)

    if st.session_state.last_chat_meta:
        result = st.session_state.last_chat_meta
        with st.expander("Answer details", expanded=False):
            d1, d2, d3 = st.columns(3)
            d1.metric("Retrieved chunks", result.get("retrieved_count", 0))
            d2.metric("Latency", f"{result.get('latency_ms', 0):.0f} ms")
            d3.metric("Model", result.get("model") or "N/A")
            st.caption(
                "Diagnostics are informational. They do not establish answer correctness."
            )


# ---------------------------------------------------------------------
# Route the authenticated UI
# ---------------------------------------------------------------------
if st.session_state.page == "Documents":
    render_documents_page()
else:
    render_chat_page()
