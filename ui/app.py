from __future__ import annotations

import os
import time
import html
from typing import Any
from uuid import uuid4

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

    /* Keep app content below Streamlit's own header/Deploy toolbar. */
    [data-testid="stHeader"] {
        background: rgba(11, 15, 25, .96);
        min-height: 3.25rem;
        z-index: 1000;
    }

    [data-testid="stToolbar"] {
        z-index: 1001;
    }

    .block-container,
    [data-testid="stAppViewContainer"] .main .block-container {
        max-width: 1480px;
        /* Streamlit Cloud can overlay the top-right area with its Deploy control. */
        padding-top: 5.25rem !important;
        padding-bottom: 2.5rem;
    }

    @media (max-width: 768px) {
        .block-container,
        [data-testid="stAppViewContainer"] .main .block-container {
            padding-top: 4.25rem !important;
            padding-left: 1rem;
            padding-right: 1rem;
        }
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

    .rn-pipeline-badge {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 2px 8px;
        margin: 2px 0 5px 0;
        font-size: 10px;
        font-weight: 750;
        letter-spacing: .06em;
        color: #dbeafe;
        background: rgba(59, 130, 246, .18);
        border: 1px solid rgba(96, 165, 250, .35);
    }
    .rn-pipeline-badge.agentic {
        color: #ede9fe;
        background: rgba(139, 92, 246, .18);
        border-color: rgba(167, 139, 250, .35);
    }
    .rn-metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 7px;
        margin: 5px 0 8px 0;
    }
    .rn-metric-tile {
        min-width: 0;
        padding: 7px 9px;
        border: 1px solid rgba(148, 163, 184, .14);
        border-radius: 9px;
        background: rgba(20, 29, 47, .58);
    }
    .rn-metric-label {
        color: #94a3b8;
        font-size: 10px;
        line-height: 1.25;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .rn-metric-value {
        color: #e8edf7;
        font-size: 14px;
        line-height: 1.35;
        font-weight: 700;
        overflow-wrap: anywhere;
    }
    .rn-metric-meta {
        color: #94a3b8;
        font-size: 10px;
        line-height: 1.4;
        margin-top: 5px;
    }
    [data-testid="stChatInput"] {
        z-index: 1002;
    }
    [data-testid="stChatInput"] textarea {
        border-radius: 14px;
        line-height: 1.45;
    }
    @media (max-width: 700px) {
        .rn-metrics-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
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
        "history_search": "",
        "profile_loaded": False,
        "pending_chat_request_id": None,
        "pending_chat_query": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


def chat_request_timeout() -> int:
    try:
        configured = int(os.getenv("RAGNEXUS_CHAT_TIMEOUT_SECONDS", "600"))
    except (TypeError, ValueError):
        configured = 600
    return max(120, min(configured, 1800))


def api() -> RAGNexusAPI:
    client = st.session_state.get("_api_client")
    if client is None or client.base_url != st.session_state.api_base.rstrip("/"):
        client = RAGNexusAPI(st.session_state.api_base, st.session_state.token)
        st.session_state._api_client = client
    else:
        client.set_token(st.session_state.token)
    return client


def reset_chat() -> None:
    st.session_state.session_id = None
    st.session_state.messages = []
    st.session_state.last_chat_meta = None
    st.session_state.pending_chat_request_id = None
    st.session_state.pending_chat_query = None


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
                confirm_password = None
                if st.session_state.auth_mode == "Register":
                    confirm_password = st.text_input("Confirm password", type="password", placeholder="Re-enter your password")
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
                if st.session_state.auth_mode == "Register" and password != confirm_password:
                    st.error("Password and confirmation do not match.")
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
        ["Chat", "Documents", "Profile"],
        index={"Chat": 0, "Documents": 1, "Profile": 2}.get(st.session_state.page, 0),
        label_visibility="collapsed",
    )

    st.divider()
    if st.session_state.page == "Chat" and st.session_state.workspace_id:
        st.caption("RECENT CONVERSATIONS")
        history_query = st.text_input("Search chats", key="history_search", placeholder="Search conversations")
        try:
            history_rows = api().chat_sessions(st.session_state.workspace_id, history_query, limit=40)
            for chat in history_rows:
                chat_id = str(chat["id"])
                chat_title = chat.get("title") or "New conversation"
                hcol, mcol = st.columns([5, 1])
                with hcol:
                    if st.button(chat_title[:34], key=f"open_chat_{chat_id}", use_container_width=True):
                        st.session_state.session_id = chat_id
                        st.session_state.messages = []
                        st.session_state.last_chat_meta = None
                        st.session_state.page = "Chat"
                        st.rerun()
                with mcol:
                    with st.popover("⋯"):
                        new_title = st.text_input("Rename", value=chat_title, key=f"rename_title_{chat_id}")
                        if st.button("Save title", key=f"rename_{chat_id}"):
                            api().rename_chat_session(chat_id, new_title)
                            st.rerun()
                        if st.button("Delete chat", key=f"delete_chat_{chat_id}"):
                            api().delete_chat_session(chat_id)
                            if st.session_state.session_id == chat_id:
                                reset_chat()
                            st.rerun()
        except APIError as exc:
            st.caption(f"History unavailable: {exc}")

    st.divider()
    workspaces = st.session_state.workspaces or []
    if workspaces:
        workspace_labels = {
            str(w["id"]): f"{w.get('name', 'Workspace')} · {w.get('role', 'member')}"
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
            st.session_state.selected_document_ids = []
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
    if st.button("👤  Profile", key="sidebar_profile_nav", use_container_width=True):
        st.session_state.page = "Profile"
        st.rerun()
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
    uploaded_files = st.file_uploader(
        "Drop documents here or browse (select multiple files)",
        type=["pdf", "txt", "csv", "docx", "html", "htm"],
        accept_multiple_files=True,
        key=f"document_upload_{workspace_id}",
    )

    if uploaded_files and st.button("Upload and ingest", type="primary"):
        try:
            payload = [(item.name, item.getvalue(), item.type) for item in uploaded_files]
            with st.spinner(f"Submitting {len(payload)} document(s)..."):
                batch = api().upload_many(workspace_id, payload)
            for item in batch.get("results", []):
                filename = item.get("filename") or "Document"
                if item.get("duplicate"):
                    st.info(f"{filename}: document already exists in this workspace; skipped duplicate ingestion.")
                elif item.get("status") == "rejected":
                    st.error(f"{filename}: {item.get('error', 'Upload rejected')}")
                elif item.get("status") == "dispatch_failed":
                    st.error(f"{filename}: saved, but queue dispatch failed. Check Redis/Celery.")
                else:
                    st.success(f"{filename}: accepted ({item.get('status', 'pending')}).")
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
                if status == "indexed":
                    st.caption(f"{int(item.get('indexed_chunk_count') or 0):,} chunks")
                else:
                    st.caption("Chunks pending")
            if status == "indexed":
                embedding_model = item.get("embedding_model") or "Embedding model unavailable"
                embedding_dimension = item.get("embedding_dimension")
                dimension_label = f" · {int(embedding_dimension)} dimensions" if embedding_dimension else ""
                st.caption(f"Embedding · {embedding_model}{dimension_label}")
            error_message = item.get("error_message") or item.get("failure_reason")
            if status == "failed" and error_message:
                st.error(str(error_message))
            if status != "deleted":
                with st.popover("Document actions"):
                    st.warning("Deleting removes this document from retrieval and removes its stored file.")
                    confirm = st.checkbox("I understand this action cannot be undone", key=f"confirm_delete_doc_{item['id']}")
                    if st.button("Delete document", key=f"delete_doc_{item['id']}", type="primary", disabled=not confirm):
                        try:
                            api().delete_document(workspace_id, str(item["id"]))
                            refresh_documents()
                            st.success("Document deleted.")
                            st.rerun()
                        except APIError as exc:
                            show_api_error(exc)


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


def load_chat_history(*, force: bool = False) -> None:
    if st.session_state.session_id and (force or not st.session_state.messages):
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
        pending_id = st.session_state.get("pending_chat_request_id")
        if pending_id and any(
            message["role"] == "assistant"
            and (message.get("metadata") or {}).get("client_request_id") == pending_id
            for message in st.session_state.messages
        ):
            st.session_state.pending_chat_request_id = None
            st.session_state.pending_chat_query = None
        elif st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            # Also recover the pending state after a browser refresh, when
            # Streamlit's in-memory request ID may no longer be available.
            last_user_meta = st.session_state.messages[-1].get("metadata") or {}
            recovered_id = last_user_meta.get("client_request_id")
            if recovered_id:
                st.session_state.pending_chat_request_id = recovered_id
                st.session_state.pending_chat_query = st.session_state.messages[-1]["content"]


def render_citations(citations: list[dict[str, Any]]) -> None:
    if not citations:
        return
    with st.expander(f"Sources · {len(citations)}"):
        for cite in citations:
            source_no = cite.get("source_number", "?")
            if cite.get("source_type") == "web":
                title = cite.get("title") or "Web source"
                url = cite.get("url") or ""
                st.markdown(f"**[{source_no}] [{title}]({url})**")
                st.caption("External web source · Verify the source before relying on it.")
            else:
                filename = cite.get("filename") or "Document"
                st.markdown(f"**[{source_no}] {filename}**")
                st.caption(f"Document: {cite.get('document_id', 'N/A')} · Chunk: {cite.get('chunk_id', 'N/A')}")
                supporting_chunks = cite.get("supporting_chunk_ids") or []
                if len(supporting_chunks) > 1:
                    st.caption("Additional supporting chunks: " + ", ".join(supporting_chunks[1:]))

def render_message_diagnostics(metadata: dict[str, Any]) -> None:
    """Render compact, per-message pipeline diagnostics with seconds as the display unit."""
    diagnostics = metadata.get("diagnostics") or {}
    if not diagnostics and not metadata.get("model"):
        return

    def seconds(value: Any) -> str:
        if value is None:
            return "N/A"
        try:
            return f"{float(value) / 1000.0:.2f} s"
        except (TypeError, ValueError):
            return "N/A"

    def tile(label: str, value: Any) -> str:
        return (
            '<div class="rn-metric-tile">'
            f'<div class="rn-metric-label">{html.escape(str(label))}</div>'
            f'<div class="rn-metric-value">{html.escape(str(value))}</div>'
            '</div>'
        )

    pipeline = str(metadata.get("pipeline_mode") or diagnostics.get("pipeline_mode") or "unknown").lower()
    pipeline_label = "AGENTIC" if pipeline == "agentic" else "STANDARD" if pipeline == "standard" else pipeline.upper()
    intent = diagnostics.get("intent") or metadata.get("intent")
    total_latency = metadata.get("latency_ms", diagnostics.get("total_latency_ms"))
    retrieval_latency = diagnostics.get("retrieval_latency_ms")
    generation_latency = diagnostics.get("generation_latency_ms")
    model = metadata.get("model") or "N/A"
    retrieved = metadata.get("retrieved_count", 0)
    citations = diagnostics.get("citation_count", len(metadata.get("citations") or []))
    reranking = (
        "Cached" if diagnostics.get("answer_cache_hit")
        else "Applied" if diagnostics.get("reranking_applied")
        else "Not applied"
    )

    with st.expander("Answer details", expanded=False):
        badge_class = "agentic" if pipeline == "agentic" else "standard"
        st.markdown(
            f'<span class="rn-pipeline-badge {badge_class}">{html.escape(pipeline_label)} PIPELINE</span>',
            unsafe_allow_html=True,
        )
        summary_tiles = "".join([
            tile("Total", seconds(total_latency)),
            tile("Retrieval", seconds(retrieval_latency)),
            tile("Generation", seconds(generation_latency)),
            tile("Model", model),
            tile("Retrieved chunks", retrieved),
            tile("Citations", citations),
            tile("Reranking", reranking),
            tile("Intent", intent or "knowledge"),
        ])
        st.markdown(f'<div class="rn-metrics-grid">{summary_tiles}</div>', unsafe_allow_html=True)

        stage_timings = diagnostics.get("stage_timings_ms") or {}
        if stage_timings:
            stage_values = [
                ("Routing", stage_timings.get("routing")),
                ("Retrieval", stage_timings.get("retrieval")),
                ("Grading", stage_timings.get("grading")),
                ("Rewrite", stage_timings.get("corrective_rewrite")),
                ("Web search", stage_timings.get("web_search")),
                ("Generation", stage_timings.get("generation")),
                ("Verification", stage_timings.get("verification")),
            ]
            stage_tiles = "".join(tile(label, seconds(value)) for label, value in stage_values if value is not None)
            if stage_tiles:
                st.markdown("**Agentic stage timings**")
                st.markdown(f'<div class="rn-metrics-grid">{stage_tiles}</div>', unsafe_allow_html=True)
        elif not diagnostics.get("answer_cache_hit"):
            stage_values = [
                ("Dense search", diagnostics.get("dense_latency_ms")),
                ("BM25 search", diagnostics.get("bm25_latency_ms")),
                ("RRF fusion", diagnostics.get("fusion_latency_ms")),
                ("Reranking", diagnostics.get("rerank_latency_ms")),
            ]
            st.markdown("**Retrieval stages**")
            st.markdown(
                f'<div class="rn-metrics-grid">{"".join(tile(label, seconds(value)) for label, value in stage_values)}</div>',
                unsafe_allow_html=True,
            )

        usage = (
            f"Answer-generation prompt tokens: {metadata.get('prompt_tokens') if metadata.get('prompt_tokens') is not None else 'N/A'}"
            f" · Completion: {metadata.get('completion_tokens') if metadata.get('completion_tokens') is not None else 'N/A'}"
        )
        st.caption(usage)
        cache_labels = []
        if diagnostics.get("conversation_cache_hit") is not None:
            cache_labels.append(f"Conversation cache: {'hit' if diagnostics.get('conversation_cache_hit') else 'miss'}")
        if diagnostics.get("retrieval_cache_hit") is not None:
            cache_labels.append(f"Retrieval cache: {'hit' if diagnostics.get('retrieval_cache_hit') else 'miss'}")
        if diagnostics.get("answer_cache_hit") is not None:
            cache_labels.append(f"Answer cache: {'hit' if diagnostics.get('answer_cache_hit') else 'miss'}")
        web_diag = diagnostics.get("web_search_diagnostics") or {}
        if diagnostics.get("web_search_attempted"):
            cache_labels.append(f"Tavily: {diagnostics.get('web_search_status') or web_diag.get('status') or 'attempted'}")
            if diagnostics.get("web_cache_hit") or web_diag.get("cache_hit"):
                cache_labels.append("Web cache: hit")
        if cache_labels:
            st.caption(" · ".join(cache_labels))
        if diagnostics.get("web_search_attempted"):
            web_count = diagnostics.get("web_result_count", web_diag.get("result_count", 0))
            web_latency = web_diag.get("latency_ms", diagnostics.get("web_search_latency_ms"))
            web_latency_text = seconds(web_latency) if web_latency is not None else "N/A"
            st.caption(f"Tavily results: {web_count} · Search latency: {web_latency_text}")
        elif diagnostics.get("web_search_requested"):
            route_reason = diagnostics.get("web_search_reason") or "source router selected documents"
            st.caption(f"Tavily not called · {route_reason}")

        st.caption(
            f"Scope: {diagnostics.get('document_scope', 'unknown')} · "
            f"Visibility: {diagnostics.get('visibility', 'unknown')} · "
            f"Reranker: {diagnostics.get('reranker_model') or 'not reported'} · "
            f"Candidate K: {diagnostics.get('candidate_k', 'N/A')} · "
            f"Final K: {diagnostics.get('final_k', 'N/A')}"
        )
        if diagnostics.get("web_error"):
            st.warning(f"Web search: {diagnostics.get('web_error')}")
        if diagnostics.get("historical_sources_removed"):
            st.info(
                "Some source chunks were deleted or superseded. Their citations and "
                "evidence previews are hidden; the historical answer is retained."
            )

        chunks = diagnostics.get("retrieved_chunks") or []
        if chunks:
            if diagnostics.get("route") == "web" or diagnostics.get("document_evidence_used") is False:
                evidence_label = "Retrieved document chunks · not used to generate this web-only answer"
            else:
                evidence_label = "Retrieved document evidence"
            st.markdown(f"**{evidence_label} · {len(chunks)} chunks**")
            for index, chunk in enumerate(chunks, start=1):
                filename = chunk.get("filename") or "Document"
                chunk_id = chunk.get("chunk_id") or "unknown"
                st.markdown(f"{index}. **{filename}** · chunk `{chunk.get('chunk_index', 'N/A')}`")
                st.caption(
                    f"Chunk ID: {chunk_id} · Dense: {chunk.get('dense_score') if chunk.get('dense_score') is not None else 'N/A'} · "
                    f"BM25: {chunk.get('bm25_score') if chunk.get('bm25_score') is not None else 'N/A'} · "
                    f"RRF: {chunk.get('rrf_score') if chunk.get('rrf_score') is not None else 'N/A'} · "
                    f"Rerank: {chunk.get('rerank_score') if chunk.get('rerank_score') is not None else 'N/A'}"
                )
                if chunk.get("preview"):
                    st.text(chunk["preview"])


def render_chat_page() -> None:
    title_col, context_col = st.columns([3, 2])
    with title_col:
        st.markdown('<div class="rn-section-title">Knowledge Assistant</div>', unsafe_allow_html=True)
        st.caption(f"Workspace · {workspace.get('name', 'Unknown')}")
    with context_col:
        latest_meta = next((m.get("metadata", {}) for m in reversed(st.session_state.messages) if m.get("role") == "assistant"), {})
        active_model = latest_meta.get("model") or (st.session_state.last_chat_meta or {}).get("model") or "Model selected by provider"
        st.markdown(f"<div style='text-align:right;color:#a5b4fc;font-size:.82rem'>◈ {active_model}</div>", unsafe_allow_html=True)
        st.caption(f"{len(indexed_docs)} indexed document(s) available")

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

        st.session_state.pipeline_mode = st.selectbox(
            "RAG pipeline", ["standard", "agentic"],
            index=1 if st.session_state.get("pipeline_mode", "standard") == "agentic" else 0,
            format_func=lambda value: "Agentic RAG (LangGraph)" if value == "agentic" else "Standard RAG",
            help="Standard runs the direct retrieval pipeline. Agentic runs the LangGraph routing, grading, corrective retrieval and verification workflow.",
        )
        st.session_state.enable_web_search = st.checkbox(
            "Allow web search (Agentic RAG only)", value=bool(st.session_state.get("enable_web_search", False)),
            help="External search is opt-in and requires TAVILY_API_KEY.",
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
        load_chat_history(force=bool(st.session_state.get("pending_chat_request_id")))
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

    history_area = st.container()
    with history_area:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    message_meta = message.get("metadata") or {}
                    render_citations(message_meta.get("citations") or [])
                    if (message_meta.get("diagnostics") or {}).get("citation_validation_status") == "no_supporting_source":
                        st.warning("The answer was generated, but no retrieved source could be confidently attached as a citation.")
                    render_message_diagnostics(message_meta)

    pending_request_id = st.session_state.get("pending_chat_request_id")
    if pending_request_id:
        st.info(
            "A previous request is still pending. RAGNexus is checking the saved conversation "
            "for its response. Please do not submit the same question again."
        )

    query = st.chat_input(
        "Message RAGNexus...",
        disabled=bool(pending_request_id),
    )
    if query and not pending_request_id:
        request_id = str(uuid4())
        try:
            session_id = create_session_if_needed()
            st.session_state.pending_chat_request_id = request_id
            st.session_state.pending_chat_query = query
            st.session_state.messages.append(
                {"role": "user", "content": query, "metadata": {"client_request_id": request_id}}
            )
            with history_area:
                with st.chat_message("user"):
                    st.markdown(query)
                with st.chat_message("assistant"):
                    with st.spinner("Searching your documents and preparing an answer..."):
                        selected_pipeline = st.session_state.get("pipeline_mode", "standard")
                        result = api().send_message(
                            session_id, query,
                            pipeline_mode=selected_pipeline,
                            enable_web_search=bool(st.session_state.get("enable_web_search", False)),
                            request_id=request_id,
                            timeout=chat_request_timeout(),
                        )
                    answer = result.get("answer") or "The API returned an empty answer."
                    st.markdown(answer)
                    citations = result.get("citations") or []
                    render_citations(citations)
                    response_metadata = {
                        "citations": citations,
                        "retrieved_count": result.get("retrieved_count", 0),
                        "latency_ms": result.get("latency_ms", 0),
                        "model": result.get("model"),
                        "prompt_tokens": result.get("prompt_tokens"),
                        "completion_tokens": result.get("completion_tokens"),
                        "retry_count": result.get("retry_count", 0),
                        "diagnostics": result.get("diagnostics") or {},
                        "pipeline_mode": (result.get("diagnostics") or {}).get("pipeline_mode") or selected_pipeline,
                        "client_request_id": request_id,
                    }
                    if result.get("abstained"):
                        st.info("I couldn't find sufficient supporting evidence in the selected documents.")
                    elif (result.get("diagnostics") or {}).get("citation_validation_status") == "no_supporting_source":
                        st.warning("The answer was generated, but no retrieved source could be confidently attached as a citation.")
                    render_message_diagnostics(response_metadata)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "metadata": response_metadata}
            )
            st.session_state.last_chat_meta = result
            st.session_state.pending_chat_request_id = None
            st.session_state.pending_chat_query = None
        except APIError as exc:
            if exc.timed_out:
                try:
                    load_chat_history(force=True)
                    matching_messages = [
                        message for message in st.session_state.messages
                        if (message.get("metadata") or {}).get("client_request_id") == request_id
                    ]
                    recovered = any(message["role"] == "assistant" for message in matching_messages)
                    request_was_persisted = bool(matching_messages)
                    if recovered:
                        st.success("The backend completed the request. The saved answer has been recovered.")
                        st.rerun()
                    if not request_was_persisted:
                        st.session_state.pending_chat_request_id = None
                        st.session_state.pending_chat_query = None
                        st.error(
                            "The request was not recorded by the backend. You can submit the question again."
                        )
                    else:
                        st.warning(
                            "The UI stopped waiting, but the backend may still be processing your request. "
                            "Your request ID is retained; refresh this conversation later to load the saved answer. "
                            "Do not submit the same question again."
                        )
                except APIError:
                    st.warning(
                        "The chat request timed out and the history check also failed. "
                        "Your request ID is retained so the same request is not submitted twice."
                    )
            else:
                show_api_error(exc)



def render_profile_page() -> None:
    st.markdown('<div class="rn-section-title">Your profile</div>', unsafe_allow_html=True)
    user = st.session_state.user or {}
    email = user.get("email", "Signed-in user")
    st.markdown(f"<div class='rn-panel'><div style='font-size:1.15rem;font-weight:750'>{email}</div><div class='rn-muted'>RAGNexus account</div></div>", unsafe_allow_html=True)
    st.write("")
    current = next((w for w in st.session_state.workspaces if str(w.get("id")) == str(st.session_state.workspace_id)), {})
    a, b, c = st.columns(3)
    a.metric("Workspaces", len(st.session_state.workspaces))
    b.metric("Documents in current workspace", len(documents))
    c.metric("Indexed documents", len(indexed_docs))
    st.markdown("### Current workspace")
    st.info(f"{current.get('name', 'No workspace selected')} · {current.get('role', 'member')}")
    st.markdown("### Preferences")
    st.caption("Conversation visibility and knowledge scope are selected per new conversation. Existing conversations retain their original settings.")
    st.markdown("### Security")
    st.caption("Your access token is used for authenticated API requests. Sign out to end this browser session. Password changes and session revocation are not exposed by the current backend.")
    if st.button("Sign out", type="primary"):
        clear_auth()
        st.rerun()


# ---------------------------------------------------------------------
# Route the authenticated UI
# ---------------------------------------------------------------------
if st.session_state.page == "Documents":
    render_documents_page()
elif st.session_state.page == "Profile":
    render_profile_page()
else:
    render_chat_page()
