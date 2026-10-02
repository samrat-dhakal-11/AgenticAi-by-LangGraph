import streamlit as st
import uuid
from langgraph_rag_backend import (
    chatbot,
    get_all_threads,
    ingest_pdf,
    thread_document_metadata,
)

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

#------------------------------------------------------------------------------------
# Utility functions
def generate_thread_id():
    return str(uuid.uuid4())

def new_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id)
    st.session_state['msg_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])

def extract_message_content(content):
    """Helper to cleanly extract text whether content is a string or list block."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list) and len(content) > 0:
        block = content[0]
        if isinstance(block, dict) and 'text' in block:
            return block['text']
    return str(content) if content else ""

#------------------------------------------------------------------------------------
# Session State Initialization
if 'msg_history' not in st.session_state:
    st.session_state['msg_history'] = []
    
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()
    
if 'chat_threads' not in st.session_state:
    # Pull existing thread IDs from SQLite checkpointer database on startup
    st.session_state['chat_threads'] = get_all_threads()
    
if 'chat_titles' not in st.session_state:
    st.session_state['chat_titles'] = {}

if 'ingested_docs' not in st.session_state:
    st.session_state['ingested_docs'] = {}

add_thread(st.session_state['thread_id'])
thread_key = str(st.session_state['thread_id'])
thread_docs = st.session_state['ingested_docs'].setdefault(thread_key, {})

# Auto-generate or restore meaningful titles for loaded database threads
for t_id in st.session_state['chat_threads']:
    if t_id not in st.session_state['chat_titles']:
        msgs = load_conversation(t_id)
        if msgs:
            # Find the first human message to make a smart Gemini-style title
            first_user_msg = next((m.content for m in msgs if isinstance(m, HumanMessage)), None)
            if first_user_msg:
                txt = extract_message_content(first_user_msg)
                st.session_state['chat_titles'][t_id] = txt[:28] + "..." if len(txt) > 28 else txt
            else:
                st.session_state['chat_titles'][t_id] = f"Chat {t_id[:4]}"
        else:
            st.session_state['chat_titles'][t_id] = "New Chat"

#------------------------------------------------------------------------------------
# Sidebar Layout
st.sidebar.title('LangGraph PDF Chatbot')
st.sidebar.markdown(f"**Thread ID:** `{thread_key[:8]}...`")

if st.sidebar.button('New Chat', use_container_width=True):
    new_chat()
    st.rerun()

st.sidebar.divider()

# PDF Ingestion Widget (Professor Code Integration)
if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"📄 `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks)"
    )
else:
    st.sidebar.info("No PDF indexed in this chat yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        pass
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)

st.sidebar.divider()
st.sidebar.header('My Conversations')

selected_thread = None
for thread_id in st.session_state['chat_threads'][::-1]:
    title = st.session_state['chat_titles'].get(thread_id, f"Chat {thread_id[:4]}")
    if st.sidebar.button(title, key=f"btn_{thread_id}", use_container_width=True):
        selected_thread = thread_id

#------------------------------------------------------------------------------------
# Main Application Layout
st.title("Samrat Dhakal RAG Chatbot")

# Loading the conversation history into view
for msg in st.session_state['msg_history']:
    with st.chat_message(msg['role'], avatar=msg.get('avatar')):
        st.markdown(msg['content'])

user_input = st.chat_input('Ask about your document or use tools...')

if user_input:
    current_thread = st.session_state['thread_id']
    current_messages = load_conversation(current_thread)
    
    if len(current_messages) == 0:
        clean_title = user_input[:28] + "..." if len(user_input) > 28 else user_input
        st.session_state['chat_titles'][current_thread] = clean_title

    CONFIG = {
        'configurable': {'thread_id': current_thread},
        'metadata': {'thread_id': current_thread},
        'run_name': 'chat_turn',
    }
    
    # Adding user message to state history & UI
    st.session_state['msg_history'].append({'role': 'user', 'content': user_input, 'avatar': '🧑'})
    with st.chat_message('user', avatar='🧑'):
        st.markdown(user_input)
    
    # Generating response with tool status indicator & stream output
    with st.chat_message('assistant', avatar='✨'):
        status_holder = {'box': None}
        
        def generate_chat_response():
            stream_data = chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=CONFIG, #type:ignore
                stream_mode='messages'
            )
            for message_chunk, metadata in stream_data:
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status( #type:ignore
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …", state="running", expanded=True
                        )

                if isinstance(message_chunk, AIMessage):
                    txt_content = extract_message_content(message_chunk.content)
                    if txt_content:
                        yield txt_content
                        
        ai_message = st.write_stream(generate_chat_response())
        
        if status_holder["box"] is not None:
            status_holder["box"].update(label="✅ Tool finished", state="complete", expanded=False)
            
    st.session_state['msg_history'].append({'role': 'assistant', 'content': ai_message, 'avatar': '✨'})
    st.rerun()

#------------------------------------------------------------------------------------
# Handle thread switching from sidebar clicks
if selected_thread and selected_thread != st.session_state['thread_id']:
    st.session_state['thread_id'] = selected_thread
    messages = load_conversation(selected_thread)
    
    temp_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            temp_messages.append({'role': 'user', 'content': extract_message_content(msg.content), 'avatar': '🧑'})
        elif isinstance(msg, AIMessage):
            temp_messages.append({'role': 'assistant', 'content': extract_message_content(msg.content), 'avatar': '✨'})
            
    st.session_state['msg_history'] = temp_messages
    st.session_state['ingested_docs'].setdefault(str(selected_thread), {})
    st.rerun()

# streamlit run langgraph_rag_frontend.py