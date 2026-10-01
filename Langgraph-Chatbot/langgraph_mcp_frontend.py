import queue
import uuid
import streamlit as st
from langgraph_mcp_backend import chatbot, retrieve_all_threads, submit_async_task, llm
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage

# =========================== Utilities ===========================
def generate_thread_id():
    return str(uuid.uuid4())


def new_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id, 'New Chat')
    st.session_state['msg_history'] = []


def add_thread(thread_id, title='New Chat'):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
    if thread_id not in st.session_state['chat_titles'] or st.session_state['chat_titles'][thread_id] == 'New Chat':
        st.session_state['chat_titles'][thread_id] = title


def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])


def content_separator(msgs):
    if isinstance(msgs, str):
        return msgs
    elif isinstance(msgs, list) and len(msgs) > 0:
        block = msgs[0]
        if isinstance(block, dict) and 'text' in block:
            return block['text']
    return str(msgs)


def generate_smart_title(user_input: str) -> str:
    """Uses a quick LLM call to create a short, Gemini-style topic name (3-5 words)."""
    try:
        prompt = [
            SystemMessage(content="You are a helpful assistant. Summarize the user's message into a very short, punchy chat topic title (maximum 4 words). Do not use quotation marks, markdown, or punctuation."),
            HumanMessage(content=user_input)
        ]
        # Quick invoke on the lightweight model to summarize title
        response = llm.invoke(prompt)
        title = response.content.strip() #type:ignore
        # Fallback if empty or too long
        if not title or len(title) > 30:
            return user_input[:25] + "..."
        return title
    except Exception:
        return user_input[:25] + "..."


def reconstruct_chat_titles_from_history():
    """Scans all loaded threads to recover titles from the first message if missing."""
    titles = {}
    for tid in st.session_state['chat_threads']:
        try:
            state = chatbot.get_state(config={'configurable': {'thread_id': tid}})
            messages = state.values.get('messages', [])
            if messages:
                # Find the first human message to extract a title
                for msg in messages:
                    if isinstance(msg, HumanMessage):
                        txt = content_separator(msg.content)
                        titles[tid] = txt[:25] + "..." if len(txt) > 25 else txt
                        break
            if tid not in titles:
                titles[tid] = 'New Chat'
        except Exception:
            titles[tid] = f"Chat {tid[:4]}"
    return titles


# ======================= Session Initialization ===================
if 'msg_history' not in st.session_state:
    st.session_state['msg_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

# Restore chat titles dynamically from persistent database state on load/reload
if 'chat_titles' not in st.session_state:
    st.session_state['chat_titles'] = reconstruct_chat_titles_from_history()

add_thread(st.session_state['thread_id'])

# ============================ Sidebar ============================
st.sidebar.title('LangGraph MCP Chatbot')
if st.sidebar.button('New Chat'):
    new_chat()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads'][::-1]:
    title = st.session_state['chat_titles'].get(thread_id, f'Chat {thread_id[:4]}')
    if st.sidebar.button(title, key=f"btn_{thread_id}", use_container_width=True):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)
        
        temp_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                role = 'user'
                avatar = '🧑'
            else:
                role = 'assistant'
                avatar = '✨'
            content = content_separator(message.content)
            temp_messages.append({'role': role, 'content': content, 'avatar': avatar})
        
        st.session_state['msg_history'] = temp_messages

# ============================ Main UI ============================

# Render conversation history
for msg in st.session_state['msg_history']:
    with st.chat_message(msg['role'], avatar=msg['avatar']):
        st.markdown(msg['content'])

user_input = st.chat_input('Enter your query')

if user_input:
    current_thread = st.session_state['thread_id']
    current_messages = load_conversation(current_thread)
    
    # Generate smart, Gemini-style topic name on the first message of a thread
    if len(current_messages) == 0:
        smart_title = generate_smart_title(user_input)
        st.session_state['chat_titles'][current_thread] = smart_title

    CONFIG = {
        'configurable': {'thread_id': current_thread},
        'metadata': {'thread_id': current_thread},
        'run_name': 'chat_turn',
    }
    
    # Append & display user message
    st.session_state['msg_history'].append({'role': 'user', 'content': user_input, 'avatar': '🧑'})
    with st.chat_message('user', avatar='🧑'):
        st.markdown(user_input)
    
    # Assistant response streaming block with async bridge
    with st.chat_message('assistant', avatar='✨'):
        status_holder = {'box': None}
        
        def ai_only_stream():
            event_queue: queue.Queue = queue.Queue()

            async def run_stream():
                try:
                    async for message_chunk, metadata in chatbot.astream(
                        {'messages': [HumanMessage(content=user_input)]},
                        config=CONFIG, #type:ignore
                        stream_mode='messages',
                    ):
                        event_queue.put((message_chunk, metadata))
                except Exception as exc:
                    event_queue.put(("error", exc))
                finally:
                    event_queue.put(None)

            submit_async_task(run_stream())

            while True:
                item = event_queue.get()
                if item is None:
                    break
                message_chunk, metadata = item
                if message_chunk == "error":
                    raise metadata

                # Lazily create & update tool execution status
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(  #type:ignore
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream assistant content chunks
                if isinstance(message_chunk, AIMessage):
                    content = message_chunk.content
                    if content:
                        if isinstance(content, str):
                            yield content
                        elif isinstance(content, list) and len(content) > 0:
                            block = content[0]
                            if isinstance(block, dict) and 'text' in block:
                                yield block['text']

        ai_message = st.write_stream(ai_only_stream())

        # Finalize tool status UI box if active
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )
        
    st.session_state['msg_history'].append({'role': 'assistant', 'content': ai_message, 'avatar': '✨'})
    st.rerun()

# streamlit run langgraph_mcp_frontend.py 