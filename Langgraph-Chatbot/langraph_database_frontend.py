import streamlit as st
import uuid
from langgraph_database_backend import chatbot,get_all_threads
from langchain_core.messages import HumanMessage

#utility functions

def generate_thread_id():
    thread_id=str(uuid.uuid4())
    return thread_id

#new chat
def new_chat():
    thread_id=generate_thread_id()
    st.session_state['thread_id']=thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['msg_history']=[]
    
def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
        if thread_id not in st.session_state['chat_titles']:
            st.session_state['chat_titles'][thread_id]='New Chat'

def load_conversation(thread_id):
    state=chatbot.get_state(config={'configurable':{'thread_id':thread_id}})
    return state.values.get('messages',[])


#st.session_state ->dict 
if 'msg_history' not in st.session_state:
    st.session_state['msg_history']=[]
    
if 'thread_id' not in st.session_state:
    st.session_state['thread_id']=generate_thread_id()
    
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads']=get_all_threads()
    
if 'chat_titles' not in st.session_state:
    st.session_state['chat_titles'] = {}




add_thread(st.session_state['thread_id'])


st.sidebar.title('LangGraph Chatbot')
if st.sidebar.button('New Chat'):
    new_chat()
st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads'][::-1]:
    title=st.session_state['chat_titles'].get(thread_id,f'Chat{thread_id[:4]}')
    if st.sidebar.button(title,key=f"btn_{thread_id}",use_container_width=True):
        st.session_state['thread_id']=thread_id

        messages=load_conversation(thread_id)
        
        temp_messages=[]
        
        for message in messages:
            
            def content_seperator(msgs):
                if isinstance(msgs,str):
                    return msgs
                elif isinstance(msgs, list) and len(msgs) > 0:
                    block = msgs[0]
                    if isinstance(block, dict) and 'text' in block:
                        return  block['text']
                    
            
            if isinstance(message,HumanMessage):
                role='user'
                avatar='🧑'
                content=content_seperator(message.content)
            else:
                role='assistant'
                avatar='✨'
                content=content_seperator(message.content)
            temp_messages.append({'role':role,'content':content,'avatar':avatar}) 
        
        st.session_state['msg_history']=temp_messages
        

#loading the conversation history
for msg in st.session_state['msg_history']:
    with st.chat_message(msg['role'],avatar=msg['avatar']):
        st.markdown(msg['content'])
    

user_input=st.chat_input('Enter your query')
if user_input:
    current_thread = st.session_state['thread_id']
    current_messages = load_conversation(current_thread)
    if len(current_messages) == 0:
        # Feature: Truncate user query to ~28 characters for a clean sidebar title
        clean_title = user_input[:28] + "..." if len(user_input) > 28 else user_input
        st.session_state['chat_titles'][current_thread] = clean_title

    CONFIG={
        'configurable':{'thread_id':st.session_state['thread_id']},
        'metadata':{ 
            'thread_id':st.session_state['thread_id']
                },
        'run_name':'chat_turn',
        }
    
    #adding that  user msg to msg_history
    st.session_state['msg_history'].append({'role':'user','content':user_input,'avatar':'🧑'})
    with st.chat_message('user',avatar='🧑'):
        st.text(user_input)
    
    #adding that  assistant msg to msg_history
    with st.spinner("Generating Response..."):      
        with st.chat_message('assistant', avatar='✨'):
            def generate_chat_response():
                stream_data=chatbot.stream(
                    {'messages':[HumanMessage(content=user_input)]},
                    config=CONFIG, #type:ignore
                    stream_mode='messages' 
                    )
                for message_chunk,metadata in stream_data:
                    if message_chunk.content: #type:ignore
                        if isinstance(message_chunk.content, str): #type:ignore
                            yield message_chunk.content #type:ignore
                        elif isinstance(message_chunk.content, list) and len(message_chunk.content) > 0: #type:ignore
                            block = message_chunk.content[0] #type:ignore
                            if isinstance(block, dict) and 'text' in block:
                                yield block['text']
                                    
        ai_message=st.write_stream(generate_chat_response())
        
        st.session_state['msg_history'].append({'role':'assistant','content':ai_message,'avatar':'✨'})  
        st.rerun()    
           
           
           
# streamlit run langraph_database_frontend.py