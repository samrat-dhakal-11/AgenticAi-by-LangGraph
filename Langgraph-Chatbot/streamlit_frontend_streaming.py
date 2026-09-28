import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

#st.session_state ->dict 
if 'msg_history' not in st.session_state:
    st.session_state['msg_history']=[]

#loading the conversation history
for msg in st.session_state['msg_history']:
    with st.chat_message(msg['role'],avatar=msg['avatar']):
        st.text(msg['content'])
    
        
user_input=st.chat_input('Enter your query')
if user_input:
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
                    config={'configurable':{'thread_id':'thread_1'}},
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
           
# streamlit run streamlit_frontend_streaming.py