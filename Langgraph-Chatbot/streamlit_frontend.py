import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

CONFIG={'configurable':{'thread_id':'thread_1'}}
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
        response=chatbot.invoke({'messages':[HumanMessage(content=user_input)]},config=CONFIG) #type:ignore
        ai_message=response['messages'][-1].content[0]['text']
        st.session_state['msg_history'].append({'role':'assistant','content':ai_message,'avatar':'✨'})
        with st.chat_message('assistant',avatar='✨'):
            st.text(ai_message)
    
# streamlit run streamlit_frontend.py
