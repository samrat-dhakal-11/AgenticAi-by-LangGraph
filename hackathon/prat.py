import os
import base64
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel

# LangGraph & Tools
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools.tavily_search import TavilyAnswer
from langchain_google_community import GmailToolkit
from langchain_google_community.gmail.utils import build_resource_service

load_dotenv()

#--------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_headers=['*'],
    allow_methods=['*']
)
#--------------------------------

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.7)
embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')

# Setup Tools: Tavily Search and Gmail Toolkit
tavily_tool = TavilyAnswer(max_results=5, name="tavily_search") #type:ignore

try:
    gmail_toolkit = GmailToolkit()
    gmail_tools = gmail_toolkit.get_tools()
except Exception:
    gmail_tools = []

tools = [tavily_tool] + gmail_tools
llm_with_tools = llm.bind_tools(tools)

#--------------------------------
class Storeval():
    def __init__(self):
        self.content = ''
        self.filename = ''
    
    def uploadfile(self, content, filename):
        self.content = content
        self.filename = filename
    
    def getfile(self):
        return self.content, self.filename
    
vals = Storeval()
#--------------------------------

@app.post('/app/upload')
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        filename = file.filename
        vals.uploadfile(contents, filename)
        
        return {
            'status': 'uplaod sucessful',
            'filename': filename
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'{e}')

#--------------------------------

@app.post('/app/response')
async def giveresponse():
    try:
        info, name = vals.getfile()
        if not info:
            return {'reply': 'No file uploaded yet'}
        
        is_image = name.lower().endswith(('.png', '.jpeg', '.jpg', '.webp', '.gif'))
        
        if is_image:
            base64_image = base64.b64encode(info).decode('utf-8') #type:ignore
            
            mine_type = 'image/jpeg'
            if name.lower().endswith('.png'):
                mine_type = 'image/png'
            elif name.lower().endswith('.gif'):
                mine_type = 'image/gif'

            message = HumanMessage(
                content=[
                    {
                        'type': 'text',
                        'text': 'Describe this image clearly and perfectly. Do not use ** and ###, and keep it well-mannered.'
                    },
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:{mine_type};base64,{base64_image}'
                        }
                    }
                ]
            )
            response = llm.invoke([message])
            content = response.content
        else: 
            try:
                decode_info = info.decode('utf-8') #type:ignore
            except Exception:
                decode_info = '[Binary File Data Attached]'
                
            prompt = PromptTemplate(
                template="""
                You are an expert Document Analyst in Nepal.
                The contents of {name} is-> \n {content}:
                Important:
                    - Describe the document clearly and perfectly 
                    - The description should not contain ** and ###
                    - It should be well-mannered
                """,
                input_variables=['name', 'content']
            )
            chain = prompt | llm | StrOutputParser()
            content = chain.invoke({'name': name, 'content': decode_info})
    
        return {   
            'reply': f'The detail of {name} is {content}'
        }
        
    except Exception as e:
        return {
            'reply': f'error ={e}'
        }
        
#--------------------------------
graph = StateGraph(MessagesState)

def chatbro(state: MessagesState):
    system_prompt = SystemMessage(content="""
    You are an Expert AI assistant specialized in Nepal administration and government coordination. 
    Please provide responses in a well-mannered way.
    Important restrictions:
    - The response must NOT contain markdown symbols like ** and ###.
    - Use the built-in official Nepal Government email directory tool/knowledge base whenever an email needs to be sent or checked:
      * Office of the Prime Minister and Council of Ministers (OPMCM): info@opmcm.gov.np
      * Office of the President: mail@president.gov.np or info.officer@president.gov.np
      * Ministry of Home Affairs: control@moha.gov.np or gunaso@moha.gov.np
      * Ministry of Finance: moev@mof.gov.np
      * Ministry of Foreign Affairs: info@mofa.gov.np
      * Ministry of Health and Population: info@mohp.gov.np
      * Ministry of Education, Science and Technology: info@moest.gov.np
      * Ministry of Physical Infrastructure and Transportation: info@mopit.gov.np
      * Department of Roads (DoR): dgdor@dor.gov.np (Planning: planning@dor.gov.np, Maintenance: dormaintb@dor.gov.np)
      * Department of Transport Management (DoTM): info@dotm.gov.np (Grievances: gunaso@dotm.gov.np)
      * Roads Board Nepal: info@rbn.org.np
      * Election Commission Nepal: info@election.gov.np
      * Federal Parliament Secretariat: info@parliament.gov.np
      * Hello Sarkar Complaint System: 1111@nepal.gov.np
    - If a specific body is requested that is not listed, use tavily_search to find it.
    - Once the gmail tool succeeds, report back clearly: "Email sent successfully".
    """)

    prompt = [system_prompt] + state["messages"]
    response = llm_with_tools.invoke(prompt)
    return {"messages": [response]}

graph.add_node('chat', chatbro)
graph.add_node('tools', ToolNode(tools))

graph.add_edge(START, 'chat')
graph.add_conditional_edges(
    'chat',
    tools_condition,
)
graph.add_edge('tools', 'chat')

chatbot = graph.compile(
    checkpointer=InMemorySaver()
)

#--------------------------------
class Query(BaseModel):
    UserQuery: str
    thread_id: str = '1234'

@app.post('/app/chat')
async def chat(msg: Query):
    userInput = msg.UserQuery
    
    async def generate():
        async for chunk, _metadata in chatbot.astream(
            {'messages': [HumanMessage(content=userInput)]},
            config={'configurable': {'thread_id': msg.thread_id}},
            stream_mode='messages'
        ):
            if isinstance(chunk, AIMessage) and chunk.content:
                yield chunk.content
            
    return StreamingResponse(generate(), media_type='text/plain') #type:ignore
