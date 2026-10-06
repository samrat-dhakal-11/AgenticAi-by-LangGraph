from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
model=ChatOpenAI(model='gpt-4o-mini')

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_headers=['*'],
    allow_credentials=True,
    allow_methods=['*']
)

class UserMsg(BaseModel):
    user_msg:str

@app.post('/app/chat-backend')
async def chat(msg:UserMsg):
    user_msg=msg.user_msg
    
    async def generate():
        async for chunk in model.astream(user_msg):
            if chunk.content:
                yield chunk.content
      
    
    return StreamingResponse(generate(),media_type='text/plain') #type:ignore


