import os
import asyncio
from langgraph.graph import StateGraph, START
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()  # Load environment variables from .env file

llm = ChatOpenAI(model="gpt-4o-mini")
#mcp client 

server_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp-math-server.py")

client=MultiServerMCPClient(
    {
        'arith':{
            'transport':'stdio',
            'command':'python3',
            'args':[server_path]
        },
        'expense':{
            'transport':'streamable_http',# if this fails ,try "sse"
            'url':'https://splendid-gold-dingo.fastmcp.app/mcp'
        }
    }
)

# state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

#build graph

async def build_graph():
    
    tools=await client.get_tools()
    llm_with_tools=llm.bind_tools(tools)
    
    # nodes
    async def chat_node(state: ChatState):

        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {'messages': [response]}

    tool_node = ToolNode(tools)
    
    
    
    # defining graph and nodes
    graph = StateGraph(ChatState)

    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    # defining graph connections
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    chatbot = graph.compile()

    return chatbot
    
    
async def main():
    chatbot= await build_graph()
    # running the graph
    result =  await chatbot.ainvoke({"messages": [HumanMessage(content="Give me all my expense of november from 1st to 30th November in 2025")]}) 

    print(result['messages'][-1].content)

if __name__=='__main__':
    asyncio.run(main())