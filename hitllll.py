# practice hitl by langgraph
import os
import requests
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.types import interrupt, Command
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model='gpt-4o-mini')

# Initialize search tool
search_tool = TavilySearch(
    api_key=os.getenv('TAVILY_API_KEY'),
    max_results=3
)

@tool
def get_stock_info(symbol: str) -> dict:
    """
    Receives info about the company or any institute in the symbol (e.g. 'AAPL', 'TSLA') 
    and fetches the stock return back the 
    stock details of the company or any institute
    """
    api_key = os.getenv('ALPHAVANTAGE_API_KEY')
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    )
    response = requests.get(url)
    return response.json()

@tool
def purchase_stock(symbol: str, amount: int) -> dict:
    """
    Receives the symbol of company or any institute (e.g. 'AAPL', 'TSLA') and
    no of shares to purchase 
    - HITL feature here decides whether to buy or reject the response
    """
    response = interrupt(f"Are you sure to buy {amount} shares of the {symbol}? y/n")
    if isinstance(response, str) and response.strip().lower() in ['y', 'yes', 'sure', 'whynot']:
        return {
            'status': 'success',
            'info': f"The {amount} share of {symbol} has been purchased successfully",
            'symbol': symbol,
            'quantity': amount
        }
    else:
        return {
            'status': 'negative',
            'info': f"The {amount} share of {symbol} purchase is canceled by the user",
            'symbol': symbol,
            'quantity': amount
        }

tools = [get_stock_info, purchase_stock,search_tool]
llm_with_tools = llm.bind_tools(tools)

class HitlState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: HitlState):
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {'messages': response}

tool_node = ToolNode(tools)
checkpointer = MemorySaver()

graph = StateGraph(HitlState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    print("Write exit or quit to end chat")
    while True:
        config = {'configurable': {'thread_id': 'thread_123'}}
        user_input = input("You: ")
        if user_input.lower().strip() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # Build initial state for this turn
        state = {"messages": [HumanMessage(content=user_input)]}

        # Run the graph (may hit an interrupt)
        result = chatbot.invoke(
            state,  # type:ignore
            config=config  # type:ignore
        )
        
        interrupts = result.get("__interrupt__", [])
        if interrupts:
            # Our interrupt payload is the string we passed to interrupt(...)
            prompt_to_human = interrupts[0].value
            print(f'HITL: {prompt_to_human}')
            decision = input("Your decision: ").strip().lower()
            
            # Resume graph with the human decision ("yes" / "no" / whatever)
            result = chatbot.invoke(
                Command(resume=decision),
                config=config  # type:ignore
            )
            
        # Get the latest message from the assistant
        messages = result["messages"]
        last_msg = messages[-1]
        print(f"Bot: {last_msg.content}\n")
