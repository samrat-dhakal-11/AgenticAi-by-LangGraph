import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import requests
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic import hub 
from dotenv import load_dotenv
from langchain_tavily import TavilySearch

load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "React Agent"
tv_api=os.getenv("TAVILY_API_KEY")

# Initialize the Tavily search tool
search_tool = TavilySearch(max_results=3,api_key=tv_api)

@tool
def get_weather_data(city: str) -> str:

  """
  This function fetches the current weather data for a given city
  """
  try:
    api_key=os.getenv('WEATHERSTACK_API_KEY')
    
  except:
      print("Load Fail of weather Stack api Key. \nExiting.....")
      exit()
  url = f'https://api.weatherstack.com/current?access_key={api_key}&query={city}'

  response = requests.get(url)

  return response.json()

llm = ChatOpenAI(model='gpt-4o-mini')

# Step 2: Pull the ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")  # pulls the standard ReAct agent prompt

# Step 3: Create the ReAct agent manually with the pulled prompt
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

# Step 4: Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True,
    max_iterations=5
)

# What is the release date of Dhadak 2?
# What is the current temp of bharatpur-chitwan-nepal
# Identify the birthplace city of kalpana chawla (search) and give its current temperature.

# Step 5: Invoke
response = agent_executor.invoke({"input": "Identify the birthplace city of kalpana chawla (search) and give its current temperature."})
print(response)
print('\n\n')
print(response['output'])