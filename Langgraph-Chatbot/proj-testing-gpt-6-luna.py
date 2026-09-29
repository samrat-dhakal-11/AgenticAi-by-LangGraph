import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
os.environ['LANGCHAIN_PROJECT'] = 'proj-testing-apikey'

model=ChatOpenAI(model='gpt-4o-mini')

response=model.invoke('What is ai?')

print("\n\n",response)
print('\n\nResponse rom gpt-4o-mini:\n',response.content)
