import os
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

llm=ChatOpenAI(model='gpt-4o-mini',temperature=0.9)
embedding_model=OpenAIEmbeddings(model='text-embedding-3-small')



embeding=embedding_model.embed_query('What is the capital of nepal')

print(embeding[:10])

chat=llm.invoke(f'what is this meaninig{embeding}')

print(chat.content)

print('/n/n',chat)