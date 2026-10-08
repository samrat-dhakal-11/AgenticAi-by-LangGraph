from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
app = FastAPI()

# Allow the HTML page (opened from your computer) to call this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Needs your OPENAI_API_KEY environment variable
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# One prompt for each feature
PROMPTS = {
    "roadmap": ChatPromptTemplate.from_messages([
        ("system", "You are a friendly career mentor for BSc CSIT students."),
        ("human",
         "My current skills: {skills}\n"
         "My target career: {goal}\n\n"
         "Create a career roadmap with 4 milestones. For each milestone, "
         "give a title, what to learn, and a suggested timeline. Keep it short and simple."
         "answer shouldnot be in markodwn format it shouldnot consist ** and ###")
        ,
    ]),
    "projects": ChatPromptTemplate.from_messages([
        ("system", "You are a helpful mentor for BSc CSIT students."),
        ("human",
         "My skills: {skills}\n"
         "My interests: {goal}\n\n"
         "Suggest 3 beginner-friendly project ideas. For each, give the name, "
         "a short description, the tech stack, and the difficulty level."
         "answer shouldnot be in markodwn format it shouldnot consist ** and ###"),
    ]),
    "planner": ChatPromptTemplate.from_messages([
        ("system", "You are a helpful study coach for BSc CSIT students."),
        ("human",
         "My subjects: {skills}\n"
         "My goal or exam date: {goal}\n\n"
         "Make a simple weekly study plan with a day-by-day schedule."
         "answer shouldnot be in markodwn format it shouldnot consist ** and ###"),
    ]),
}


class GuidanceRequest(BaseModel):
    feature: str   # "roadmap", "projects", or "planner"
    skills: str    # skills, or subjects for the planner
    goal: str      # career goal, interests, or exam date


@app.post("/generate")
def generate(req: GuidanceRequest):
    prompt = PROMPTS.get(req.feature)
    if prompt is None:
        return {"error": "Unknown feature"}

    # prompt -> LLM -> plain text answer
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"skills": req.skills, "goal": req.goal})
    return {"result": answer}