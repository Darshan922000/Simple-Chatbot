from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes
from langchain_ollama.llms import OllamaLLM

import uvicorn
import os
from langchain_community.llms import ollama

from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app = FastAPI(
    title="langchain Server",
    version="1.0.0",
    description="Simple API server"
)

add_routes(
    app,
    ChatOpenAI(),
    path="/openai"
)

openai_llm = ChatOpenAI()

#ollama llms
ollama_llm = OllamaLLM(model="llama3.2:3b")

prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")

prompt2 = ChatPromptTemplate.from_template("Write me a poem about {topic} with 10 phrases")

add_routes(
    app,
    prompt1|openai_llm,
    path="/essay"
)

add_routes(
    app,
    prompt2|ollama_llm,
    path="/poem"
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)