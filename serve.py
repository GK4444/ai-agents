import os
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
# Add this import at the top
from langserve import CustomUserType

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


# Define your input schema as a Pydantic model
class TranslatorInput(CustomUserType):
    language: str
    text: str

# load model
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# prompt
system_prompt = "Convert following text to {language}. Don't add other information."
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{text}")
])

# output parser
parser = StrOutputParser()

# chain all steps
chain = (
    {"language": lambda x: x.language, "text": lambda x: x.text}
    | prompt
    | model
    | parser
)

# app definition
app = FastAPI(
    title="Langchain FastAPI",
    version="0.1",
    description="Simple FastAPI for Langchain language translator using Langserve."
)

# Update add_routes with input schema
add_routes(
    app,
    chain,
    path="/chain",
    input_type=TranslatorInput  # Explicit input type
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="localhost",
        port="8000"
    )
    # running at http://localhost:8000/chain/playground/
