import os
from fastapi import FastAPI

from models import LLMResponse
from llms import MistralAIRetriever, ChromaConnector

app = FastAPI()

mistral_api_key = os.getenv("MISTRAL_API_KEY")
db_port = os.getenv("DB_PORT")
db_host = os.getenv("DB_HOST")
embbeding_model_path = os.getenv("EMBEDDING_MODEL_PATH")
embbeding_model_name = os.getenv("EMBEDDING_MODEL_NAME")
inference_model_name = os.getenv("INFERECE_MODEL")

db_connector = ChromaConnector(
    embedding_model_name=embbeding_model_name,
    embedding_model_folder=embbeding_model_path,
    db_host=db_host,
    db_port=int(db_port),
)

llm = MistralAIRetriever(
    api_key=mistral_api_key,
    vector_db=db_connector.init_vector_db(),
    model_id=inference_model_name,
)

@app.get("/")
async def home():
    return "Hello from Loka"

@app.post("/inference_endpoint")
async def query(message: str) -> LLMResponse:
    """
    Endpoint responsable of triggering the LLM and generate the
    answer for the user query.
    """
    answer, docs = llm.answer(question=message)
    return LLMResponse(docs=docs, answer=answer)