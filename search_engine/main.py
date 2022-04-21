from search_logic.models.models import QueryResult
from fastapi import FastAPI
from search_logic import get_documents

app = FastAPI()


@app.get("/query")
async def get_query_result(query:str, offset:int) -> QueryResult:
    return get_documents(query)

@app.get("/")
async def get_query_result() -> QueryResult:
    return get_documents("query") 
