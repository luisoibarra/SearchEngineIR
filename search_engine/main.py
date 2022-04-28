from search_logic.models.models import QueryResult
from fastapi import FastAPI
from search_logic import get_documents

app = FastAPI()


@app.get("/query")
async def get_query_result(query:str, offset:int) -> QueryResult:
    """
    Returns the ranked documents associated with the `query` skipping `offset`
    """
    return get_documents(query)

@app.get("/document")
async def get_query_result(document_id:str) -> str:
    """
    Returns document's content associated with the given `document_id`
    """
    raise NotImplementedError()
