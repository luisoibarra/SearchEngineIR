from search_logic.models.models import FeedbackModel, QueryResult
from fastapi import FastAPI
from search_logic import  get_documents, get_document_content, apply_feedback_to_model
import uvicorn

app = FastAPI()


@app.get("/query")
async def get_query_result(query:str, offset:int) -> QueryResult:
    """
    Returns the ranked documents associated with the `query` skipping `offset`
    """

    return get_documents(query,offset=1)


@app.get("/document")
async def get_query_result(document_dir:str) -> str:
    """
    Returns document's content associated with the given `document_dir`
    """
    # return get_document_content(document_dir)
    return "asdasd"

@app.post("/feedback")
async def apply_feedback(feedback: FeedbackModel):
    """
    Apply the feedback to the model
    """
    apply_feedback_to_model(feedback)

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
