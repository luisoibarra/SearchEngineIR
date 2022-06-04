from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.append(str((Path(__file__) / ".." / "..").resolve()))

import pytest as pt
from api_model import get_document_content, get_documents, get_query_expansions, apply_feedback_to_model
from models.models import FeedbackModel

base_corpus_path = (Path(__file__) / ".." / "cranfield_corpus").resolve()

@pt.mark.parametrize("feedback",
[
    FeedbackModel(query= "", relevants=[], not_relevants=[]),
    FeedbackModel(
        query= "air", 
        relevants=[
            base_corpus_path/"1.txt",
        ], 
        not_relevants=[]),
    FeedbackModel(
        query= "air", 
        relevants=[
        ], 
        not_relevants=[
            base_corpus_path/"1.txt",
        ]),
    FeedbackModel(
        query= "air", 
        relevants=[
            base_corpus_path/"2.txt",
        ], 
        not_relevants=[
            base_corpus_path/"1.txt",
        ]),
])
def test_apply_feedback_to_model(feedback: FeedbackModel):
    apply_feedback_to_model(feedback)

@pt.mark.parametrize("query",
[
    "",
    "air",
])
def test_get_query_expansions(query:str):
    expansion = get_query_expansions(query)
    assert expansion, f"Expansion empty wit query [{query}]"

@pt.mark.parametrize(["query", "offset", "batch_size"],
[
    ("", 0, 10),
    ("air", 0, 10),
])
def test_get_documents(query: str, offset:int, batch_size:int):
    result = get_documents(query, offset, batch_size)
    result2 = get_documents(query, offset+1, batch_size)
    if query:
        assert result.documents, "No documents returned"
        assert len(result.documents) <= batch_size, "Batch size is larger than the requested size"
        assert len(result.documents) - len(result2.documents) == 1, "Offset doesn't work properly"
    else:
        assert len(result.documents) == 0, "Returning information with empty query"

def test_get_document_content():
    pass
