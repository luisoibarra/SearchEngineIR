from typing import List
from .models.models import Document, FeedbackModel, QueryResult
from .ir_models.vectorial import VectorialModel

import os
import time as t
dirname, filename = os.path.split(os.path.abspath(__file__))
CORPUS = os.path.join(dirname, "corpus")

model = VectorialModel(CORPUS)

model.build()

def get_documents(query: str) -> QueryResult:
    s = t.time()
    values = model.resolve_query(query)
    e = t.time()
    return QueryResult(
        documents = [
            # Document(
            #     documentName=os.path.split(x["dir"])[-1],
            #     documentDir=x["dir"]
            # ) for _,x in values
            Document(documentName="Doc1", documentDir="Someplace"),
            Document(documentName="Doc2", documentDir="Someplace"),
        ],
        responseTime=int((e - s) * 1000),
        query=query,
        queryExpansions=[f"{query} Expansion TODO1", f"{query} Expansion TODO2"] # TODO
    )

def get_document_content(document_dir: str) -> str:
    doc = [doc["text"] for doc in model.build_result["documents"] if doc["dir"] == document_dir]
    if not doc:
        raise Exception(f"{document_dir} wasn't found")
    return doc[0]

def apply_feedback_to_model(feedback: FeedbackModel):
    relevant = [doc for doc in model.build_result["documents"] if doc["dir"] in feedback.relevants]
    not_relevant = [doc for doc in model.build_result["documents"] if doc["dir"] in feedback.not_relevants]
    model.add_relevant_and_not_relevant_documents(feedback.query, relevant, not_relevant)