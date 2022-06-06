from typing import List
from models.models import Document, FeedbackModel, QueryResult, ResponseModel
from search_logic.ir_models.vectorial import VectorialModel
# from nltk.corpus import wordnet
from pathlib import Path

import time as t

# path = Path(__file__) / ".." / "search_logic" / "corpus"
path = Path(__file__) / ".." / "test" / "cranfield_corpus"
CORPUS = path.resolve()

model = VectorialModel(CORPUS)

model.build()

print("Model built successfully")

def get_documents(query: str, offset:int, batch_size: int=15) -> QueryResult:
    s = t.time()
    values = model.resolve_query(query)[offset:offset+batch_size]
    e = t.time()

    return ResponseModel(
         documents = [Document(documentName=Path(doc["dir"]).name, documentDir=doc["dir"], documentTopic=doc["topic"]) for _,doc in values],
        responseTime=int((e - s) * 1000)
    )
    
    # return QueryResult(
    #     documents = [Document(documentName=Path(doc["dir"]).name, documentDir=doc["dir"], documentTopic=doc["topic"]) for _,doc in values],
    #     responseTime=int((e - s) * 1000),
    #     query=query,
    #     queryExpansions=[f"{query} Expansion TODO1", f"{query} Expansion TODO2"] # TODO
    # )

def get_document_content(document_dir: str) -> str:
    doc = [doc["text"] for doc in model.build_result["documents"] if doc["dir"] == document_dir]
    if not doc:
        raise Exception(f"{document_dir} wasn't found")
    return doc[0]

def apply_feedback_to_model(feedback: FeedbackModel):
    relevant = [doc for doc in model.build_result["documents"] if doc["dir"] in feedback.relevants]
    not_relevant = [doc for doc in model.build_result["documents"] if doc["dir"] in feedback.not_relevants]
    model.add_relevant_and_not_relevant_documents(feedback.query, relevant, not_relevant)

def get_query_expansions(query: str) -> List[str]:
    return model.get_expansion_query(query)
