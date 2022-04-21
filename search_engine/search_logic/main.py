from typing import List
from .models.models import Document, QueryResult
from .ir_models.vectorial import VectorialModel

import os

dirname, filename = os.path.split(os.path.abspath(__file__))
CORPUS = os.path.join(dirname, "corpus")

model = VectorialModel(CORPUS)

model.build()

def get_documents(query: str) -> QueryResult:
    values = model.resolve_query(query)
    return QueryResult(documents = [
        Document(documentName=os.path.split(x["dir"])[-1]) for _,x in values
    ])