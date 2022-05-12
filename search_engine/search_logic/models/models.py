from pydantic import BaseModel
from typing import List

class Document(BaseModel):
    documentName:str

class QueryResult(BaseModel):
    documents: List[Document]
    responseTime: int = 0
    precision: float = 0
    recall: float = 0
    f1: float = 0
    topic: str = ""

    