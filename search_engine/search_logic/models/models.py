from pydantic import BaseModel
from typing import List

class Document(BaseModel):
    documentName:str

class QueryResult(BaseModel):
    documents: List[Document]
    responseTime: int = 0
    