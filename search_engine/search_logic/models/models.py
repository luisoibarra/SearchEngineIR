from pydantic import BaseModel
from typing import List

class Document(BaseModel):
    title:str

class QueryResult(BaseModel):
    documents: List[Document]
    