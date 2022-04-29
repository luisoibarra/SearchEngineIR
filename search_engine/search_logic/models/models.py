from pydantic import BaseModel
from typing import List

class Document(BaseModel):
    documentName:str
    documentDir:str

class QueryResult(BaseModel):
    documents: List[Document]
    """
    Query response in milliseconds
    """
    responseTime: int = 0
    query:str
    queryExpansions: List[str]

class FeedbackModel(BaseModel):
    query:str
    relevants: List[str]
    not_relevants: List[str]