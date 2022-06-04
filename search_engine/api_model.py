from typing import List
from models.models import Document, FeedbackModel, QueryResult
from search_logic.ir_models.vectorial import VectorialModel
# from nltk.corpus import wordnet
from pathlib import Path

import os
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

    # TODO Remove commented code below
    # PRECISION, RECALL and F1 MEASURES are computed only when testing. 

    # docs= [doc for _,doc in values]
    # def similarity(word,topics,threshold=0.5):
    #     """
    #     Check similarity using wu-palmer formula between word and any of the topic of the list of topics
    #     """
    #     for topic in topics:
    #         syn1 = wordnet.synsets(word)[0]
    #         syn2 = wordnet.synsets(topic)[0]
    #         if syn1.wup_similarity(syn2) > threshold:
    #             return True

    # # Check if a word of the query is similar to docs topic words
    # relevants_docs_recover = [doc for doc in docs if any(similarity(word,doc["topic"]) for word in query.split())]
    # non_relevants_docs_recover = [doc for doc in docs if not any(word in doc["topic"] for word in query.split())]
    
    # relevant_docs_dirs = set([doc["root"] for doc in relevants_docs_recover])
    # non_relevant_docs_dirs = set([doc["root"] for doc in non_relevants_docs_recover])

    # relevant_count=1
    # for dir in relevant_docs_dirs:
    #     relevant_count+= len(os.listdir(dir))
    
    # non_relevant_count=1
    # for dir in non_relevant_docs_dirs:
    #     non_relevant_count+= len(os.listdir(dir))
    
    
    return QueryResult(
        documents = [Document(documentName=Path(doc["dir"]).name, documentDir=doc["dir"], documentTopic=doc["topic"]) for _,doc in values],
        responseTime=int((e - s) * 1000),
        # precision=len(relevants_docs_recover)/(len(relevants_docs_recover)+len(non_relevants_docs_recover)),
        # recall=len(relevants_docs_recover)/relevant_count,
        # f1=2*len(relevants_docs_recover)/(len(relevants_docs_recover)+relevant_count),
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

def get_query_expansions(query: str) -> List[str]:
    return model.get_expansion_query(query)
