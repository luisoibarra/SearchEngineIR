from math import log
from typing import List, Tuple
import numpy as np
from ..pipes.pipeline import Pipe, Pipeline
from .base import *


def calculate_idf(context: dict):
    """
    Calculate the inverse document frequency in `documents` key storing 
    the results in `idf` key
    """
    idf = {}
    documents = context["documents"]
    matrix = context["term_matrix"]
    
    N = len(documents)
    for term in matrix.all_terms:
        ni = len([i for i in range(len(documents)) if matrix[term, i] > 0])
        idf[term] = log(N/ni)
    
    context["idf"] = idf
    
    return context

def convert_doc_to_vec(context: dict, is_query=False):
    """
    Convert the `documents` key in vectors representing
    the documents storing the results in `vector` key. If `idf` key is available
    the normalize the weight accordingly
    """
    documents = context["documents"] if not is_query else [context["query"]]
    matrix = context["term_matrix"]
    idf = context.get("idf")

    for i in range(len(documents)):
        current_vector = np.array([0.0 for _ in range(len(matrix.all_terms))])
        
        if is_query:
            query_fa = {y:len([x for x in documents[0]["tokens"] if x.lower() == y]) for y in set(documents[0]["tokens"])}
        
        maxim = max(matrix[term,i] for term in matrix.all_terms) if not is_query else max(query_fa.values())
        for j,term in enumerate(matrix.all_terms):
            fa = matrix[term, i] if not is_query else query_fa.get(term, 0)
            
            # idf associated with term
            idf_val = idf[term] if idf else 1
            
            # Vector representation with normalized absolute frequency and idf
            current_vector[j] = fa/maxim * idf_val 
        
        documents[i]["vector"] = current_vector

    return context

def convert_query_to_vec(context: dict) -> dict:
    return convert_doc_to_vec(context, True)

def smooth_query_vec(context: dict):
    """
    Smooth calculated query vector in `query` by some constant
    if any in `smooth_query_alpha`, defualts to 0.4.
    
    alpha*idf_i + (1-alpha)ntf_{iq} idf_i
    """
    query = context["query"]
    alpha = context.get("smooth_query_alpha", 0.4)
    idf = context.get("idf")
    matrix = context["term_matrix"]
    for i,term in enumerate(matrix.all_terms):
        query["vector"][i] = alpha * (idf[term] if idf else 1) + (1 - alpha)*query["vector"][i]

    return context

def rank_documents(context: dict):
    """
    Ranks the `documents` with the `query` returning in the result in
    `ranked_documents` key. If `rank_threshold` is given then only values 
    higher will be returned
    """
    
    def sim(x,y):
        """
        Finds the cosine between the x and y vectors 
        """
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        if 0 in [norm_y, norm_x]:
            return 0
        return np.dot(x,y)/norm_x/norm_y

    rank_threshold = context.get("rank_threshold", 0)
    
    query = context["query"]
    documents = context["documents"]
    ranking = []
    for doc in documents:
        s = sim(query["vector"], doc["vector"])
        if s > rank_threshold:
            ranking.append((s, doc))
    ranking.sort(key=lambda x: -x[0])
    
    context["ranked_documents"] = ranking
    
    return context

class VectorialModel(InformationRetrievalModel):
    
    def __init__(self, corpus_address: str, smooth_query_alpha= 0.4, language="english", rank_threshold=0.5) -> None:
        # , query_pipeline: Pipeline, build_pipeline: Pipeline, query_context: dict, build_context: dict
        query_pipeline = Pipeline(tokenize_query, remove_stop_words_query, stemming_words_query, convert_query_to_vec, smooth_query_vec, rank_documents)
        build_pipeline = Pipeline(read_documents_from_hard_drive, tokenize_documents, remove_stop_words, stemming_words, add_term_matrix, calculate_idf, convert_doc_to_vec)
        query_context = {
            "smooth_query_alpha": smooth_query_alpha,
            "language": language,
            "rank_threshold": rank_threshold,
        }
        build_context = {
            "language": language
        }
        super().__init__(corpus_address, query_pipeline, build_pipeline, Pipeline(lambda x: x), query_context, build_context)
