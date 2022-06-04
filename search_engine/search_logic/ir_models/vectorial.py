import numpy as np

from .feedback import add_feedback_manager, add_feedback_to_query
from .query_expansion import add_query_expansion_manager
from ..pipes.pipeline import Pipeline
from .base import *
from sklearn.feature_extraction.text import TfidfVectorizer

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
    if any in `smooth_query_alpha`, defaults to 0.4.
    
    alpha*idf_i + (1-alpha)ntf_{iq} idf_i
    """
    query = context["query"]
    alpha = context.get("smooth_query_alpha", 0.4)
    idf = context.get("idf")
    matrix = context["term_matrix"]
    for i,term in enumerate(matrix.all_terms):
        query["vector"][i] = (alpha + (1 - alpha) *
                              query["vector"][i])*(idf[term] if idf else 1)

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

def add_vectorizer_vectorial(context: dict) -> dict:
    """
    Build and add a TF-IDF vectorizer to the context
    """
    return add_vectorizer(context, vectorizer_class=TfidfVectorizer)

def add_idf(context: dict):
    """
    Build an idf dictionary based in the `vectorizer` and the `term_matrix`. 
    The vectorizer must have an idf_ porperty that holds the idf for the i
    term matching matrix.all_terms index
    """
    vectorizer = context["vectorizer"]
    matrix = context["term_matrix"]
    context["idf"] = {term: vectorizer.idf_[i] for i,term in enumerate(matrix.all_terms)}
    return context

class VectorialModel(InformationRetrievalModel):
    
    def __init__(self, corpus_address: str, smooth_query_alpha= 0.0, language="english", rank_threshold=0.0,
                 alpha_rocchio=1, beta_rocchio=0.75, ro_rocchio=0.1) -> None:

        query_to_vec_pipeline = Pipeline(
            apply_text_processing_query, 
            build_query_matrix, 
            add_vector_to_query,
        )
        build_pipeline = Pipeline(
            read_documents_from_hard_drive, 
            add_tokens,
            add_stopwords,
            add_lemmatizer, # Stemmer XOR Lemmatizer 
            # add_stemmer, # Stemmer XOR Lemmatizer
            add_feedback_manager,
            add_query_expansion_manager,
            add_vectorizer_vectorial, 
            apply_text_processing, 
            build_matrix, 
            add_idf,
            add_vector_to_doc,
        )
        
        query_pipeline = Pipeline(
            add_feedback_to_query, 
            smooth_query_vec, 
            rank_documents,
        )
        query_context = {
            "smooth_query_alpha": smooth_query_alpha,
            "language": language,
            "rank_threshold": rank_threshold,
            "alpha_rocchio": alpha_rocchio,
            "beta_rocchio": beta_rocchio,
            "ro_rocchio": ro_rocchio,
        }
        build_context = {
            "language": language,
        }
        super().__init__(corpus_address, query_pipeline, query_to_vec_pipeline, build_pipeline, query_context, build_context)
