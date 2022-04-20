from cmath import log
from typing import List

from .base import *

from ..pipes.pipeline import Pipe, Pipeline

def set_initial_prob_values(context: dict) -> dict:
    """
    Sets the initial probability `p` and `u` values. Both 
    are saved into `p_values`` and `u_values` keys respectively.
    """
    initial_p = {}
    initial_u = {}
    matrix = context["term_matrix"]
    initial_p = context.get('initial_p', 0.5)
    documents = context["documents"]
    for term in matrix.all_terms():
        initial_p[term] = initial_p # Constant value for p
        initial_u[term] = sum([1 for j in range(len(documents)) if matrix[term,j] > 0])/len(documents)
    
    context["p_values"] = initial_p
    context["u_values"] = initial_u
    return context

def calculate_rsv(context: dict) -> dict:
    """
    Calculates the RSV for each document. And return the results
    in `ranked_documents` key.
    """
    rsv = []
    rank_threshold = context.get("rank_threshold", 0)

    for doc in context["documents"]:
        rsv = 0
        for term in set(doc["tokens"]).intersection(context["query"]["tokens"]):
            p_val = context["p_values"][term]
            u_val = context["u_values"][term]
            rsv += log(p_val*(1-u_val)/u_val/(1-p_val))
        if rsv >= rank_threshold:
            rsv.append((rsv, doc))
    rsv.sort(key=lambda x: -x[0])
    context["ranked_documents"] = rsv
    return rsv

def apply_psedo_feedback(context: dict) -> dict:
    """
    Apply the pseudo-feedback algorithm updating `p_values` and `u_values`
    """
    n = 10
    while n > 0:
        returned_documents = [y for _,y in calculate_rsv(context)]
        for term in context["term_matrix"].all_terms():
            appears_in_relevant = 0
            appears_in = 0
            for doc in context["documents"]:
                if doc in returned_documents:
                    appears_in_relevant += 1
                if term in doc["tokens"]:
                    appears_in += 1
            context["p_values"][term] = appears_in_relevant/len(returned_documents)
            context["u_values"][term] = (appears_in - appears_in_relevant)/(len(context["documents"]) - len(returned_documents))
        n -= 1
    return context

def apply_feedback(context: dict) -> dict:
    """
    Apply the feedback algorithm updating `p_values` and `u_values` using the given
    `relevant_documents` and `non_relevant_documents` keys.
    """
    smooth_feedback_coef = context.get("smooth_feedback_coef", 0.5)
    n = 10
    relevant_documents = context.get("relevant_documents", [])
    non_relevant_documents = context.get("non_relevant_documents", [])
    while n > 0:
        for term in context["term_matrix"].all_terms():
            appears_in_relevant = 0
            appears_in = 0
            for doc in context["documents"]:
                if doc in relevant_documents:
                    appears_in_relevant += 1
                if term in doc["tokens"]:
                    appears_in += 1
            context["p_values"][term] = (appears_in_relevant + smooth_feedback_coef*context["p_values"][term])/(len(relevant_documents) + smooth_feedback_coef)
            context["u_values"][term] = (appears_in - appears_in_relevant + smooth_feedback_coef*context["u_values"][term])/(len(relevant_documents) + len(non_relevant_documents) - len(relevant_documents) + smooth_feedback_coef)
        n -= 1
    return context


class ProbabilisticModel(InformationRetrievalModel):

    def __init__(self, corpus_address: str, smooth_feedback_coef= 0.4, initial_p=0.5, language="english", rank_threshold=0.5, pseudo_feedback=True) -> None:
        # , query_pipeline: Pipeline, build_pipeline: Pipeline, query_context: dict, build_context: dict
        query_pipeline = Pipeline(tokenize_query, remove_stop_words_query, stemming_words_query, calculate_rsv)
        feedback_pipe = apply_feedback if not pseudo_feedback else apply_psedo_feedback
        build_pipeline = Pipeline(read_documents_from_hard_drive, tokenize_documents, remove_stop_words, stemming_words, set_initial_prob_values, feedback_pipe)
        feedback_pipeline = Pipeline(feedback_pipe)
        query_context = {
            "smooth_feedback_coef": smooth_feedback_coef,
            "initial_p": initial_p,
            "language": language,
            "rank_threshold": rank_threshold,
        }
        build_context = {
            "language": language
        }
        super().__init__(corpus_address, query_pipeline, build_pipeline, feedback_pipeline, query_context, build_context)

