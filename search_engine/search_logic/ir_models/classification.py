from typing import List, Tuple
from sklearn.svm import LinearSVC
import ir_datasets as ir
import numpy as np
from pathlib import Path
from .base import InformationRetrievalModel
from .utils import get_object, save_object

from .vectorial import VectorialModel
from ..pipes.pipeline import Pipeline

def __rank_dist(classifier: LinearSVC, query: dict, doc_dicts: dict) -> List[Tuple[float, dict]]:
    ranking = []

    # See pag ~340 Manning, Optimizing Search Engines using Clicktrhough Data
    for doc_i in doc_dicts.values():
        d_q_feature = __get_feature(doc_i, query)

        # Calculating the distance from the margin plane
        w = classifier.decision_function([d_q_feature])

        ranking.append((w[0], doc_i))
    
    return ranking

def __rank_pairwise(classifier: LinearSVC, query: dict, doc_dicts: dict) -> List[Tuple[float, dict]]:
    ranking = {doc_id: 0 for doc_id in doc_dicts}

    # Optimization
    feature_dict = {}

    # See pag 345 Manning # Not feasible |D|^2
    list_doc = list(doc_dicts)
    for i, d_id_i in enumerate(list_doc):
        d_i_q_feature = __get_feature(doc_dicts[d_id_i], query)
        for j, d_id_j in enumerate(list_doc[i+1:], i+1):
            if d_id_j not in feature_dict:
                d_j_q_feature = __get_feature(doc_dicts[d_id_j], query)
                feature_dict[d_id_j] = d_j_q_feature
            else:
                d_j_q_feature = feature_dict[d_id_j]

            i_better_than_j = classifier.predict([d_i_q_feature - d_j_q_feature])[0]
            if i_better_than_j == 1:
                ranking[d_id_i] += 1
            else:
                ranking[d_id_j] += 1

    ranking = [(value, doc_dicts[doc_id]) for doc_id, value in ranking.items()]

    return ranking

def __get_feature(doc_i_repr: dict, query_repr: dict):
    """
    Get the feature vector for the document and the query
    """
    q_vec = query_repr["vector"]
    d_vec = doc_i_repr["vector"]
    
    # Cosine Similarity
    cosine = np.dot(q_vec,d_vec)/(np.linalg.norm(q_vec) * np.linalg.norm(d_vec))

    # Feature: QueryVec|DocVec|Cosine|TODO FEATURE ENGINEERING
    
    feature = np.concatenate((
            q_vec, 
            d_vec, 
            [
                len(set(query_repr["tokens"]).intersection(doc_i_repr["tokens"])), 
            ],
            [
                cosine
            ]
        ), 
    axis=0)
    return feature

def add_crainfield_training_documents(context: dict):
    """
    Adds the Cranfield corpus documents and training information.
    """

    documents = []
    dataset = ir.load("cranfield")
    for doc in dataset.docs_iter():
        doc_id, title, text, author, bib = doc
        if text:
            documents.append({
                "text": text,
                "dir": doc_id,
                "topic": title
            })

    queries_dict = { query_id: text for query_id, text in dataset.queries_iter() }

    relevance = set()
    for q_id, d_id, rel, _ in dataset.qrels_iter():
        relevance.add((q_id, d_id, rel))

    context["documents"] = documents
    
    context["training_documents"] = documents
    context["training_queries_dict"] = queries_dict
    context["training_relevance_tuples"] = relevance
    
    return context


def add_vectorial(context: dict):
    """
    Adds a vectorial model. This model will vectorize querys and documents for feature engineering
    """
    # Prevent error of multiple definitions for corpus_address
    corpus_address = context["corpus_address"]
    context.__delitem__("corpus_address")

    # Creating and building training model
    training_model = VectorialModel(corpus_address, add_document_pipe=add_crainfield_training_documents, **context)
    training_build_context = training_model.build()

    context.update(training_build_context)

    # Assigning trained document representation
    context["training_documents"] = training_build_context["documents"]

    # Model in charge of building vec representation
    context["vectorial"] = training_model 
    context["documents"] = training_build_context["documents"]

    return context


def train_margin_classifier(context: dict):
    """
    Train a SVM Classification model. 
    """

    training_model = context["vectorial"]

    queries_dict = context["training_queries_dict"]
    relevances = context["training_relevance_tuples"]
    training_documents = context["training_documents"]


    training_doc_cache_key = [doc["dir"] for doc in training_documents]
    classifier = get_object(training_doc_cache_key, "svm")
    if classifier: # Classifier is saved
        context["classifier"] = classifier
        return context

    classifier = LinearSVC()

    doc_dicts = { doc_id: doc_repr for doc_id, doc_repr in [(Path(doc["dir"]).name.split(".")[0], doc) for doc in training_documents] }

    # Getting query representations
    query_repr_dicts = get_object(training_doc_cache_key, "query_dict")
    if not query_repr_dicts:
        query_repr_dicts = {}
        for query_id, query in queries_dict.items():
            training_model.resolve_query(query)
            query_dict = training_model.last_resolved_query_context["query"]
            query_repr_dicts[query_id] = query_dict # Calculated representation
        save_object(training_doc_cache_key, query_repr_dicts, "query_dict")

    query_doc_relevance = {(q_id,d_id): rel for q_id, d_id, rel in relevances}

    # Building Features
    features = [ ]
    labels = [ ]

    # See pag ~340 Manning
    for (q_id, d_id), rel in query_doc_relevance.items():
        if q_id not in query_repr_dicts:
            print(f"Missing query for query_id: {q_id}")
            continue
        if d_id not in doc_dicts:
            print(f"Missing doc for doc_id: {d_id}")
            continue
        d_q_feature = __get_feature(doc_dicts[d_id], query_repr_dicts[q_id])
        features.append(d_q_feature)
        label = 0 if rel <= 0 else 1
        labels.append(label)

    print("Training:")
    print("Feature size:", len(features[0]))
    print("Training samples:", len(features))
    classifier.fit(features, labels)

    save_object([doc["dir"] for doc in training_documents], classifier, "svm")

    mean_accuracy = classifier.score(features, labels)
    print("Mean accuracy:", mean_accuracy)

    context["classifier"] = classifier

    return context


def train_rank_classifier(context: dict):
    """
    Train a RankSVM model.
    """

    training_model = context["vectorial"]

    queries_dict = context["training_queries_dict"]
    relevances = context["training_relevance_tuples"]
    training_documents = context["training_documents"]


    training_doc_cache_key = [doc["dir"] for doc in training_documents]
    classifier = get_object(training_doc_cache_key, "svm_rank")
    if classifier: # Classifier is saved
        context["classifier"] = classifier
        return context

    classifier = LinearSVC()

    doc_dicts = { doc_id: doc_repr for doc_id, doc_repr in [(Path(doc["dir"]).name.split(".")[0], doc) for doc in training_documents] }

    # Getting query representations
    query_repr_dicts = get_object(training_doc_cache_key, "query_dict")
    if not query_repr_dicts:
        query_repr_dicts = {}
        for query_id, query in queries_dict.items():
            training_model.resolve_query(query)
            query_dict = training_model.last_resolved_query_context["query"]
            query_repr_dicts[query_id] = query_dict # Calculated representation
        save_object(training_doc_cache_key, query_repr_dicts, "query_dict")

    query_doc_relevance = {(q_id,d_id): rel for q_id, d_id, rel in relevances}
    query_info = { q_id: [] for q_id in queries_dict }
    for q_id, d_id in query_doc_relevance:
        if q_id in query_info: # Some querys are missing in the coprus
            query_info[q_id].append(d_id)

    # Building Features
    features = [ ]
    labels = [ ]

    # See pag 345 Manning
    for q_id, query_docs_info in query_info.items():
        for i, d_id_i in enumerate(query_docs_info):
            d_i_rel = query_doc_relevance[q_id, d_id_i]
            d_i_feature = __get_feature(doc_dicts[d_id_i], query_repr_dicts[q_id])
            
            for j, d_id_j in enumerate(query_docs_info[i+1:], i+1):
                d_j_rel = query_doc_relevance[q_id, d_id_j]
                
                if d_i_rel == d_j_rel: # Same relevance ignored
                    continue

                if d_i_rel < d_j_rel:
                    d_id_i, d_id_j = d_id_j, d_id_i

                d_j_feature = __get_feature(doc_dicts[d_id_j], query_repr_dicts[q_id])
                
                positive_result = d_i_feature - d_j_feature
                
                # Balancing data
                if (i + j) % 2:
                    features.append(positive_result)
                    labels.append(1) # doc_i > doc_j given query
                else:
                    features.append(-positive_result)
                    labels.append(0) 


    print("Training:")
    print("Feature size:", len(features[0]))
    print("Training samples:", len(features))
    classifier.fit(features, labels)

    save_object(training_doc_cache_key, classifier, "svm_rank")

    mean_accuracy = classifier.score(features, labels)
    print("Mean accuracy:", mean_accuracy)

    context["classifier"] = classifier

    return context


def rank_documents_svm_margin_distance_classifier(context: dict):
    """
    Ranking is done by comparing the distance of the document/query features to its margin.
    Distance will be negative in case of been in the NonRelevant class
    """

    classifier: LinearSVC = context["classifier"]

    query = context["query"]
    documents = context["documents"]

    doc_dicts = { doc_id: doc_repr for doc_id, doc_repr in [(doc["dir"], doc) for doc in documents] }

    ranking = __rank_dist(classifier, query, doc_dicts)

    ranking.sort(key=lambda x: -x[0])

    context["ranked_documents"] = ranking

    return context


def rank_documents_rank_svm_classifier(context: dict):
    """
    Ranking is done by comparing the amount doc_i's winnings.
    """
    
    classifier: LinearSVC = context["classifier"]

    query = context["query"]
    documents = context["documents"]

    doc_dicts = { doc_id: doc_repr for doc_id, doc_repr in [(doc["dir"], doc) for doc in documents] }

    # ranking = __rank_pairwise(classifier, query, doc_dicts) # Not feasible
    ranking = __rank_dist(classifier, query, doc_dicts)

    ranking.sort(key=lambda x: -x[0])

    context["ranked_documents"] = ranking

    return context


def classifier_query_to_vec(context: dict):
    """
    """
    model = context["vectorial"]
    return model.query_to_vec_pipeline(context)


def classifier_query(context: dict):
    """
    """
    model = context["vectorial"]
    return model.query_pipeline(context)


class ClassificationSVMModel(InformationRetrievalModel):
    
    def __init__(self, corpus_address: str, use_rank_svm=True, language: str = "english") -> None:

        query_pipeline = Pipeline(
            classifier_query,
            rank_documents_svm_margin_distance_classifier if not use_rank_svm else rank_documents_rank_svm_classifier,
        )

        build_pipeline = Pipeline(
            add_crainfield_training_documents,
            add_vectorial,
            train_margin_classifier if not use_rank_svm else train_rank_classifier,
        )

        query_to_vec_pipeline = Pipeline(
            classifier_query_to_vec,
        )

        query_context = {
            "language": language
        }

        build_context = {
            "language": language
        }

        feedback_pipeline = None # Default

        expansion_query_pipeline = None # Default

        super().__init__(corpus_address, query_pipeline, query_to_vec_pipeline, build_pipeline, query_context, build_context, feedback_pipeline, expansion_query_pipeline)