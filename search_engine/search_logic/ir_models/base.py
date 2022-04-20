from typing import List

from ..pipes.pipeline import Pipe, Pipeline 
import os
from typing import List, Tuple
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def read_documents_from_hard_drive(context: dict) -> dict:
    """
    Read documents from the directory stored in `corpus_address` key 
    and saved the raw texts in `raw_documents` key
    """
    documents = []
    corpus_address = context["corpus_address"]
    for doc in os.listdir(corpus_address):
        doc = os.path.join(corpus_address, doc)
        if os.path.isfile(doc):
            with open(doc) as file:
                documents.append({
                    "text":file.read(),
                    "dir": doc,
                })
    context["documents"] = documents
    return context

def tokenize_documents(context: dict, is_query=False) -> dict:
    """
    Read raw documents stored in `documents` key and add the
    processed documents as list of tokens in `tokens` key indice the 
    document dictionary
    """
    raw_documents = context["documents"] if not is_query else [context["query"]]
    language = context.get("language")
    language = language if language else 'english'
    for raw_doc in raw_documents:
        tokens = word_tokenize(raw_doc["text"], language=language)
        raw_doc["tokens"] = tokens
    return context

def tokenize_query(context: dict) -> dict:
    return tokenize_documents(context, True)

def remove_stop_words(context: dict, is_query=False) -> dict:
    """
    Remove the stop words and punctuation signs from `tokens` key in the documents
    """
    tokenized_documents = context["documents"] if not is_query else [context["query"]]
    language = context.get("language")
    language = language if language else 'english'
    stop_words = set(stopwords.words(language))
    punct = set(string.punctuation)
    ignore = stop_words.union(punct)
    
    for doc in tokenized_documents:
        # Filtering stopword
        no_stopwords = [w for w in doc["tokens"] if not w.lower() in ignore]
        doc["tokens"] = no_stopwords
    
    return context

def remove_stop_words_query(context: dict) -> dict:
    return remove_stop_words(context, True)

def stemming_words(context: dict, is_query=False) -> dict:
    """
    Apply stemming to `tokens` key in documents
    """
    documents = context["documents"] if not is_query else [context["query"]]
    language = context.get("language")
    language = language if language else 'english'
    stemmer = PorterStemmer()
    
    for doc in documents:
        # Stemming tokens
        stemmed = [stemmer.stem(w) for w in doc["tokens"]]
        doc["tokens"] = stemmed
    
    return context

def stemming_words_query(context: dict) -> dict:
    return stemming_words(context, True)

def add_term_matrix(context: dict) -> dict:
    """
    Adds a term-document matrix in `term_matrix` key
    """
    matrix = Matrix([doc["tokens"] for doc in context["documents"]])
    context["term_matrix"] = matrix
    return context

class Matrix:
    def __init__(self, tokens: List[List[str]]) -> None:
        self.all_terms = list(set(x for y in tokens for x in y))
        self.all_terms.sort()
        self.__matrix = {(t,i): len([y for y in doc if y.lower() == t.lower()]) for t in self.all_terms for i,doc in enumerate(tokens)}

    def __getitem__(self, key: Tuple[str,int]) -> int:
        return self.__matrix.get(key, 0)

class InformationRetrievalModel:
    
    def __init__(self, corpus_address:str, query_pipeline: Pipeline, build_pipeline: Pipeline, feedback_pipeline: Pipeline, query_context: dict, build_context: dict) -> None:
        """
        Returns the 'ranked_documents' key from the last result of `query_pipeline`.
        
        The corpus_address and the query can be found in equaly named keys in the dictionary received as argument in the pipes.
        
        The `query_context` and `build_context` are added as initial values for the corresponding pipelines
        
        Basic recomended query_pipeline:
        get_relevant_doc_pipe: Pipe, rank_doc_pipe: Pipe
        
        Basic recomended build_pipeline
        """
        self.corpus_address = corpus_address
        self.query_context = query_context
        self.build_context = build_context
        self.query_pipeline = query_pipeline
        self.build_pipeline = build_pipeline
        self.feedback_pipeline = feedback_pipeline
    
    def resolve_query(self, query:str) -> List[dict]:
        """
        Returns an ordered list of the ranked relevant documents.
        """
        pipeline = Pipeline(Pipe(lambda x: {"corpus_address": x, "query": {"text": query}, **self.query_context, **self.build_result}), self.query_pipeline)
        result = pipeline(query)
        return result["ranked_documents"]
    
    def build(self) -> dict:
        """
        Builds the model according the documents returning the context
        """
        pipeline = Pipeline(Pipe(lambda x: {"corpus_address": x, **self.build_context}), self.build_pipeline)
        self.build_result = pipeline(self.corpus_address)
        return self.build_result

    def add_relevant_and_non_relevant_documents(self, new_relevant_documents: List[dict], new_non_relevant_documents: List[str]) -> List[dict]:
        """
        Adds the relevant and non relevant documents to the model and apply the feedback pipeline
        """
        if "relevant_documents" not in self.build_result:
            self.build_result["relevant_documents"] = []
        if "non_relevant_documents" not in self.build_result:
            self.build_result["non_relevant_documents"] = []
        self.build_result["relevant_documents"] = self.build_result["relevant_documents"].extend(new_relevant_documents)
        self.build_result["non_relevant_documents"] = self.build_result["non_relevant_documents"].extend(new_non_relevant_documents)
        self.feedback_pipeline(self.build_result)