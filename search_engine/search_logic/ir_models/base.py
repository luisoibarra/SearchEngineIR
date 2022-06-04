from pathlib import Path
from re import A
from typing import Callable, List

import numpy as np

from .feedback import add_feedback_vectors
from .query_expansion import add_query_expansions

from .utils import get_object, save_object
from ..pipes.pipeline import Pipe, Pipeline 
import os
from typing import List, Tuple
from nltk.corpus import stopwords
from nltk.corpus import words
import string
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from concurrent.futures import ThreadPoolExecutor, wait

from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# READ PIPES
def read_documents_from_hard_drive(context: dict) -> dict:
    """
    Read documents from the directory stored in `corpus_address` key 
    and saved the raw texts in `raw_documents` key
    """
    
    documents = []
    corpus_address = context["corpus_address"]
    # Recursively read all files in the directory
    addresses = [dir for dir in os.walk(corpus_address)]
    max_workers = 20
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as exe:

        def read_files(section:int):
            for root, dirs, files in addresses[max_workers*section:max_workers*(section+1)]:
                for file in files:
                    with open(os.path.join(root, file), "r", encoding="utf8", errors='ignore') as f:
                        try:
                            documents.append({
                                "text": f.read(),
                                "root": root,
                                "dir": os.path.join(root, file),
                                "topic": root.split("/")[-1].split()
                                })
                        except Exception as e:
                            print("Error reading file", file, e)

        for i in range(max_workers):
            futures.append(exe.submit(read_files, section=i))

    wait(futures)
    documents.sort(key=lambda x: x['dir'])
    context["documents"] = documents
    print("Documents read", len(documents))
    print("End document collecting")
    return context

def add_tokens(context: dict) -> dict:
    """
    Adds the saved tokens if any to the docs representation in `tokens` key
    """
    documents = context["documents"]
    
    tok_dict = get_object([doc['dir'] for doc in documents], suffix="tok")
    if tok_dict:
        for doc in documents:
            tokens = tok_dict.get(doc['dir'])
            if tokens:
                doc['tokens'] = tokens

    return context

def add_stopwords(context: dict) -> dict:
    """
    Adds the `stop_words` to the context
    """
    language = context.get("language", "english")
    stop_words = set(stopwords.words(language))
    punct = set(string.punctuation)
    ignore = stop_words.union(punct)
    context["stop_words"] = ignore
    
    
    return context

def add_stemmer(context: dict) -> dict:
    """
    Adds the `stemmer` used to the context
    """
    context["stemmer"] = PorterStemmer()
    return context

def add_lemmatizer(context: dict) -> dict:
    """
    Adds the `lemmatizer` used to the context
    """
    context["lemmatizer"]= WordNetLemmatizer()
    return context

def apply_text_processing_query(context: dict, tokenizer=word_tokenize) -> dict:
    """
    Apply all preprocessing to query before creating the vector matrix
    """
    return apply_text_processing(context, tokenizer, is_query=True)

def apply_text_processing(context: dict, tokenizer=word_tokenize, is_query=False) -> dict:
    """
    Apply all preprocessing to text before creating the vector matrix
    """

    language = context.get("language", "english")
    documents = context["documents"] if not is_query else [context["query"]]
    stopwords = context.get("stop_words")
    stemmer = context.get("stemmer")
    lemmatizer = context.get("lemmatizer")
    englishwords = set(words.words())

    all_tokens = True

    for doc in documents:
        if 'tokens' not in doc: # Tokens aren't saved
            tokens = tokenizer(doc['text'], language=language)
            if stopwords:
                tokens = [w for w in tokens  # this last and could be removed
                                if not w.lower() in stopwords and w.isalpha() and w.lower() in englishwords]
                #print("Stop words removed")    
            if lemmatizer:
                tokens = [lemmatizer.lemmatize(x) for x in tokens]
                #print("Lemmatizing applied")
            if stemmer:
                tokens = [stemmer.stem(x) for x in tokens]
                #print("Stemming applied")
            doc['tokens'] = tokens
            all_tokens = False

    if not is_query and not all_tokens:
        save_object([doc['dir'] for doc in documents], { doc['dir']:doc['tokens'] for doc in documents }, suffix="tok")

    return context

def add_vectorizer(context: dict, vectorizer_class=CountVectorizer, vectorizer_kwargs={}) -> dict:
    """
    Adds the given `vectorizer` to the context with a custom tokenizer function
    """
    documents = context["documents"]

    vectorizer = get_object([doc['dir'] for doc in documents])
    context["vectorizer_fitted"] = True

    if vectorizer is None:
        vectorizer = vectorizer_class(**vectorizer_kwargs)
        context["vectorizer_fitted"] = False

    context["vectorizer"] = vectorizer
    
    return context

def build_matrix(context:dict, is_query=False) -> dict:
    """
    Builds a `term_matrix` based on the `vectorizer` provided
    """
    documents = context["documents"] if not is_query else [context["query"]]
    vectorizer = context["vectorizer"]
    vectorizer_fitted = context.get("vectorizer_fitted", False)
    
    text_documents = [doc["tokens"] for doc in documents]
    text_documents = [" ".join(toks) for toks in text_documents]

    if is_query:
        matrix = vectorizer.transform(text_documents)
    else: 
        dir_documents = [doc["dir"] for doc in documents]
        matrix = get_object(dir_documents, suffix="mtx")
        if matrix is None:
            if not vectorizer_fitted:
                matrix = vectorizer.fit_transform(text_documents)
                save_object(dir_documents, vectorizer)
                context["vectorizer_fitted"] = True
            else:
                matrix = vectorizer.transform(text_documents)
            save_object(dir_documents, matrix, suffix="mtx")

    vec_matrix = VecMatrix(vectorizer.get_feature_names_out(), matrix)

    context["term_matrix" if not is_query else "query_matrix"] = vec_matrix

    return context

def build_query_matrix(context: dict) -> dict:
    return build_matrix(context, is_query=True)

def add_vector_to_doc(context: dict, is_query=False) -> dict:
    """
    Add the document's vector representation to the doc dictionary based on
    the `term_matrix`
    """
    documents = context["documents"] if not is_query else [context["query"]]
    matrix = context["term_matrix" if not is_query else "query_matrix"]

    for i, doc in enumerate(documents):
        vec = matrix.matrix[i, :]
        doc["vector"] = vec.toarray()[0]
    
    return context

def add_vector_to_query(context: dict) -> dict:
    return add_vector_to_doc(context, is_query=True)

class VecMatrix:
    def __init__(self, all_terms, matrix) -> None:
        self.all_terms = all_terms
        self.matrix = matrix
        self.__index_map = {x:i for i,x in enumerate(self.all_terms)}

    def __getitem__(self, key: Tuple[str,int]) -> int:
        return self.matrix[key[1], self.__index_map[key[0]]]

class InformationRetrievalModel:
    
    def __init__(self, corpus_address:str, query_pipeline: Pipeline, query_to_vec_pipeline: Pipeline, build_pipeline: Pipeline, query_context: dict, build_context: dict,
                 feedback_pipeline: Pipeline=None, expansion_query_pipeline: Pipeline=None) -> None:
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
        self.query_to_vec_pipeline = query_to_vec_pipeline
        self.build_pipeline = build_pipeline
        self.feedback_pipeline = feedback_pipeline if feedback_pipeline else Pipeline(add_feedback_vectors)
        self.query_expansion_pipeline = expansion_query_pipeline if expansion_query_pipeline else Pipeline(add_query_expansions)

    def resolve_query(self, query:str) -> List[dict]:
        """
        Returns an ordered list of the ranked relevant documents.
        """
        pipeline = Pipeline(
            Pipe(
                lambda x: {
                    "corpus_address": x, 
                    "query": {"text": query}, 
                    **self.query_context, **self.build_result
                }), 
            self.query_to_vec_pipeline, 
            self.query_pipeline,
        )
        result = pipeline(query)
        return result["ranked_documents"]
    
    def build(self) -> dict:
        """
        Builds the model according the documents returning the context
        """
        pipeline = Pipeline(
            Pipe(
                lambda x: {
                    "corpus_address": x, 
                    **self.build_context
                }), 
            self.build_pipeline,
        )
        self.build_result = pipeline(self.corpus_address)
        print("build ended")
        return self.build_result

    def add_relevant_and_not_relevant_documents(self, query:dict, new_relevant_documents: List[dict], new_not_relevant_documents: List[str]):
        """
        Adds the relevant and not relevant documents to the model and apply the feedback pipeline
        """
        feedback_vector = self.build_result.copy()
        feedback_vector["query"] = self.query_to_vec_pipeline({"query": {"text":query}, **self.query_context, **feedback_vector})["query"]
        feedback_vector["new_relevant_documents"] = new_relevant_documents
        feedback_vector["new_not_relevant_documents"] = new_not_relevant_documents
        self.feedback_pipeline(feedback_vector)

    def get_expansion_query(self, query: str) -> List[str]:
        query_expansion_context = self.query_to_vec_pipeline({"query": {"text":query}, **self.query_context, **self.build_result})
        query_expansion_context = self.query_expansion_pipeline(query_expansion_context)
        return query_expansion_context["query_expansions"]
