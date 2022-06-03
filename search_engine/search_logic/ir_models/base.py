from pathlib import Path
from re import A
from typing import Callable, List

import numpy as np

from .utils import get_object, save_object
from ..pipes.pipeline import Pipe, Pipeline 
import os
from typing import List, Tuple
from nltk.corpus import stopwords
from nltk.corpus import words
import string
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

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
    for root, dirs, files in os.walk(corpus_address):
        # print("Actual dir topics", root.split("/")[-1].split())
        # if root.split("/")[-1] not in [
        #      "cars",
        #      "sport hockey",
        #      "atheism",
        #      "computer system ibm pc hardware",
        #      "random",
        # ]:
        #     continue
        for file in files:
            # if len(documents) > 5:
            #     break
            # print("File processed",file)
            with open(os.path.join(root, file), "r", encoding="utf8", errors='ignore') as f:
                try:
                    documents.append({
                        "text": f.read(),
                        "root": root,
                        "dir": os.path.join(root, file),
                        "topic": root.split("/")[-1].split()
                        })
    # for doc in os.listdir(corpus_address):
    #     doc = os.path.join(corpus_address, doc) 
        
    #     if os.path.isfile(doc):
    #         with open(doc) as file:
    #             try:
    #                 documents.append({
    #                     "text":file.read(),
    #                     "dir": doc,
    #                 })
                except Exception as e:
                    print("Error reading file", file, e)
    context["documents"] = documents
    print("Documents read", len(documents))
    print("End document collecting")
    return context

## MANUAL TEXT PROCESSING

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
    print("Tokens extracted")
    return context

def tokenize_query(context: dict) -> dict:
    return tokenize_documents(context, True)

def remove_stop_words(context: dict, is_query=False) -> dict:
    """
    Remove the stop words and punctuation signs from `tokens` key in the documents
    """
    tokenized_documents = context["documents"] if not is_query else [context["query"]]
    language = context.get("language", "english")
    stop_words = set(stopwords.words(language))
    punct = set(string.punctuation)
    ignore = stop_words.union(punct)
    englishwords = set(words.words())
    

    for doc in tokenized_documents:
        # Filtering stopword
        no_stopwords = [w for w in doc["tokens"]                  #this last and could be removed
                        if not w.lower() in ignore and w.isalpha() and w.lower() in englishwords]
        doc["tokens"] = no_stopwords
    print("Stop words removed")
    return context

def remove_stop_words_query(context: dict) -> dict:
    return remove_stop_words(context, True)

def lemmatizing_words(context: dict, is_query=False) -> dict:
    """
    Lemmatize the words in `tokens` key in documents
    """
    documents = context["documents"] if not is_query else [context["query"]]
    language = context.get("language")
    language = language if language else 'english'
    lemma = WordNetLemmatizer()

    for doc in documents:
        # Lemmatizing tokens
        lemmatized_tokens = [lemma.lemmatize(w.lower()) for w in doc["tokens"]]
        doc["tokens"] = lemmatized_tokens
        lemmatized_topic = [lemma.lemmatize(w.lower()) for w in doc["topic"]]
        doc["topic"] = lemmatized_topic
    print("Lemmatizing applied")
    return context

def lemmatizing_query(context: dict) -> dict:
    return lemmatizing_words(context, True)


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
        stemmed = [stemmer.stem(w.lower()) for w in doc["tokens"]]
        stemmed_topic = [stemmer.stem(w.lower()) for w in doc["topic"]]
        doc["tokens"] = stemmed
        doc["topic"] = stemmed_topic

    print(" Stemming applied")
    return context

def stemming_words_query(context: dict) -> dict:
    return stemming_words(context, True)

def add_term_matrix(context: dict) -> dict:
    """
    Adds a inverted index dictionary in `term_matrix` key
    """
    documents = context["documents"]
    term_matrix = {}
    for doc in documents:
        for token in doc["tokens"]:
            if token not in term_matrix.keys():
                term_matrix[token] = []
            term_matrix[token].append(doc["dir"])
    context["term_matrix"] = term_matrix
    
    print("Term matrix created")
    return context

def add_feedback_vectors(context: dict):
    """
    Adds the `new_relevant_documents` and the `new_not_relevant_documents` to
    the `feedback_manager` associated with `query`
    """
    feedback_manager = context.get("feedback_manager")
    if feedback_manager:
        query = context["query"]["vector"]
        new_relevant_documents = context.get("new_relevant_documents", [])
        new_not_relevant_documents = context.get("new_not_relevant_documents", [])
        for rel in new_relevant_documents:
            feedback_manager.mark_relevant(query, rel["vector"])
        for not_rel in new_not_relevant_documents:
            feedback_manager.mark_not_relevant(query, not_rel["vector"])
        print("Feedback vectors added")
    return context

class Matrix:
    def __init__(self, tokens: List[List[str]]) -> None:
        self.all_terms = list(set(x for y in tokens for x in y))
        self.all_terms.sort()
        self.__matrix = {(t,i): len([y for y in doc if y.lower() == t.lower()]) for t in self.all_terms for i,doc in enumerate(tokens)}

    def __getitem__(self, key: Tuple[str,int]) -> int:
        return self.__matrix.get(key, 0)

## SKLEARN VECTORIZER

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
    if not is_query and context.get("vectorizer_fitted"):
        # Processing not needed
        return context

    language = context.get("language", "english")
    documents = context["documents"] if not is_query else [context["query"]]
    stopwords = context.get("stop_words")
    stemmer = context.get("stemmer")
    lemmatizer = context.get("lemmatizer")
    englishwords = set(words.words())

    for doc in documents:
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
        
        doc['text'] = " ".join(tokens)

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
    
    text_documents = [doc["text"] for doc in documents]
    
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
                 feedback_pipeline: Pipeline=None) -> None:
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

class FeedbackManager:
    """
    Base class to manage the relevant and not relevant documents for a given query 
    """

    def __init__(self) -> None:
        self.relevant_dict = {}
        self.not_relevant_dict = {}

    def _mark_document(self, query, document, relevant_dict):

        # Adds the document in a set with all relevant or not relevant documents of the query
        query = tuple(query)
        document = tuple(document)
        if query in relevant_dict:
            relevant_dict[query].update([document])
        else:
            relevant_dict[query] = set([document])

    def mark_relevant(self, query, document):
        """
        Mark the document as relevant to the query
        """
        self._mark_document(query, document, self.relevant_dict)
    
    def mark_not_relevant(self, query, document):
        """
        Mark the document as not relevant to the query
        """
        self._mark_document(query, document, self.not_relevant_dict)

    def get_relevants(self, query):
        """
        Return the list of relevant documents given the query
        """
        try:
            return [np.array(x) for x in self.relevant_dict[tuple(query)]]
        except KeyError:
            return []

    def get_not_relevants(self, query):
        """
        Return the list of non relevant documents given the query
        """
        try:
            return [np.array(x) for x in self.not_relevant_dict[tuple(query)]]
        except KeyError:
            return []
    
