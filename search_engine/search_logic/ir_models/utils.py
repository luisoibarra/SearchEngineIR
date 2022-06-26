import hashlib
import pickle
from typing import List
from pathlib import Path
import re
import ir_datasets as ir
import numpy as np


def sim(x, y):
    """
        Finds the cosine between the x and y vectors
    """
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    if 0 in [norm_y, norm_x]:
        return 0
    return np.dot(x, y)/(norm_x*norm_y)


def add_training_documents(context: dict):
    """
    Adds the training corpus documents and training information.
    """

    dataset_name = context.get("dataset_name", "cranfield")

    documents = []
    relevance = set()
    queries_dict = {}

    if dataset_name in ["cranfield"]:
        dataset = ir.load(dataset_name)
        # Documents
        for doc in dataset.docs_iter():
            doc_id, title, text = doc.doc_id, doc.title, doc.text
            if text:
                documents.append({
                    "text": text,
                    "dir": doc_id,
                    "topic": title
                })

        # Queries
        queries_dict = {q.query_id: q.text for q in dataset.queries_iter()}

        # Relevance
        for qrel in dataset.qrels_iter():
            q_id, d_id, rel = qrel.query_id, qrel.doc_id, qrel.relevance
            if q_id in queries_dict:
                relevance.add((q_id, d_id, rel))
    elif dataset_name in ["med"]:

        # Documents
        document_path = Path(__file__, "..", "..", "..", "test",
                             f"{dataset_name}_raw", f"{dataset_name.upper()}.ALL").resolve()
        for doc_id, text in read_document(document_path):
            documents.append({
                "text": text,
                "dir": doc_id,
                "topic": dataset_name,
            })

        # Queries
        query_path = Path(__file__, "..", "..", "..", "test",
                          f"{dataset_name}_raw", f"{dataset_name.upper()}.QRY").resolve()
        for q_id, text in read_document(query_path):
            queries_dict[q_id] = text

        # Relevance
        relevance_path = Path(__file__, "..", "..", "..", "test",
                              f"{dataset_name}_raw", f"{dataset_name.upper()}.REL").resolve()
        relevance = set(x for x in read_relevance(relevance_path))
    else:
        raise Exception(f"Corpus {dataset_name} not supported")

    context["documents"] = documents

    context["training_documents"] = documents
    context["training_queries_dict"] = queries_dict
    context["training_relevance_tuples"] = relevance

    return context



def get_object(documents: List[str], suffix="vec"):
    """
    Returns an object associated with documents. If the object doesn't exist
    returns None
    """
    path = (Path(__file__) / ".." / "pickles").resolve()
    text_hash = str(hashlib.sha224(b''.join(x.encode() for x in documents)).hexdigest())
    filename = text_hash + "." + suffix
    for file in path.iterdir():
        if file.is_file() and file.name == filename:
            obj_path = path / filename
            obj = pickle.load(obj_path.open('rb'))
            return obj
    return None

def save_object(documents: List[str], obj, suffix="vec"):
    """
    Saves an object associated with documents
    """
    path = (Path(__file__) / ".." / "pickles").resolve()
    text_hash = str(hashlib.sha224(b''.join(x.encode() for x in documents)).hexdigest())
    obj_path = path / (text_hash + "." + suffix)
    obj_path.touch()
    pickle.dump(obj, obj_path.open("wb"))


def read_document(path: Path):
    """
    Returns an interator (doc_id, text)

    Format:\n
    I DocID.\n
    W.\n
    TEXT
    ...
    """
    lines = path.read_text().splitlines()

    id_regex = r"^.I (?P<id>\d+)\s*"
    id_regex = re.compile(id_regex)

    i = 0
    while i < len(lines):
        # Get doc id
        doc_id = id_regex.match(lines[i]).groupdict()["id"]
        i+=1
        # Skip word separator
        assert ".W" in lines[i]
        i+=1
        start_text_line = i
        end_text_line = i
        is_new_doc = id_regex.match(lines[i])
        while not is_new_doc:
            i+=1
            end_text_line = i
            if i >= len(lines):
                break
            is_new_doc = id_regex.match(lines[i])
        text = "\n".join(lines[start_text_line:end_text_line])
        yield doc_id, text

def read_relevance(path: Path):
    """
    Returns an iterator of (query_id, doc_id, relevance)
    """
    for line in path.read_text().splitlines():
        query_id, _, doc_id, relevance = line.split()
        yield query_id, doc_id, int(relevance)
