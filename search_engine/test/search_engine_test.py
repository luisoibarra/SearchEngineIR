import pytest
import ir_datasets as ir
import sys
from pathlib import Path
import time

if __name__ == "__main__":
    sys.path.append(str((Path(__file__) / ".." / "..").resolve()))

from search_logic.ir_models.vectorial import VectorialModel
from search_logic.ir_models.base import InformationRetrievalModel

def test_model():
    """
    Simple test to Cranfield to see basic metrics.
    """
    base_path = (Path(__file__) / "..").resolve() / "cranfield_corpus"
    
    start = time.time()
    model = VectorialModel(base_path)
    model.build()
    print("Build Time:", time.time() - start, "seconds")

    dataset = ir.load("cranfield")
    queries = { query_id: text for query_id, text in dataset.queries_iter() }

    tp, fn, fp, tn = 0,0,0,0

    for qrel in dataset.qrels_iter():
        # Relevance
        # -1: Not relevant
        # 1: Minimun interest
        # 2: Useful references
        # 3: High degree of relevance
        # 4: Complete answer to the question
        query_id, doc_id, relevance, iteration = qrel
        query = queries.get(query_id)
        if not query: # The querys in some relations are missing
            continue
        rank = model.resolve_query(queries[query_id])
        rank = rank[:30]
        rec_docs = [Path(doc['dir']).name[:-4] for _, doc in rank]
        if relevance > 0:
            if doc_id in rec_docs:
                # print("True Positive")
                tp+=1
            else:
                # print("False Negative")
                fn+=1
        else:
            if doc_id in rec_docs:
                # print("False Positive")
                fp+=1
            else:
                # print("True Negative")
                tn+=1
    print("True positive", tp)
    print("False negative", fn)
    print("False positive", fp)
    print("True negative", tn)
    print("Test Time:", time.time() - start, "seconds")

test_model()