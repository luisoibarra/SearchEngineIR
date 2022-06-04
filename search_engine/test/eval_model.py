from cmath import nan
import ir_datasets as ir
import sys
from pathlib import Path
import time
import pandas as pd

if __name__ == "__main__":
    sys.path.append(str((Path(__file__) / ".." / "..").resolve()))

from search_logic.ir_models.vectorial import VectorialModel

def get_qrels_dataframe():
    """
    Converts the Cranfield Relations Dataset into a DataFrame
    """
    dataset = ir.load("cranfield")
    df = pd.DataFrame(((q_id, d_id, rel) for q_id, d_id, rel, _ in dataset.qrels_iter()) ,columns=["query_id", "doc_id", "relevance"])
    return df

def get_pickled_stats() -> pd.DataFrame:
    """
    Read pickled stats.df 
    """
    base_path = (Path(__file__) / "..").resolve()
    stats_path = (base_path / "stats.df").resolve()
    stats = pd.read_pickle(str(stats_path))
    return stats

def eval_model(use_pickled_stats=False):
    """
    Simple test to Cranfield to see basic metrics.
    """
    base_path = (Path(__file__) / "..").resolve() / "cranfield_corpus"
    relevance_threshold = 0
    stats_path = (base_path / ".." / "stats.df").resolve()
    
    if use_pickled_stats and stats_path.exists():
        stats = get_pickled_stats()
        print_stats_info(stats)
        return

    start = time.time()
    model = VectorialModel(base_path)
    model.build()
    print("Build Time:", time.time() - start, "seconds")

    dataset = ir.load("cranfield")
    queries = { query_id: text for query_id, text in dataset.queries_iter() }
    qrels_df = get_qrels_dataframe()

    stats = pd.DataFrame(columns=["query_id", "recall", "precision", "f1", "rank_threshold"])

    for query_id, qrel in qrels_df.groupby(["query_id"]):
        query = queries.get(query_id)
        if not query: # The querys in some relations are missing
            continue
        # Relevance
        # -1: Not relevant
        # 1: Minimun interest
        # 2: Useful references
        # 3: High degree of relevance
        # 4: Complete answer to the question

        relevant_docs = qrel[qrel["relevance"] >= relevance_threshold]
        # non_relevant_docs = qrel[qrel["relevance"] < relevance_threshold]

        query_rank = model.resolve_query(queries[query_id])
        query_rec_docs = [Path(doc['dir']).name[:-4] for _, doc in query_rank]
        for total_rank_to_check in [30, 50, 100]:
            rec_docs = query_rec_docs[:total_rank_to_check]

            rec_rel_docs = relevant_docs[relevant_docs["doc_id"].isin(rec_docs)]
            prec = len(rec_rel_docs)/len(rec_docs) if rec_docs else nan
            rec = len(rec_rel_docs)/len(relevant_docs) if not relevant_docs.empty else nan
            f1 = 2 * prec * rec/(prec + rec if prec + rec != 0 else 1)
            stats = stats.append({
                "query_id": query_id, 
                "recall": rec, 
                "precision": prec, 
                "f1": f1,
                "rank_threshold": total_rank_to_check,
            }, ignore_index=True)

    print("Test Time:", time.time() - start, "seconds")
    stats.to_pickle(str(stats_path))
    print_stats_info(stats)

def print_stats_info(stats: pd.DataFrame):
    """
    Given a DataFrame with metrics (recall, precision, f1) print the
    results grouped by `rank_threshold`
    """
    clean_nan_stats = stats.dropna(axis=0)

    print("Nans", len(stats) - len(clean_nan_stats))

    for rank_threshold, clean_stats in clean_nan_stats.groupby("rank_threshold"):
        print("Rank Threshold", rank_threshold)
        print("Recall mean", clean_stats["recall"].mean())
        print("Precision mean", clean_stats["precision"].mean())
        print("F1 mean", clean_stats["f1"].mean())
        print()

eval_model()