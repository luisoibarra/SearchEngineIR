import numpy as np

def add_feedback_manager(context: dict) -> dict:
    """
    Add the feedback maanger used by the IR model
    """
    manager = FeedbackManager()
    context["feedback_manager"] = manager
    manager.build(context)

    return context

def add_feedback_vectors(context: dict):
    """
    Adds the `new_relevant_documents` and the `new_not_relevant_documents` to
    the `feedback_manager` associated with `query`
    """
    feedback_manager = context.get("feedback_manager")
    if feedback_manager:
        query = context["query"]
        new_relevant_documents = context.get("new_relevant_documents", [])
        new_not_relevant_documents = context.get("new_not_relevant_documents", [])
        for rel in new_relevant_documents:
            feedback_manager.mark_relevant(query, rel)
        for not_rel in new_not_relevant_documents:
            feedback_manager.mark_not_relevant(query, not_rel)
        print("Feedback vectors added")
    return context

def add_feedback_to_query(context: dict):
    """
    Applies the Rocchio Algorithm to the given `query` if
    `feedback_manager` is given
    """
    feedback_manager: FeedbackManager = context.get("feedback_manager")
    if not feedback_manager:
        return context
    query = context["query"]
    alpha = context.get("alpha_rocchio", 1)
    beta = context.get("beta_rocchio", 0.75)
    ro = context.get("ro_rocchio", 0.1)

    relevants = feedback_manager.get_relevants(query)
    not_relevants = feedback_manager.get_not_relevants(query)

    def vec_mean(vectors, query):
        if not vectors:
            return np.zeros_like(query)
        s = sum(vectors)
        return s/len(vectors)

    # Apply Rocchio algorithm
    feedback_vec = alpha * query["vector"] + beta * vec_mean(relevants, query["vector"]) - ro * vec_mean(not_relevants, query["vector"])
    feedback_vec = np.array([max(0,x) for x in feedback_vec])
    query["vector"] = feedback_vec

    return context


class FeedbackManager:
    """
    Base class to manage the relevant and not relevant documents for a given query 
    """

    def __init__(self) -> None:
        self.relevant_dict = {}
        self.not_relevant_dict = {}

    def build(self, context: dict):
        """
        Initialize the manager
        """
        return

    def _mark_document(self, query: dict, document: dict, relevant_dict: dict):

        # Adds the document in a set with all relevant or not relevant documents of the query
        query = tuple(query['vector'])
        document = tuple(document['vector'])
        if query in relevant_dict:
            relevant_dict[query].update([document])
        else:
            relevant_dict[query] = set([document])

    def mark_relevant(self, query: dict, document: dict):
        """
        Mark the document as relevant to the query
        """
        self._mark_document(query, document, self.relevant_dict)
    
    def mark_not_relevant(self, query: dict, document: dict):
        """
        Mark the document as not relevant to the query
        """
        self._mark_document(query, document, self.not_relevant_dict)

    def _get_relevants(self, query: dict, relevant_dict: dict):
        try:
            return [np.array(x) for x in relevant_dict[tuple(query['vector'])]]
        except KeyError:
            return []

    def get_relevants(self, query: dict):
        """
        Return the list of relevant documents given the query
        """
        return self._get_relevants(query, self.relevant_dict)

    def get_not_relevants(self, query: dict):
        """
        Return the list of non relevant documents given the query
        """
        return self._get_relevants(query, self.not_relevant_dict)

