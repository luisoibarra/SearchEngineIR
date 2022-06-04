

from typing import List


def add_query_expansion_manager(context: dict) -> dict:
    """
    Add the feedback maanger used by the IR model
    """
    manager = QueryExpansionManager()
    context["query_expansion_manager"] = manager
    manager.build(context)

    return context

def add_query_expansions(context: dict) -> dict:
    """
    Adds the `query_expantions` of `query` to the context
    """

    query_expansion_manager: QueryExpansionManager = context.get('query_expansion_manager')
    query = context["query"]
    
    if query_expansion_manager is None:
        context["query_expansions"] = []
        return context
    
    expansions = query_expansion_manager.expand_query(query)

    context["query_expansions"] = expansions

    return context



class QueryExpansionManager:
    """
    Base class that manages the query expansion
    """

    def __init__(self) -> None:
        pass

    def build(self, context: dict):
        """
        Initialize the manager
        """
        return

    def expand_query(self, query: dict) -> List[str]:
        """
        Returns a rank for the query expansion for the given query
        """
        return [query['text'] + "1", query['text'] + "2", query['text'] + "3"] # TODO
