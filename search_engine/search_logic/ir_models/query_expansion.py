

from typing import List

import numpy as np
from scipy.sparse import csr_matrix
from .utils import save_object,get_object




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
    
    expansions = query_expansion_manager.expand_query(context,query)

    context["query_expansions"] = expansions

    return context



class QueryExpansionManager:
    """
    Base class that manages the query expansion
    """

    def __init__(self) -> None:
        pass
    
    
    # def generate_data():
    #     context={}
    #     context["corpus_address"] = Path(__file__) / ".." / "search_logic" / "corpus"
    #     return read_documents_from_hard_drive(context)

  
    def generate_words_dict(self,context:dict):
        
        docs = context["documents"]
        
        words_dict = get_object([doc['dir'] for doc in docs],"words_dict")
        word_list = get_object([doc['dir'] for doc in docs],"words_list")
        
        if words_dict is None:
            words = {""}
           

            for doc in docs:
                text_split =[ i.lower() for i in  doc["text"].split()]
                words = words.union(set(text_split))

            words_dict = {}
            for i,word in enumerate(words):
                words_dict[word] = i

            word_list = list(words)
            save_object([doc['dir'] for doc in docs],words_dict, "words_dict")
            save_object ([doc['dir'] for doc in docs],word_list, "words_list")

        context["words_dict"] = words_dict
        context["words_list"] = word_list
            
        return context
    
    def generate_sparse_matrix(self,context:dict):
        
        docs = context["documents"]
        sparse_matrix = get_object([doc['dir'] for doc in docs],"sparse_matrix")

        if sparse_matrix is None:
            words_dict = context["words_dict"]
            sparse_matrix = csr_matrix((len(words_dict), len(words_dict)), dtype = np.int8)
            count = 0
            for doc in docs:
                text_split = doc["text"].split()
                for i in range(len(text_split)-1):
                    sparse_matrix[words_dict[text_split[i]], words_dict[text_split[i+1]]] += 1
                count +=1 
                print(str(count) + " documentos procesados ")
            
            save_object([doc['dir'] for doc in docs],sparse_matrix, "sparse_matrix")
        
        context["sparse_matrix"] = sparse_matrix
        return context

    def get_expand_query(self,context:dict,word:str):
        dict = context["words_dict"]
        words_list = context["words_list"]
        sparse_m = context["sparse_matrix"]
        if(word in dict):
            word_index = dict[word]
            word_row = sparse_m[word_index]
            non_zero_values = word_row.nonzero()[1]
            rank_words = []
            for val in non_zero_values:
                rank_words.append([word_row[0,val],val])

            rank_words.sort(reverse=True)    

            result = [words_list[i[1]] for i in rank_words]

            return result
        else: 
            return []
          
   
    
    
    
    
    def build(self, context: dict):
        """
        Initialize the manager
        """
        self.generate_words_dict(context)
        self.generate_sparse_matrix(context)
        return

    def expand_query(self,context:dict, query: dict) -> List[str]:
        """
        Returns a rank for the query expansion for the given query
        """
        words = query["text"].split()
        word = words[len(words)-1].lower()
        rank = self.get_expand_query(context,word)
        return [query['text'] + " "+ x for x in rank[:5]] # TODO
