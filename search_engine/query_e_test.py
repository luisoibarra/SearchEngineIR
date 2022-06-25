
from itertools import count
from pkgutil import get_data
from unittest import result
from search_logic.ir_models.base import *
from pathlib import Path
import numpy as np
from search_logic.ir_models.utils import save_object,get_object
import numpy as np
from scipy.sparse import csr_matrix
  
SIZE = 377776
  
  
def generate_data():
    context={}
    context["corpus_address"] = Path(__file__) / ".." / "search_logic" / "corpus"
    return read_documents_from_hard_drive(context)

  
def generate_words_dict(contex:dict):
    words = {""}
    
    docs = contex["documents"]
    
    for doc in docs:
        text_split = doc["text"].split()
        words = words.union(set(text_split))
    
    words_dict = {}
    for i,word in enumerate(words):
        words_dict[word] = i
    
    word_list = list(words)
    
    contex["words_dict"] = words_dict
    context["words_list"] = word_list
    save_object(["words_dict"],words_dict, "words_dict")
    save_object (["words_list"],word_list, "words_list")
    
    return contex
  
def generate_sparse_matrix(contex:dict):
    
    docs = contex["documents"]
    words_dict = contex["words_dict"]
   
    
    
    sparse_matrix = csr_matrix((SIZE, SIZE), dtype = np.int8)
    count = 0
    for doc in docs:
        text_split = doc["text"].split()
        for i in range(len(text_split)-1):
            sparse_matrix[words_dict[text_split[i]], words_dict[text_split[i+1]]] += 1
        count +=1 
        print(str(count) + " documentos procesados ")
            
    contex["sparse_matrix"] = sparse_matrix
    save_object(["sparse_matrix"],sparse_matrix, "sparse_matrix")
   
    return contex

def get_expand_query(contex:dict,word:str):
    dict = contex["words_dict"]
    words_list = context["words_list"]
    sparse_m = contex["sparse_matrix"]
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
          
            
    
context = generate_data()
context["words_dict"] = get_object(["words_dict"],"words_dict")
context["sparse_matrix"] = get_object(["sparse_matrix"],"sparse_matrix")
context["words_list"] = get_object(["words_list"],"words_list")

list = get_expand_query(context,"air")
print (context["sparse_matrix"])
    

     



# words_dict = get_object(["Dict of words on the Corpus"], "words_dict")


