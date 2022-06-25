
from itertools import count
from pkgutil import get_data
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

  
def get_words_dict(contex:dict):
    words = {""}
    
    docs = contex["documents"]
    
    for doc in docs:
        text_split = doc["text"].split()
        words = words.union(set(text_split))
    
    words_dict = {}
    for i,word in enumerate(words):
        words_dict[word] = i
    
    contex["words_dict"] = words_dict
    
    return contex
  
def generate_sparce_matrix(contex:dict):
    
    docs = contex["documents"]
    dict = contex["words_dict"]
    
    sparse_matrix = csr_matrix((SIZE, SIZE), dtype = np.int8)
    count = 0
    for doc in docs:
        text_split = doc["text"].split()
        for i in range(len(text_split)-1):
            sparse_matrix[dict[text_split[i]], dict[text_split[i+1]]] += 1
        count +=1 
        print(str(count) + " documentos procesados ")
            
    contex["sparse_matrix"] = sparse_matrix
    save_object(["sparse_matrix"],sparse_matrix, "sparse_matrix")
    print (sparse_matrix)
    return contex
            
            
    
context = generate_data()
context["words_dict"] = get_object(["Dict of words on the Corpus"],"words_dict")
context = generate_sparce_matrix(context)
    

     



# words_dict = get_object(["Dict of words on the Corpus"], "words_dict")


