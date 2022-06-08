import hashlib
import pickle
from typing import List
from pathlib import Path

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