import ir_datasets as ir
from pathlib import Path

def create_local_cranfield_corpus():
    """
    Create a local Cranfield corpus. The names are the id of the files
    """

    base_path = Path(__file__) / ".."
    base_path = base_path.resolve()
    
    dataset = ir.load("cranfield")
    for doc in dataset.docs_iter():
        doc_id, title, text, author, bib = doc
        path = path / "cranfield_corpus" / f"{doc_id}.txt"
        path.touch() # Creates the file if doesn't exist
        with path.open("w") as file:
            file.write(text)

