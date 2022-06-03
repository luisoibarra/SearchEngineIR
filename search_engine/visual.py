from pathlib import Path
import streamlit as st
from models.models import Document, FeedbackModel
from api_model import model, apply_feedback_to_model, get_document_content, get_documents

# base_path = (Path(__file__) / "..").resolve()

# def get_model():
#     if 'model' in st.session_state:
#         return st.session_state["model"]
#     else:
#         model = VectorialModel(base_path / "test" / "cranfield_corpus")
#         model.build()
#         st.session_state['model'] = model
#         return model

# stats = get_pickled_stats()

# stats

# Initialization
if 'document_amount' not in st.session_state:
    st.session_state['document_amount'] = 30

# Body

st.header("Search Engine")

query = st.text_input("Query")

with st.expander("Text" if 'text_name' not in st.session_state else st.session_state['text_name']):
    text = st.empty()

if 'text' in st.session_state:
    text.text(st.session_state['text'])

# Callbacks

def on_button_click(doc: Document):
    content = get_document_content(doc.documentDir) # doc['text']
    st.session_state['text'] = content
    st.session_state['text_name'] = doc.documentName
    text.text(content)

def on_mark_relevant(doc: Document, query: str, relevant: bool):
    if relevant:
        apply_feedback_to_model(FeedbackModel(query=query, relevants=[doc.documentDir], not_relevants=[]))
        # model.add_relevant_and_not_relevant_documents(query, [doc], [])
    else:
        apply_feedback_to_model(FeedbackModel(query=query, relevants=[], not_relevants=[doc.documentDir]))
        # model.add_relevant_and_not_relevant_documents(query, [], [doc])

def on_show_more(more: bool):
    st.session_state["document_amount"] += 30 if more else -30

# Show results

if query:
    ranked_documents = get_documents(query, 0, batch_size=2000).documents # model.resolve_query(query)

    for doc in ranked_documents[:st.session_state["document_amount"]]:
        col1, col2, col3 = st.columns(3)
        col1.button(doc.documentName, key=doc.documentDir+"btn1", on_click=on_button_click, args=(doc,))
        col2.button("Relevant", key=doc.documentDir+"btn2", on_click=on_mark_relevant, args=(doc, query, True))
        col3.button("Not Relevant", key=doc.documentDir+"btn3", on_click=on_mark_relevant, args=(doc, query, False))
    
    col_show1, col_show2 = st.columns(2)
    col_show1.button("Show more", on_click=on_show_more, args=(True,))
    col_show2.button("Show less", on_click=on_show_more, args=(False,))