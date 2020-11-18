from transformers import DistilBertTokenizer
from data.categories import cat_map
from sentence_transformers import SentenceTransformer
from utils import retrieve_model_classes, retrieve_arXiv_link, cosine
import torch
import numpy as np
import pandas as pd
import streamlit as st

st.title('STEM Paper Discovery App')
st.header('Relevant Fields:\n Biology, Computer Science, Economics, Math, Physics, and Statistics')
st.text('The dataset consists of 1,700,000 research papers from arXiv.org')

@st.cache
def get_data(*args):
    """
    Decorated function to retrieve arxiv dataset
    from local directory and cache in pandas df
    """
    return pd.read_parquet('./data/arxiv_processed.pqt')

@st.cache
def get_embeddings():
    """
    Decorated function to retrieve pre-trained
    arXiv embeddings and cache in numpy array
    """
    return np.load('./data/embeddings.npy')

def display_paper(item):
    st.header(item.title.replace('\n ', ''))

    link = retrieve_arXiv_link(item.id)
    st.markdown(f'**PDF Link:**\n[{link}]({link})')

    st.markdown(item.abstract)


df = get_data()
embeddings = get_embeddings()
tokenizer = DistilBertTokenizer.from_pretrained('./models/vocab_distilbert_arxiv.bin')
classifier = torch.load('./models/pytorch_distilbert_arxiv.bin')

#sentence_transformer = SentenceTransformer('./models/bert-base-nli-mean-tokens')
sentence_transformer = SentenceTransformer('./models/distilbert-base-nli-stsb-mean-tokens')
model_classes = retrieve_model_classes()

st.header('Enter a question or topic you would like to learn more about')
user_query = st.text_input('What would you like to learn next?')

if user_query:
    query_tokens = tokenizer.encode_plus(
        user_query,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        return_token_type_ids=True)
    
    ids = torch.tensor(query_tokens['input_ids']).unsqueeze(0)
    mask = torch.tensor(query_tokens['attention_mask']).unsqueeze(0)

    pred = classifier(ids, mask)
    top_val, top_idx = torch.topk(pred[0], 3, dim=1)
    pred_categories = model_classes[top_idx].tolist()[0]
    pred_topics = [cat_map[cat] for cat in pred_categories]

    st.subheader('Topics of Interest...')
    top1, top2, top3 = st.beta_columns(3)
    with top1:
        st.button(pred_topics[0])

    with top2:
        st.button(pred_topics[1])

    with top3:
        st.button(pred_topics[2])

    # encode user query 
    query_embedding = sentence_transformer.encode(user_query)

    # grab relevant embeddings based on category prediction
    relevant_docs = df[df['categories'].isin(pred_categories)]
    relevant_embeddings = embeddings[relevant_docs.index]

    sims = []
    for doc in relevant_embeddings:
        doc_sim = cosine(query_embedding, doc)
        sims.append(doc_sim)

    top_matches = np.argsort(sims)[::-1]
    st.title('Most relevant papers')

    match = st.selectbox("Recommendation Number", [1, 2, 3], 0)
    show_item = relevant_docs.iloc[top_matches[match - 1]]

    display_paper(show_item)

