from transformers import DistilBertTokenizer
from networks.classification import DistillBertClass
from data.categories import cat_map
from sentence_transformers import SentenceTransformer
from utils import *
import numpy as np
import pandas as pd
import streamlit as st
import json

st.title('STEM Paper Discovery App')
st.header('Relevant Fields:\n Biology, Computer Science, Economics, Math, Physics, and Statistics')
st.text('The dataset consists of 1,700,000 research papers from arXiv.org')

@st.cache
def get_data(*args):
    """
    Decorated function to retrieve arxiv dataset
    from local directory and cache in pandas df
    """
    return data_to_df(*args)

@st.cache
def get_embeddings():
    """
    Decorated function to retrieve pre-trained
    arXiv embeddings and cache in numpy array
    """
    return retrieve_arXiv_embeddings()

def display_paper(n):
    st.header(df.iloc[top_matches[n]]['title'].replace('\n ', ''))

    link = retrieve_arXiv_link(df.iloc[top_matches[n]]['id'])
    st.markdown(f'**PDF Link:**\n[{link}]({link})')

    st.text(df.iloc[top_matches[n]]['abstract'])



df = get_data(10000)
embeddings = get_embeddings()
tokenizer = DistilBertTokenizer.from_pretrained('./models/vocab_distilbert_arxiv.bin');
classifier = torch.load('./models/pytorch_distilbert_arxiv.bin');

sentence_transformer = SentenceTransformer('./models/bert-base-nli-mean-tokens')
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
    top_val, top_idx = torch.topk(pred, 3, dim=1)
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

    # embeddings
    query_embedding = sentence_transformer.encode(user_query)
    sims = []
    for ind, doc in embeddings.items():
        doc_sim = cosine(query_embedding, doc)
        sims.append(doc_sim)

    top_matches = np.argsort(sims)[::-1]
    st.title('Most relevant papers')

    match = st.selectbox("Recommendation Number", [1, 2, 3], 0)

    display_paper(match-1)

