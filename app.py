from transformers import DistilBertTokenizer
from networks.classification import DistillBertClass
from data.categories import cat_map
import sentence_transformers
import json
import numpy as np
import pandas as pd
import streamlit as st
from utils import *

st.title('arXiv Paper Discovery App')
st.markdown('Input a question or topic you would like to find out more about\
             and relevant research papers will be returned')
st.text('The dataset consists of 1,000,000 research papers from arXiv.org')

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

df = get_data(10000)
embeddings = get_embeddings()
tokenizer = DistilBertTokenizer.from_pretrained('./models/vocab_distilbert_arxiv.bin')
classifier = torch.load('./models/pytorch_distilbert_arxiv.bin')

st.header('Enter your question or topic you would like to learn more about')
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

    pred = model(ids, mask)
    top_val, top_idx = torch.topk(pred_categories, 3, dim=1)
    #pred_categories = class_array[top_idx].tolist()
    #pred_topics = [cat_map[cat] for cat in pred_categories]
st.text(top_idx)


