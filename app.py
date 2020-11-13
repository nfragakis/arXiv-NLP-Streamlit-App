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

df = get_data(1000000)
embeddings = get_embeddings()

st.header('Enter your question or topic you would like to learn more about')
st.text_input('What would you like to learn next?')
