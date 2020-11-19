from transformers import DistilBertTokenizer
from data.categories import cat_map
from sentence_transformers import SentenceTransformer
from utils import retrieve_model_classes, retrieve_arXiv_link, cosine
import torch
import argparse
import time
import numpy as np
import pandas as pd

if __name__ == "__main__":
    # Instantiate argument parser to take cmd line query
    parser = argparse.ArgumentParser()
    parser.add_argument('--q', type=str, default='What is Dark Matter?')
    args = parser.parse_args()
    
    # Store cmd line query
    query = args.q

    # Download data and embeddings
    df = pd.read_parquet('./data/arxiv_processed.pqt')
    embeddings = np.load('./data/embeddings.npy')

    # Instantiate query tokenizer, classifier, and embeddings transformer
    tokenizer = DistilBertTokenizer.from_pretrained('./models/vocab_distilbert_arxiv.bin')
    classifier = torch.load('./models/pytorch_distilbert_arxiv.bin')
    sentence_transformer = SentenceTransformer('./models/distilbert-base-nli-stsb-mean-tokens')

    # Retrieve model classes from file
    model_classes = retrieve_model_classes()

    # Encode input query and predict top 3 topics
    query_tokens = tokenizer.encode_plus(
        query,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        return_token_type_ids=True)
    
    ids = torch.tensor(query_tokens['input_ids']).unsqueeze(0)
    mask = torch.tensor(query_tokens['attention_mask']).unsqueeze(0)

    pred = classifier(ids, mask)
    top_val, top_idx = torch.topk(pred[0], 3, dim=1)
    pred_categories = model_classes[top_idx].tolist()[0]
    topics = [cat_map[cat] for cat in pred_categories]

    ### Encode query to embedding and return top item

    # encode user query 
    query_embedding = sentence_transformer.encode(query)

    # filter df to relevant categories and grab embeddings
    relevant_docs = df[df['categories'].isin(pred_categories)]
    relevant_embeddings = embeddings[relevant_docs.index]

    # Calculate cosine similarity of user query and relevant embeddings
    sims = []
    for doc in relevant_embeddings:
        doc_sim = cosine(query_embedding, doc)
        sims.append(doc_sim)

    top_matches = np.argsort(sims)[::-1]
    top_item = relevant_docs.iloc[top_matches[0]]
    

    print(f'User Query: {query}\n')
    print(f'Predicted Topics {topics}\n')
    print(f'Recomended Paper:\n {top_item.title}\n')
    print(f'Abstract:\n{top_item.abstract}\n')
