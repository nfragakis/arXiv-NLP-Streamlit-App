import torch
import json
import pickle
import pandas as pd
import numpy as np
from data.categories import cat_map
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer

def get_data(data_file):
    """
    inputs:
        data_file : JSON file

    uses generator function to loop through json file
    to avoid memory issues if loaded direct
    """

    with open(data_file, 'r') as f:
        for line in f:
            yield line

def data_to_df(pap_categories=[], min_year=2010):
    """
    inputs:
        pap_categories : specify categories to download (default is all)
        min_year : minimum year of publication
    takes arXiv json file and returns
    pandas df with appropriate columns
    """

    papers = get_data('data/arxiv-metadata-oai-snapshot.json')

    ids = []
    titles = []
    abstracts = []
    categories = []

    # Use all categories if non specified
    if not pap_categories:
        pap_categories = list(cat_map.keys())

    for pap in papers:
        papDict = json.loads(pap)
        category = papDict.get('categories')
        
        # Get Year of publication
        try:
            try:
                year = int(papDict.get('journal-ref')[-4:])    # Example Format: "Phys.Rev.D76:013009,2007"
            except:
                year = int(papDict.get('journal-ref')[-5:-1])    # Example Format: "Phys.Rev.D76:013009,(2007)"

            # Only save items if they're in specified categores and above min_year
            if category in pap_categories and year >= min_year:
                ids.append(papDict.get('id'))
                titles.append(papDict.get('title'))
                abstracts.append(papDict.get('abstract'))
                categories.append(papDict.get('categories'))
        except:
            pass

    # Save as pandas dataframe
    df = pd.DataFrame({'id': ids,
                       'title': titles,
                       'abstract': abstracts,
                       'categories': categories})
    return df


def preprocess_data(df, save_pqt=False):
    """
    input: Pandas DF

    - Removes irrelevant characters in abstract
    - Combines the title and abstract into one text field
    - Removes categories w/ less than 250 occurences
    - Option to save .pqt file of text for application
    - Transforms category field into binary vector using
        sklearn MultiLabelBinarizer

    output: Processed DF and MultiLabelBinarizer Class Array
    """

    # Split Multi-Label categories into tuple 
    df['categories'] = df['categories'].apply(lambda x: tuple(x.split()))
    catcount = df['categories'].value_counts()
    relevant_cats = catcount[catcount > 250].index.tolist()

    # Remove all items from categories w/ less than 250 entries
    df = df[df['categories'].isin(relevant_cats)].reset_index(drop=True)

    # Save intermediatet .pqt file
    if save_pqt:
        df.to_parquet('./data/arxiv_processed.pqt')
    
    # Replace \n characters and combine title and abstract
    df['abstract'] = df['abstract'].apply(lambda x: x.replace("\n", ""))
    df['abstract'] = df['abstract'].apply(lambda x: x.strip())
    df['text'] = df['title'] + '. ' + df['abstract']

    # Create one-hot encoding of category
    mlb = MultiLabelBinarizer()
    mlb.fit(df['categories'])
    df['category_encoding'] = df['categories'].apply(lambda x: mlb.transform([x])[0])

    df = df[['text', 'categories', 'category_encoding']]

    return df, mlb.classes_


def retrieve_model_classes():
    """
    Retrieve model class file used in training classifier
    """
    with open('./data/class_array', 'rb') as f:
        class_array = pickle.load(f)
    
    return class_array


def retrieve_arXiv_link(papID):
    """
    Takes arXiv paper ID and returns link
    """
    return f'https://arxiv.org/pdf/{papID}'

def cosine(u, v):
    "Compute cosine similarity of two vectors"
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


