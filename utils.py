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
        cat_map : mapping of all categories found in data dir
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

    if not pap_categories:
        pap_categories = list(cat_map.keys())

    for pap in papers:
        papDict = json.loads(pap)
        category = papDict.get('categories')
        
        try:
            try:
                year = int(papDict.get('journal-ref')[-4:])    # Example Format: "Phys.Rev.D76:013009,2007"
            except:
                year = int(papDict.get('journal-ref')[-5:-1])    # Example Format: "Phys.Rev.D76:013009,(2007)"

            if category in pap_categories and year >= min_year:
                ids.append(papDict.get('id'))
                titles.append(papDict.get('title'))
                abstracts.append(papDict.get('abstract'))
                categories.append(papDict.get('categories'))
        except:
            pass

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

    df['abstract'] = df['abstract'].apply(lambda x: x.replace("\n", ""))
    df['abstract'] = df['abstract'].apply(lambda x: x.strip())
    df['text'] = df['title'] + '. ' + df['abstract']

    df['categories'] = df['categories'].apply(lambda x: tuple(x.split()))
    catcount = df['categories'].value_counts()
    relevant_cats = catcount[catcount > 250].index.tolist()

    df = df[df['categories'].isin(relevant_cats)].reset_index(drop=True)

    if save_pqt:
        df['categories'] = df['categories'].apply(lambda x: list(x))
        df.to_parquet('./data/arxiv_processed')

    mlb = MultiLabelBinarizer()
    mlb.fit(df['categories'])
    df['category_encoding'] = df['categories'].apply(lambda x: mlb.transform([x])[0])

    df = df[['text', 'categories', 'category_encoding']]

    return df, mlb.classes_


class Data_Processor(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.len = len(df)
        self.data = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        text = str(self.data.text[index])
        text = ' '.join(text.split())
        # tokenize text w/ pretrained tokenizer
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        targets = self.data.category_encoding[index]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(targets, dtype=torch.long)
        }

    def __len__(self):
        return self.len


def retrieve_arXiv_embeddings():
    with open('./data/embeddings', 'rb') as f:
        embeddings = pickle.load(f)

    return embeddings

def retrieve_model_classes():
    with open('./data/class_array', 'rb') as f:
        class_array = pickle.load(f)
    
    return class_array


def retrieve_arXiv_link(papID):
    """
    Takes arXiv paper ID and returns link
    """
    return f'https://arxiv.org/pdf/{papID}'

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
