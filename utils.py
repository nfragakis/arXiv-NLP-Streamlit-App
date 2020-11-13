import torch
import json
import pickle
import pandas as pd
import numpy as np
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


def data_to_df(n):
    """
    inputs:
        n : total files to download
    takes arXiv json file and returns
    pandas df with appropriate columns
    """

    papers = get_data('data/arxiv-metadata-oai-snapshot.json')

    ids = []
    titles = []
    abstracts = []
    categories = []

    for _ in range(n):
        try:
            papDict = json.loads(next(papers))

            ids.append(papDict['id'])
            titles.append(papDict['title'])
            abstracts.append(papDict['abstract'])
            categories.append(papDict['categories'])

        except:
            pass

    df = pd.DataFrame({'id': ids,
                       'title': titles,
                       'abstract': abstracts,
                       'category': categories})

    return df


class Data_Processor(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.len = len(df)
        self.data = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mlb = MultiLabelBinarizer()
        self.targets = self.mlb.fit_transform(df['category'].apply(lambda x: x.split()))

    def __getitem__(self, index):
        # get title of paper
        title = str(self.data.title[index])
        title = " ".join(title.split())

        # tokenize text w/ pretrained tokenizer
        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        # Encode Targets
        targets = self.targets[index]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(targets, dtype=torch.long)
        }

    def __len__(self):
        return self.len

    def classes_(self):
        "Returns classes from label binarizer"
        return self.mlb.classes_


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
