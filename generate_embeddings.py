from utils import data_to_df, preprocess_data
from data.categories import cat_map
import numpy as np
import argparse
from sentence_transformers import SentenceTransformer

if __name__ == '__main__':
    """
    Program downloads arXiv data from file, generates vector
    embeddings using a pretrained BERT model from SentenceTransformer
    library, then saves vectors as .npy in data directory
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='n')
    args = parser.parse_args()

    # Store cmd line arg
    save = args.save
    save_pqt = True if save == 'y' else False

    # Download arXiv data in pandas dataframe
    df = data_to_df(cat_map, min_year=2010)
    df, _ = preprocess_data(df, save_pqt)

    # Import pre-trained BERT embedding generator
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

    # Generate embeddings for all text in dataset
    embeddings = model.encode(df['text'])

    # Save np array as .npy
    np.save('.data/embeddings.npy', embeddings)

    # Save model for future decoding
    model.save('./models/distilbert-base-nli-stsb-mean-tokens')
