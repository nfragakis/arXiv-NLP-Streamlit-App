import argparse
import pickle
from sentence_transformers import SentenceTransformer
from utils import data_to_df

if __name__ == '__main__':
    """
    Program downloads arXiv data from file, generates n word
    embeddings and saves dictionary of id, embedding pairs
    in a serialized pickle file
    """

    # Take command line arguments for number of embedding to generate
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1000)
    args = parser.parse_args()

    n = args.n

    # Download arXiv data in pandas dataframe
    df = data_to_df(n)

    # Import pre-trained BERT embedding generator
    model = SentenceTransformer('bert-base-nli-mean-tokens')

    # Generate embeddings for all abstracts in dataset
    embeddings = model.encode(df['abstract'])

    # Save embeddings in dictionary w/ id as key
    embedDict = dict(zip(df['id'], embeddings))

    outfile = open('./data/embeddings', 'wb')
    pickle.dump(embedDict, outfile)
    outfile.close()

    # Save model for future decoding
    model.save('./models/bert-base-nli-mean-tokens')
