import pickle
import pandas as pd
from annoy import AnnoyIndex
from bert_squad import QABot
from feature_pipeline import DenseTransformer  # Needs to be in memory to load pipeline

ARTICLE_DF_PATH = 'data/articles.csv'
PREPROCESSING_PIPELINE_PATH = 'data/preprocessing_pipeline.pickle'
ANNOY_PATH = 'data/annoy_idx.ann'
N_DIM = 100
N_ANSWERS = 5

article_df = pd.read_csv(ARTICLE_DF_PATH)

with open(PREPROCESSING_PIPELINE_PATH, 'rb') as f:
    preprocessing_pipeline = pickle.load(f)

annoy_idx = AnnoyIndex(N_DIM, 'euclidean')
annoy_idx.load(ANNOY_PATH)

qa_bot = QABot()

while True:
    question = input("('exit' to quit) Question: ")
    if question.lower() == 'exit':
        break

    x = preprocessing_pipeline.transform([question])[0]
    doc_idxs = annoy_idx.get_nns_by_vector(x, N_ANSWERS)

    for doc_idx in doc_idxs:
        row = article_df.iloc[doc_idx, :]
        answer = qa_bot.ask_question(row["text"], question)
        print(f'\t* {answer} (Source: {row["url"]})')
