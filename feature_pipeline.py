import json
import glob
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.base import TransformerMixin
from annoy import AnnoyIndex

ARTICLE_DF_PATH = 'data/articles.csv'
PREPROCESSING_PIPELINE_PATH = 'data/preprocessing_pipeline.pickle'
ANNOY_PATH = 'data/annoy_idx.ann'

# From Annoy docs:
#     Works better if you don't have too many dimensions (like <100)
#     but seems to perform surprisingly well even up to 1,000 dimensions
N_DIM = 512
N_TREES = 50


def read_json_articles_to_df(json_glob='data/articles/*.json'):
    article_list = []
    for p in glob.glob(json_glob):
        with open(p, 'r') as f:
            article_list.append(json.load(f))

    articles_df = pd.DataFrame(article_list)

    # TODO: don't duplicate full_text like this
    articles_df['full_text'] = articles_df['text'].apply(lambda x: '\n\n'.join(x))
    articles_df = articles_df.explode('text')
    articles_df = articles_df.drop_duplicates().dropna()

    return articles_df


# noinspection PyUnusedLocal
class DenseTransformer(TransformerMixin):
    def fit(self, x, y=None, **fit_params):
        return self

    @staticmethod
    def transform(x, y=None, **fit_params):
        return x.todense()


if __name__ == '__main__':
    article_df = read_json_articles_to_df()

    preprocessing_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('to_dense', DenseTransformer()),
        ('pca', PCA(n_components=N_DIM, random_state=42)),
    ], verbose=True)

    features = preprocessing_pipeline.fit_transform(article_df['text'])
    annoy_idx = AnnoyIndex(N_DIM, 'euclidean')

    for i, record in enumerate(features):
        annoy_idx.add_item(i, record)

    annoy_idx.build(N_TREES)

    annoy_idx.save(ANNOY_PATH)
    article_df.to_csv(ARTICLE_DF_PATH, index=False)
    with open(PREPROCESSING_PIPELINE_PATH, 'wb') as file:
        pickle.dump(preprocessing_pipeline, file, protocol=pickle.HIGHEST_PROTOCOL)
