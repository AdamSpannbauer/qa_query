import json
import glob
import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.base import TransformerMixin
import nltk
from annoy import AnnoyIndex

ARTICLE_DF_PATH = 'data/articles.csv'
PREPROCESSING_PIPELINE_PATH = 'data/preprocessing_pipeline.pickle'
ANNOY_PATH = 'data/annoy_idx.ann'

# From Annoy docs:
#     Works better if you don't have too many dimensions (like <100)
#     but seems to perform surprisingly well even up to 1,000 dimensions
N_DIM = 1000
N_TREES = 50


def nltk_ner_extract(text, keep_labels=None, join=True):
    if keep_labels is None:
        keep_labels = ['ORGANIZATION', 'LOCATION', 'FACILITY', 'GPE']

    chunks = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text)))
    ne_list = []
    for chunk in chunks:
        if hasattr(chunk, 'label'):
            if chunk.label() in keep_labels:
                full_ne = ' '.join(c[0] for c in chunk)
                ne_list.append(full_ne)

    if join:
        return ' '.join(ne_list)
    else:
        return ne_list


def read_json_articles_to_df(json_glob='data/articles/*.json'):
    article_list = []
    for p in glob.glob(json_glob):
        with open(p, 'r') as f:
            article_list.append(json.load(f))

    articles_df = pd.DataFrame(article_list)

    full_df = articles_df.copy()
    full_df['text'] = full_df['text'].apply(lambda x: '\n\n'.join(x))
    full_df = full_df.drop_duplicates().dropna()

    section_df = articles_df.explode('text')
    section_df = section_df.drop_duplicates().dropna()

    return section_df, full_df


# noinspection PyUnusedLocal
class DenseTransformer(TransformerMixin):
    def fit(self, x, y=None, **fit_params):
        return self

    @staticmethod
    def transform(x, y=None, **fit_params):
        return x.todense()


if __name__ == '__main__':
    section_text_df, full_text_df = read_json_articles_to_df()
    ne_text = []
    for t in tqdm(full_text_df['text']):
        ne_text.append(nltk_ner_extract(t))
    full_text_df['ne_text'] = ne_text

    section_text_df.to_csv(ARTICLE_DF_PATH, index=False)
    full_text_df.to_csv('data/ne_full_text_df.csv', index=False)

    preprocessing_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('to_dense', DenseTransformer()),
        ('pca', PCA(n_components=N_DIM, random_state=42)),
    ], verbose=True)

    features = preprocessing_pipeline.fit_transform(section_text_df['text'])
    annoy_idx = AnnoyIndex(N_DIM, 'euclidean')

    for i, record in enumerate(features):
        annoy_idx.add_item(i, record)

    annoy_idx.build(N_TREES)

    annoy_idx.save(ANNOY_PATH)
    with open(PREPROCESSING_PIPELINE_PATH, 'wb') as file:
        pickle.dump(preprocessing_pipeline, file, protocol=pickle.HIGHEST_PROTOCOL)
