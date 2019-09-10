import glob
import json
import pandas as pd
import nltk
import whoosh.index
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import StandardAnalyzer
from tqdm import tqdm

WHOOSH_DIR = 'whoosh_idx'
INDEX_NAME = 'nasdaq'

NER_STOPWORDS = {'ceo'} | whoosh.analysis.STOP_WORDS


def read_json_articles_to_df(json_glob='data/articles/*.json'):
    article_list = []
    for p in glob.glob(json_glob):
        with open(p, 'r') as f:
            article_list.append(json.load(f))

    articles_df = pd.DataFrame(article_list)

    full_df = articles_df.copy()
    full_df['text'] = full_df['text'].apply(lambda x: '\n\n'.join(x))
    full_df = full_df.drop_duplicates().dropna()

    return full_df


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


if __name__ == '__main__':
    text_df = read_json_articles_to_df()

    if not whoosh.index.exists_in(WHOOSH_DIR):
        schema = Schema(
            url=ID(stored=True),
            published_datetime=ID(stored=True),
            title=TEXT(stored=True, analyzer=StandardAnalyzer()),
            article=TEXT(stored=True, analyzer=StandardAnalyzer()),
            title_named_entities=TEXT(analyzer=StandardAnalyzer(stoplist=NER_STOPWORDS)),
            article_named_entities=TEXT(analyzer=StandardAnalyzer(stoplist=NER_STOPWORDS)),
        )

        idx = whoosh.index.create_in(WHOOSH_DIR, schema=schema, indexname=INDEX_NAME)
    else:
        idx = whoosh.index.open_dir(WHOOSH_DIR, indexname=INDEX_NAME)

    writer = idx.writer()

    for i, row in tqdm(text_df.iterrows(), total=text_df.shape[0]):
        title_ner_text = nltk_ner_extract(row['title'])
        article_ner_text = nltk_ner_extract(row['text'])

        writer.add_document(
            url=row['url'],
            published_datetime=row['published_datetime'],
            title=row['title'],
            article=row['text'],
            title_named_entities=title_ner_text,
            article_named_entities=article_ner_text,
        )

    writer.commit()
