import argparse
import os
import whoosh.index
from whoosh.fields import Schema, TEXT, ID
from tqdm import tqdm
from qa_query.scrape_utils import read_json_articles_to_df
from qa_query.whoosh_utils import QAAnalyzer


ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', default='whoosh_idx',
                help='Name of directory to build whoosh index in')
ap.add_argument('-n', '--name', default='nasdaq',
                help='Name to give whoosh index')
ap.add_argument('-g', '--glob', default='data/articles/*.json',
                help='glob to use to match input json nasdaq articles')
args = vars(ap.parse_args())


# Read in NASDAQ articles to be indexed
text_df = read_json_articles_to_df(args['glob'])

if not os.path.isdir(args['output']):
    os.mkdir(args['output'])

# Create whoosh index (or open if already exists)
if not whoosh.index.exists_in(args['output']):
    schema = Schema(
        url=ID(stored=True, unique=True),
        published_datetime=ID(stored=True),
        title=TEXT(stored=True, analyzer=QAAnalyzer()),
        article=TEXT(stored=True, analyzer=QAAnalyzer()),
        title_named_entities=TEXT(analyzer=QAAnalyzer(ner_tokenize=True)),
        article_named_entities=TEXT(analyzer=QAAnalyzer(ner_tokenize=True)),
    )

    idx = whoosh.index.create_in(args['output'], schema=schema, indexname=args['name'])
else:
    idx = whoosh.index.open_dir(args['output'], indexname=args['name'])

# Write items articles to index
writer = idx.writer()
for i, row in tqdm(text_df.iterrows(), total=text_df.shape[0]):
    writer.update_document(
        url=row['url'],
        published_datetime=row['published_datetime'],
        title=row['title'],
        article=row['text'],
        title_named_entities=row['title'],
        article_named_entities=row['text'],
    )

writer.commit()
