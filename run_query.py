import pickle
import json
import pandas as pd
from annoy import AnnoyIndex
from bert_squad import QABot
import feature_pipeline as fp
# Needs to be in memory to load pipeline
from feature_pipeline import DenseTransformer

N_ANSWERS = 5

section_text_df = pd.read_csv(fp.ARTICLE_SECTION_DF_PATH)
full_text_df = pd.read_csv(fp.ARTICLE_FULL_DF_PATH)
ne_df = pd.read_csv(fp.ARTICLE_NE_DF_PATH)
ne_df['ne_text'] = ne_df['ne_text'].str.lower()

with open(fp.PREPROCESSING_PIPELINE_PATH, 'rb') as f:
    preprocessing_pipeline = pickle.load(f)

annoy_idx = AnnoyIndex(fp.N_DIM, 'euclidean')
annoy_idx.load(fp.ANNOY_PATH)

qa_bot = QABot(download=False)

while True:
    question = input("('exit' to quit) Question: ")
    if question.lower() == 'exit':
        break

    query_nes = fp.nltk_ner_extract(question)
    query_nes = [ne.lower() for ne in query_nes]

    if query_nes:
        ne_match_records = ne_df[ne_df['ne_text'].isin(query_nes)] \
            .groupby(['url', 'title', 'published_datetime']) \
            .agg(uniq_ne_count=('count', 'size'),
                 ne_count=('count', 'sum')) \
            .reset_index() \
            .sort_values(['uniq_ne_count', 'ne_count'], ascending=False)
        urls = ne_match_records['url']
    else:
        x = preprocessing_pipeline.transform([question])[0]
        doc_idxs = annoy_idx.get_nns_by_vector(x, N_ANSWERS)
        urls = full_text_df.iloc[doc_idxs, :]['url']

    max_urls = 10
    urls = urls[:max_urls]
    answers = []
    for url in urls:
        row = full_text_df[full_text_df['url'] == url].iloc[0]
        full_answer = qa_bot.ask_question(row['text'], question)

        answer_start_idx = full_answer[1][0]
        start = max([0, answer_start_idx - 20])
        end = min([len(row['text']), answer_start_idx + 20])
        answer_context = '...' + row['text'][start:end] + '...'

        answers.append({
            'answer_text': full_answer[0][0],
            'answer_start_idx': answer_start_idx,
            'answer_context': answer_context,
            'answer_confidence': full_answer[2][0],
            'source_text': row['text'],
            'source_url': row['url'],
        })

    answer_df = pd.DataFrame(answers)
    answer_df = answer_df.sort_values('answer_confidence', ascending=False)

    answer_list = answer_df[['answer_text', 'answer_context',
                             'answer_confidence', 'source_text',
                             'source_url']].to_dict('records')

    for answer in answer_list[:2]:
        print(json.dumps(answer, indent=2))
