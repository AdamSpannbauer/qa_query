import json
import pandas as pd
import whoosh.index
from whoosh.qparser import QueryParser, MultifieldParser
from build_whoosh_index import WHOOSH_DIR, INDEX_NAME, nltk_ner_extract
from bert_squad import QABot

N_SEARCH_RESULTS = 20
N_ANSWERS = 3
CONTEXT_PADDING = 30
SEARCH_TITLE = True

qa_bot = QABot(download=False)
idx = whoosh.index.open_dir(WHOOSH_DIR, indexname=INDEX_NAME)

if SEARCH_TITLE:
    ne_query_parser = MultifieldParser(['title_named_entities', 'article_named_entities'],
                                       schema=idx.schema,
                                       group=whoosh.qparser.OrGroup)
    query_parser = MultifieldParser(['title', 'article'],
                                    schema=idx.schema,
                                    group=whoosh.qparser.OrGroup)
else:
    ne_query_parser = QueryParser('article_named_entities',
                                  schema=idx.schema,
                                  group=whoosh.qparser.OrGroup)
    query_parser = QueryParser('article',
                               schema=idx.schema,
                               group=whoosh.qparser.OrGroup)

with idx.searcher() as searcher:
    while True:
        question = input('("exit" to quit) Question: ')
        if question.lower() == 'exit':
            break

        question_ne_text = nltk_ner_extract(question)
        if question_ne_text:
            ne_query = ne_query_parser.parse(question_ne_text)
            ne_search_results = searcher.search(ne_query, limit=N_SEARCH_RESULTS // 2)
            n_ne_search_results = ne_search_results.scored_length()
        else:
            n_ne_search_results = 0

        n_search_results = N_SEARCH_RESULTS - n_ne_search_results
        query = query_parser.parse(question)
        search_results = searcher.search(query, limit=n_search_results)

        if n_ne_search_results:
            search_results.upgrade_and_extend(ne_search_results)

        answers = []
        for search_result in search_results:
            full_answer = qa_bot.ask_question(search_result['article'], question)

            answer_start_idx = full_answer[1][0]
            start = max([0, answer_start_idx - CONTEXT_PADDING])
            end = min([len(search_result['article']), answer_start_idx + CONTEXT_PADDING])
            answer_context = '...' + search_result['article'][start:end] + '...'

            answers.append({
                'question': question,
                'answer_text': full_answer[0][0],
                'answer_start_idx': answer_start_idx,
                'answer_context': answer_context,
                'answer_confidence': full_answer[2][0],
                'source_text': search_result['article'],
                'source_url': search_result['url'],
                'search_rank': float(len(answers) + 1),
            })

        answer_df = pd.DataFrame(answers)
        answer_df['squad_rank'] = answer_df['answer_confidence'].rank(ascending=False)

        answer_list = answer_df[['question', 'answer_text', 'answer_context',
                                 'answer_confidence', 'search_rank', 'squad_rank',
                                 'source_text', 'source_url']].to_dict('records')

        for i, answer in enumerate(answer_list[:N_ANSWERS]):
            print(json.dumps(answer, indent=2))

        answer_df = answer_df.sort_values('squad_rank')
        answer_list = answer_df[['question', 'answer_text', 'answer_context',
                                 'answer_confidence', 'search_rank', 'squad_rank',
                                 'source_text', 'source_url']].to_dict('records')
        for i, answer in enumerate(answer_list[:N_ANSWERS]):
            print(json.dumps(answer, indent=2))
