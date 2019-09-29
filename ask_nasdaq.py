# Silence infinite debug/warnings coming from deeppavlov & tf
# nltk uses plain old `print()` so can't silence...
import os
import logging
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.disable(logging.CRITICAL)
warnings.filterwarnings('ignore')

import json
import pandas as pd
import whoosh.index
from whoosh.qparser import MultifieldParser, OrGroup
from qa_query import BertSquad


# Force printout of all df columns in console output
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)


def format_answer(question_text, answer_list, search_result, context_pad_size=10):
    answer_start_idx = answer_list[1][0]
    context_start = max([0, answer_start_idx - context_pad_size])
    context_end = min([len(search_result['article']), answer_start_idx + context_pad_size])
    context = search_result['article'][context_start:context_end]
    formatted_context = '...' + context + '...'

    answer_dict = {
        'question': question_text,
        'answer_text': answer_list[0][0],
        'answer_start_idx': answer_start_idx,
        'answer_context': formatted_context,
        'answer_confidence': answer_list[2][0],
        'source_text': search_result['article'],
        'source_title': search_result['title'],
        'source_url': search_result['url'],
    }

    return answer_dict


def answer_display_df(answer_df, n_answers=3, rank_by='combined_rank', display_fields=None, display_names=None):
    answer_df = answer_df.sort_values(rank_by)

    if display_fields is None:
        display_fields = [
            'answer_text',
            'answer_context',
            'source_title',
            'search_rank',
            'squad_rank',
            'combined_rank',
        ]

    if display_names is None:
        display_names = [
            'Answer',
            'Context',
            'Source Title',
            'Search Rank',
            'Squad Rank',
            'Combined Rank',
        ]

    display_df = answer_df[display_fields]
    display_df.columns = display_names
    display_df = display_df.iloc[:n_answers, :]

    return display_df.reset_index(drop=True)


def qa_query_nasdaq(question_text, whoosh_query_parser, whoosh_searcher, bert_squad_inst, n_search_results=20):
    parsed_query = whoosh_query_parser.parse(question_text)
    search_results = whoosh_searcher.search(parsed_query, limit=n_search_results)

    answers = []
    for i, search_result in enumerate(search_results):
        full_answer = bert_squad_inst.ask_question(search_result['article'], question_text)
        answer = format_answer(question_text, full_answer, search_result)
        answer['search_rank'] = i + 1.0
        answers.append(answer)

    answer_df = pd.DataFrame(answers)
    answer_df['squad_rank'] = answer_df['answer_confidence'].rank(ascending=False)
    answer_df['combined_rank'] = (answer_df['squad_rank'] +
                                  answer_df['search_rank']).rank()

    return answer_df


def print_answers(answer_list, header=None):
    if header is not None:
        print(f'\n---- {header} ----')

    for answer_dict in answer_list:
        print(json.dumps(answer_dict, indent=2))


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('-r', '--rank_method', default="combined",
                    help='How to sort answers to determine the best. '
                         'Possible rank methods: ["squad", "search", "combined"]')
    ap.add_argument('-a', '--n_answers', default=10, type=int,
                    help='Number of answers to print')
    ap.add_argument('-s', '--n_search_results', default=20, type=int,
                    help='Number of search results to ask question to')
    ap.add_argument('-t', '--search_title', default=1, type=int,
                    help='Should title be searched in addition to article body? (0 if no)')
    ap.add_argument('-e', '--search_entities', default=0, type=int,
                    help='Should named entity fields be searched? (0 if no)')
    ap.add_argument('-i', '--index', default='whoosh_idx',
                    help='Name of directory containing whoosh index')
    ap.add_argument('-n', '--index_name', default='nasdaq',
                    help='Name of whoosh index')
    args = vars(ap.parse_args())

    if args['search_title'] == 0:
        search_fields = ['article']
    else:
        search_fields = ['article', 'title']

    if not args['search_entities'] == 0:
        ent_search_fields = [f + '_named_entities' for f in search_fields]
        search_fields += ent_search_fields

    bert_squad = BertSquad(download=False)
    whoosh_idx = whoosh.index.open_dir(args['index'], indexname=args['index_name'])
    query_parser = MultifieldParser(search_fields,
                                    schema=whoosh_idx.schema,
                                    group=OrGroup)

    with whoosh_idx.searcher() as searcher:
        while True:  # To infinity and beyond
            question = input('("exit" to quit) Question: ')
            if question.lower() == 'exit':
                break

            # Search documents and predict answers from search results with SQuAD model
            nasdaq_answer_df = qa_query_nasdaq(question, query_parser, searcher, bert_squad,
                                               n_search_results=args['n_search_results'])

            # Rank and subset answers
            top_answers = answer_display_df(nasdaq_answer_df,
                                            n_answers=args['n_answers'],
                                            rank_by=args['rank_method'] + '_rank')

            print(top_answers)
