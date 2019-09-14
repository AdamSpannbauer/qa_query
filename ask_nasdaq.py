import json
import pandas as pd
import whoosh.index
from whoosh.qparser import MultifieldParser
from qa_query import QABot

QA_BOT = QABot(download=False)


def format_answer(question_text, answer_list, whoosh_search_result, context_pad_size=30):
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
        'source_text': whoosh_search_result['article'],
        'source_url': whoosh_search_result['url']
    }

    return answer_dict


def print_answers(answer_df, n_answers=3, rank_by='squad_rank', print_fields=None):
    if print_fields is None:
        print_fields = ['question', 'answer_text', 'answer_context',
                        'search_rank', 'squad_rank',
                        'source_text', 'source_url']

    answer_df = answer_df[print_fields]
    answer_df = answer_df.sort_values(rank_by)
    answer_df = answer_df.iloc[:n_answers, :]

    for _, row in answer_df.iterrows():
        print(json.dumps(row.to_dict(), indent=2))


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('-a', '--n_answers', default=3, type=int,
                    help='Number of answers to print (per each of the 2 rank methods)')
    ap.add_argument('-r', '--n_search_results', default=20, type=int,
                    help='Number of search results to ask question to')
    ap.add_argument('-t', '--search_title', default=1, type=int,
                    help='Should title be searched in addition to article body? (0 if no)')
    ap.add_argument('-i', '--index', default='whoosh_idx',
                    help='Name of directory containing whoosh index')
    ap.add_argument('-n', '--index_name', default='nasdaq',
                    help='Name of whoosh index')
    args = vars(ap.parse_args())

    if args['search_title'] == 0:
        search_fields = ['article_named_entities', 'article']
    else:
        search_fields = ['article_named_entities', 'article',
                         'title_named_entities', 'title']

    whoosh_idx = whoosh.index.open_dir(args['index'], indexname=args['index_name'])
    query_parser = MultifieldParser(search_fields,
                                    schema=whoosh_idx.schema,
                                    group=whoosh.qparser.OrGroup)

    with whoosh_idx.searcher() as searcher:
        while True:
            question = input('("exit" to quit) Question: ')
            if question.lower() == 'exit':
                break

            query = query_parser.parse(question)
            search_results = searcher.search(query, limit=args['n_search_results'])

            answers = []
            for i, search_result in enumerate(search_results):
                full_answer = QA_BOT.ask_question(search_result['article'], question)
                answer = format_answer(question, full_answer, search_result)
                answer['search_rank'] = i + 1.0
                answers.append(answer)

            nasdaq_answer_df = pd.DataFrame(answers)
            nasdaq_answer_df['squad_rank'] = nasdaq_answer_df['answer_confidence'].rank(ascending=False)

            print('\n---- TOP ANSWERS BY SEARCH RANK ----')
            print_answers(nasdaq_answer_df, n_answers=args['n_answers'], rank_by='search_rank')

            print('\n---- TOP ANSWERS BY SQUAD RANK ----')
            print_answers(nasdaq_answer_df, n_answers=args['n_answers'], rank_by='squad_rank')
