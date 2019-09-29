import whoosh.index
from whoosh.qparser import MultifieldParser, OrGroup, WildcardPlugin
from qa_query import BertSquad

# Init qa process
bert_squad = BertSquad()

# Init search process
whoosh_idx = whoosh.index.open_dir('whoosh_idx', indexname='nasdaq')
query_parser = MultifieldParser(['title', 'article'],
                                schema=whoosh_idx.schema,
                                group=OrGroup)
query_parser.remove_plugin_class(WildcardPlugin)

# Perform Q&A query
question = 'What market does FitBit compete in?'
parsed_query = query_parser.parse(question)
with whoosh_idx.searcher() as searcher:
    search_results = searcher.search(parsed_query, limit=3)
    result_texts = [sr['article'] for sr in search_results]
    answers = [bert_squad.ask_question([t], [question]) for t in result_texts]

print(answers)
