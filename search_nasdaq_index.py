import whoosh.index
from whoosh.qparser import MultifieldParser, OrGroup, WildcardPlugin

whoosh_idx = whoosh.index.open_dir('whoosh_idx', indexname='nasdaq')
query_parser = MultifieldParser(['title', 'article'],
                                schema=whoosh_idx.schema,
                                group=OrGroup)
query_parser.remove_plugin_class(WildcardPlugin)

parsed_query = query_parser.parse('What market does FitBit compete in?')

with whoosh_idx.searcher() as searcher:
    search_results = searcher.search(parsed_query, limit=1)
    [print(sr['title']) for sr in search_results]
