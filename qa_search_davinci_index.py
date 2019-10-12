import whoosh.index
from whoosh.qparser import QueryParser, OrGroup, WildcardPlugin
from deeppavlov import build_model, configs

# Init Q&A model
bert_squad_model = build_model(configs.squad.squad, download=False)

# Load the index named 'davinci' in 'davinci_idx' directory
whoosh_idx = whoosh.index.open_dir('davinci_idx', indexname='davinci')

# Only one field to search, if we wanted to search multiple we
# would use MultifieldParser
query_parser = QueryParser('chapter_text',
                           schema=whoosh_idx.schema,
                           group=OrGroup)

# Our goal is to make queries as natural as possible
# About wildcards in queries
# * Overview
#     * https://nlp.stanford.edu/IR-book/html/htmledition/wildcard-queries-1.html
# * Example characters
#     * https://support.office.com/en-us/article/examples-of-wildcard-characters-939e153f-bd30-47e4-a763-61897c87b3f4
query_parser.remove_plugin_class(WildcardPlugin)

# Search index and grab top hit
with whoosh_idx.searcher() as searcher:
    while True:
        query = input('Query ("exit" to quit): ')
        if query == 'exit':
            break

        parsed_query = query_parser.parse(query)

        search_results = searcher.search(parsed_query, limit=1)
        top_hit = [hit['chapter_text'] for hit in search_results][0]

        print(bert_squad_model([top_hit], [query]))
