import whoosh.index
from whoosh.qparser import QueryParser, OrGroup, WildcardPlugin

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

# Input query and parse
# "What subject is Robert Langdon a professor of?"
# "What is the notable poetic structure of the poem inside the cryptex box?"
# "Where does Silas wear his spiked cilice belt?"
parsed_query = query_parser.parse('Who painted the Mona Lisa?')
print(parsed_query)

# Search index and grab top hit
with whoosh_idx.searcher() as searcher:
    search_results = searcher.search(parsed_query, limit=1)
    top_hit = [hit['chapter_text'] for hit in search_results][0]

# Check if known answer is in top hit text
print(f'`"Da Vinci" in top_hit`: {"Da Vinci" in top_hit}')
