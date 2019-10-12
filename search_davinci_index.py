import whoosh.index
from whoosh.qparser import QueryParser, OrGroup, WildcardPlugin
from blank import ____

# Load the index named 'davinci' in 'davinci_idx' directory
whoosh_idx = ____

# Define query parser to search the chapter_text field
query_parser = ____

# Remove WildcardPlugin from the query parser
____

# Input a query and parse it
query_text = 'Who painted the Mona Lisa?'
parsed_query = ____

# Search index and grab top hit
with ____:
    search_results = ____
    top_hit = [hit['chapter_text'] for hit in search_results][0]

# Check if known answer is in top hit text
print(f'`"Da Vinci" in top_hit`: {"Da Vinci" in top_hit}')
