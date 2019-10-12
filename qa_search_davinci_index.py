import whoosh.index
from whoosh.qparser import QueryParser, OrGroup, WildcardPlugin
from deeppavlov import build_model, configs
from blank import ____

# Init Q&A model
bert_squad_model = ____

# Load the index named 'davinci' in 'davinci_idx' directory
whoosh_idx = ____

# Define query parser to search the chapter_text field
query_parser = ____

# Remove WildcardPlugin from the query parser
____

# Search index and grab top hit
with ____:
    while True:
        query = input('Query ("exit" to quit): ')
        if query == 'exit':
            break

        parsed_query = ____

        search_results = ____
        top_hit = [hit['chapter_text'] for hit in search_results][0]

        # Ask question to top hit with SQuAD model
        print(____)
