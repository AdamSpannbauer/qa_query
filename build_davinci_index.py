import glob
from pathlib import Path
from tqdm import tqdm
import whoosh.index
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import StemmingAnalyzer

# List out all text files to be indexed
davinci_code_files = glob.glob('data/davinci_code/*.txt')

# Define schema with title as a unique key
schema = Schema(
        chapter_title=ID(stored=True, unique=True),
        chapter_text=TEXT(stored=True, analyzer=StemmingAnalyzer()),
    )

# Create an index named 'davinci' in 'davinci_idx' directory
idx = whoosh.index.create_in('davinci_idx', schema=schema, indexname='davinci')

writer = idx.writer()
for file in tqdm(davinci_code_files):
    path = Path(file)

    # Read info to be indexed
    chapter_title = path.stem
    with path.open('r') as f:
        chapter_text = f.read()

    # Add info to index
    writer.update_document(
        chapter_title=chapter_title,
        chapter_text=chapter_text,
    )

# Finalize index build
writer.commit()
