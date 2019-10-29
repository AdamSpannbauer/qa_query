import glob
from pathlib import Path
from tqdm import tqdm
import whoosh.index
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import StemmingAnalyzer
from blank import ____


# List out all text files to be indexed
davinci_code_files = glob.glob('data/davinci_code/*.txt')

# Define schema with title as a unique key
# The only fields we have are chapter_title & chapter_text
schema = ____

# Create an index named 'davinci' in 'davinci_idx' directory
idx = ____

writer = idx.writer()
for file in tqdm(davinci_code_files):
    path = Path(file)

    # Read info to be indexed
    chapter_title = path.stem
    with path.open('r') as f:
        chapter_text = f.read()

    # Add info to index
    ____

# Finalize index build
____
