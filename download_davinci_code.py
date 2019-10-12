import re
from urllib.request import urlopen
from pathlib import Path

# Download davinci code text from by archive.org
url = "https://ia800404.us.archive.org/9/items/TheDaVinciCode_201308/The%20Da%20Vinci%20Code_djvu.txt"
with urlopen(url) as f:
    davinci_code = f.read().decode('utf-8')

# Split into prologue + epilogue + chapters (drop prefacing text)
chapter_split = re.compile(r'Prologue|CHAPTER \d+|Epilogue')
chapters = re.split(chapter_split, davinci_code)[1:]

# Create dictionary of named chapters
named_chapters = {
    'prologue': chapters.pop(0).strip(),
    'epilogue': chapters.pop(-1).strip(),
}

for i, chapter in enumerate(chapters):
    named_chapters[f'chapter_{i + 1}'] = chapter.strip()

# Save text to files
for title, chapter in named_chapters.items():
    path = Path('data', 'davinci_code', title).with_suffix('.txt')
    with path.open('w') as f:
        f.write(chapter)
