"""Scrape articles from NASDAQ

Usage:

python --output data/articles --n_pages 1 --page_offset 0

Process can/will lead to duplicated articles.
For now taking lazy option (instead of using a db with key restraints, or many other options).

To remove duplicates via command line:

* PowerShell

```
ls *.* -recurse | get-filehash | group -property hash | where { $_.count -gt 1 } | % { $_.group | select -skip 1 } | del
```

* Bash

```
* Fill in when not on windows machine
```
"""
import argparse
import logging
from qa_query.scrape_utils import NasdaqScraper

logging.basicConfig(level=logging.DEBUG)


ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', default='data/articles',
                help='Path for scraped results to be written to as multiple JSON files.')
ap.add_argument('-n', '--n_pages', default=50, type=int,
                help='Number of pages of articles to scrape (~10 per page).')
ap.add_argument('-p', '--page_offset', default=0, type=int,
                help='Page number to start on.')
args = vars(ap.parse_args())


scraper = NasdaqScraper()
scraper.scrape_articles(output_dir=args['output'],
                        n_pages=args['n_pages'],
                        page_offset=args['page_offset'])
