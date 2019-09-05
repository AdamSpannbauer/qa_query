import os
import json
import uuid
import logging
from datetime import datetime
import dateutil.parser
from bs4 import BeautifulSoup
from scraper import Scraper

logger = logging.getLogger(__name__)


class NasdaqScraper(Scraper):
    robots_txt_url = 'https://www.nasdaq.com/robots.txt'
    splash_page_n_url = 'https://www.nasdaq.com/news/market-headlines.aspx?page={n}'
    article_div_selector = '#newsContent p'
    article_text_selector = '#articlebody p'

    def __init__(self, n_tries=3, html_parser="html.parser"):
        self.html_parser = html_parser
        super().__init__(robots_txt_url=self.robots_txt_url, n_tries=n_tries)

        self.article_urls = []
        self.articles = []

    def _scrape_article_urls(self, page=1):
        page_url = self.splash_page_n_url.format(n=page)
        splash_page_html = self.get_page(page_url)

        if splash_page_html is None:
            return []

        splash_page_soup = BeautifulSoup(splash_page_html, features=self.html_parser)
        article_divs = splash_page_soup.select(self.article_div_selector)

        urls = []
        for article_div in article_divs:
            article_a_tag = article_div.select('span a')[0]
            article_url = article_a_tag.get_attribute_list('href')[0].strip()
            self.article_urls.append(article_url)

        return urls

    def scrape_articles(self, output_dir, n_pages, page_offset=0):
        for i in range(n_pages):
            logger.info(f'Grabbing article links for page {i + 1} of {n_pages}')
            self._scrape_article_urls(page=i + 1 + page_offset)

        for i, article_url in enumerate(self.article_urls):
            logger.info(f'Scraping article {i + 1} of {len(self.article_urls)}')
            article_html = self.get_page(article_url)

            if article_html is None:
                continue

            article_soup = BeautifulSoup(article_html, features=self.html_parser)
            for t in article_soup.findAll('table'):
                t.decompose()

            article_title = article_soup.select('title')[0].text
            article_published_datetime = article_soup.select('span[itemprop="datePublished"]')[0]['content']
            article_published_datetime = dateutil.parser.parse(article_published_datetime)

            article_p_tags = article_soup.select(self.article_text_selector)
            article_text = [p_tag.text for p_tag in article_p_tags if p_tag.text.strip()]

            article_dict = {
                'url': article_url,
                'title': article_title,
                'text': article_text,
                'published_datetime': article_published_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            }

            article_id = uuid.uuid4()
            file_path = os.path.join(output_dir, f'article_{article_id}.json')

            with open(file_path, 'w') as f:
                f.write(json.dumps(article_dict, indent=2))

            self.articles.append(article_dict)

    def json_dump_articles(self, file_path, extend=True):
        if extend and os.path.exists(file_path):
            with open(file_path, 'r') as f:
                articles = json.load(f)['articles']

            articles.extend(self.articles)
        else:
            articles = self.articles

        articles_dict = {'articles': articles}
        with open(file_path, 'w') as f:
            f.write(json.dumps(articles_dict, indent=2))


if __name__ == '__main__':
    # Process can/will lead to duplicated articles.. until moving to a db with key restraints..
    # To remove duplicates:
    #   * PowerShell
    #        * From data/articles directory run:
    #            ls *.* -recurse | get-filehash | group -property hash | where { $_.count -gt 1 } | % { $_.group | select -skip 1 } | del
    #   * Bash
    #        * Fill in when not on windows machine
    import argparse

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
