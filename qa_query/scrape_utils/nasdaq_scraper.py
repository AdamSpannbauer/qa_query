import os
import glob
import json
import uuid
import logging
import dateutil.parser
import pandas as pd
from bs4 import BeautifulSoup
from .scraper import Scraper

logger = logging.getLogger(__name__)


def read_json_articles_to_df(json_glob='data/articles/*.json'):
    """Utility for reading json output of NasdaqScraper to pandas.DataFrame"""
    article_list = []
    for p in glob.glob(json_glob):
        with open(p, 'r') as f:
            article_list.append(json.load(f))

    articles_df = pd.DataFrame(article_list)

    full_df = articles_df.copy()
    full_df['text'] = full_df['text'].apply(lambda x: '\n\n'.join(x))
    full_df = full_df.drop_duplicates().dropna()

    return full_df


class NasdaqScraper(Scraper):
    robots_txt_url = 'https://www.nasdaq.com/robots.txt'
    splash_page_n_url = 'https://www.nasdaq.com/news-and-insights/topic/markets/page/{n}'
    article_div_selector = '.content-feed__card'
    article_text_selector = '.body__content p'

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
            article_a_tag = article_div.select('.content-feed__card-title-link')[0]
            article_url = 'https://www.nasdaq.com' + article_a_tag.get_attribute_list('href')[0].strip()
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
            article_published_datetime = article_soup.select('.timestamp__date')[0]['datetime']
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
