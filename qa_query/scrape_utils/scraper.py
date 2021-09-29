import time
import warnings
import logging
import datetime
import urllib.robotparser
import requests

logger = logging.getLogger(__name__)


class Scraper:
    def __init__(self, robots_txt_url, n_tries=3):
        self.n_tries = n_tries

        self.robot_parser = urllib.robotparser.RobotFileParser()
        self.robot_parser.set_url(robots_txt_url)
        self.robot_parser.read()

        self.crawl_delay_seconds = self.robot_parser.crawl_delay(useragent="*")
        if self.crawl_delay_seconds is None:
            self.crawl_delay_seconds = 0

        self.last_request_timestamp = None
        self._update_last_request_timestamp()

    @property
    def seconds_waited(self):
        now = datetime.datetime.utcnow()
        wait_time = now - self.last_request_timestamp
        return wait_time.seconds

    def _wait_on_request_rate(self):
        logger.info(
            f"Waiting for {self.crawl_delay_seconds} second crawl delay (per robots.txt)."
        )

        while True:
            if self.seconds_waited >= self.crawl_delay_seconds:
                break
            else:
                time.sleep(0.1)

    def _update_last_request_timestamp(self):
        self.last_request_timestamp = datetime.datetime.utcnow()

    def _get_page(self, url):
        if not self.robot_parser.can_fetch(useragent="*", url=url):
            raise PermissionError('URL disallowed for useragent="*"')

        self._wait_on_request_rate()
        self._update_last_request_timestamp()

        response = requests.get(url)
        html_text = response.text

        return html_text

    def get_page(self, url):
        for i in range(self.n_tries):
            if i:
                msg = f"Failed scrape of:\n{url}\n\nAttempting try {i + 1} of {self.n_tries}"
                warnings.warn(msg)

            # noinspection PyBroadException
            try:
                return self._get_page(url)
            except Exception:
                continue

        return None


if __name__ == "__main__":
    scraper = Scraper(robots_txt_url="https://www.nasdaq.com/robots.txt")
    page_html = scraper.get_page("https://www.nasdaq.com/news/market-headlines.aspx")
