import pymongo
import glob
import json
import re
from tqdm import tqdm

MONGO_CLIENT = pymongo.MongoClient("mongodb://localhost:27017/")
DB = MONGO_CLIENT["ques_ans"]
COL = DB["nasdaq"]


def load_json_articles_to_db(collection, json_glob='data/articles/*.json'):
    for p in tqdm(glob.glob(json_glob)):
        with open(p, 'r') as f:
            article = json.load(f)
            article['text'] = re.sub('   +', '\n', ''.join(article['text'])).split('\n')
            collection.insert_one(article)


def create_index(collection):
    collection.create_index([('text', pymongo.TEXT)], default_language='english')


if __name__ == '__main__':
    load_json_articles_to_db(COL)
    create_index(COL)
