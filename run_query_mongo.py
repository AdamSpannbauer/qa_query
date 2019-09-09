import pymongo
from bert_squad import QABot

MONGO_CLIENT = pymongo.MongoClient("mongodb://localhost:27017/")
DB = MONGO_CLIENT["ques_ans"]
COL = DB["nasdaq"]
N_ANSWERS = 10

qa_bot = QABot(download=False)

while True:
    question = input("('exit' to quit) Question: ")
    if question.lower() == 'exit':
        break

    matches = COL.find({"$text": {"$search": question}},
                        {"score": {"$meta": "textScore"}}) \
        .sort([('score', {'$meta': 'textScore'})])\
        .limit(N_ANSWERS)

    max_score = -999
    best_answer = -999
    best_doc = {}
    for doc in matches:
        full_text = ' '.join(doc['text'])
        answer = qa_bot.ask_question(full_text, question)
        try:
            this_score = answer[2][0] * doc["score"]
        except TypeError:
            this_score = -999
        if this_score > max_score:
            max_score = this_score
            best_answer = answer
            best_doc = doc
    print(f'\t* {best_answer[0][0]} (Confidence: {best_doc["score"]}, {answer[2]})')
