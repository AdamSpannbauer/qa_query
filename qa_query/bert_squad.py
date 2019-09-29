"""
Reference: https://github.com/deepmipt/DeepPavlov/blob/master/docs/features/models/squad.rst
Full DeepPavlov docs: http://docs.deeppavlov.ai/en/master/
"""
from deeppavlov import build_model, configs


class BertSquad:
    def __init__(self, use_noans=False, download=False):
        if use_noans:
            config = configs.squad.multi_squad_noans
        else:
            config = configs.squad.squad

        self.model = build_model(config, download=download)

    def ask_question(self, document, question):
        if not isinstance(document, list):
            document = [document]

        if not isinstance(question, list):
            question = [question]

        answer = self.model(document, question)
        return answer

    def qa_session(self, document):
        while True:
            question_text = input("('exit' to quit) Question: ")
            if question_text.lower() == 'exit':
                break

            answer = self.ask_question(document, question_text)
            print(answer)


if __name__ == '__main__':
    document_list = [
        "Tuesday's session closes with the NASDAQ Composite Index at 7,874.16. The total shares traded for the NASDAQ"
        " was over 2.18 billion. Declining stocks led advancers by 2.47 to 1 ratio. There were 926 advancers and 2287"
        " decliners for the day. On the NASDAQ Stock Exchange 35 stocks reached a 52 week high and 73 those reaching"
        " lows totaled. The most active, advancers, decliners, unusual volume and most active by dollar volume can be"
        " monitored intraday on the Most Active Stocks page.",
        "The NASDAQ 100 index closed down -1.06% for the day; a total of -81.49 points. The current value is 7,609.51. "
        "Alexion Pharmaceuticals, Inc. ( ALXN ) had the largest percent change down (-5.54%) while The Kraft Heinz "
        "Company ( KHC ) had the largest percent change gain rising 2.12%.",
        "The Dow Jones index closed down -1.08% for the day; a total of -285.26 points. The current value is 26,118.02."
        " Boeing Company (The) ( BA ) had the largest percent change down (-2.66%) while Pfizer, Inc. ( PFE ) had the "
        "largest percent change gain rising 1.6%.",
    ]

    input_document = '\n\n'.join(document_list)

    bert_squad = BertSquad(use_noans=False)
    bert_squad.qa_session(input_document)
