import nltk
import nltk.corpus
from whoosh.analysis import (
    StemmingAnalyzer,
    RegexTokenizer,
    LowercaseFilter,
    StopFilter,
    StemFilter,
)
from whoosh.analysis.tokenizers import default_pattern

NLTK_STOPWORDS = set(nltk.corpus.stopwords.words("english"))
QUESTION_STOPWORDS = {"who", "what", "where", "when", "why", "how"}
QA_STOPWORDS = frozenset(QUESTION_STOPWORDS | NLTK_STOPWORDS)
QA_NE_TYPES = frozenset(["ORGANIZATION", "LOCATION", "FACILITY", "GPE"])
ALL_NE_TYPES = frozenset(
    [
        "ORGANIZATION",
        "PERSON",
        "LOCATION",
        "DATE",
        "TIME",
        "MONEY",
        "PERCENT",
        "FACILITY",
        "GPE",
    ]
)


def ner_extract(text, ne_types=QA_NE_TYPES):
    """Remove non named entities from a string

    :param text: str to remove non named entities from
    :param ne_types: list/set of named entities to keep
    :return: text with non named entities removed
    """
    if ne_types is None:
        ne_types = ALL_NE_TYPES

    chunks = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text)))
    ne_list = []
    for chunk in chunks:
        if hasattr(chunk, "label"):
            if chunk.label() in ne_types:
                full_ne = " ".join(c[0] for c in chunk)
                ne_list.append(full_ne)

    return " ".join(ne_list)


class NERTokenizer(RegexTokenizer):
    """Named Entity centric version of RegexTokenizer"""

    def __init__(self, ne_types=QA_NE_TYPES, expression=default_pattern, gaps=False):
        self.ne_types = ne_types
        super().__init__(expression=expression, gaps=gaps)

    def __call__(self, text, **kwargs):
        text = ner_extract(text=text, ne_types=self.ne_types)
        return super().__call__(text, **kwargs)


# Conforming to camelCase convention used for analyzer functions in whoosh_utils
# noinspection PyPep8Naming
def NERAnalyzer(
    ne_types=QA_NE_TYPES,
    expression=default_pattern,
    stoplist=QA_STOPWORDS,
    minsize=2,
    maxsize=None,
    gaps=False,
):
    """Named Entity centric version of StandardAnalyzer

    :param ne_types: list/set of named entities to keep
    :param expression: The regular expression pattern to use to extract tokens.
    :param stoplist: A list of stop words. Set this to None to disable
        the stop word filter.
    :param minsize: Words smaller than this are removed from the stream.
    :param maxsize: Words longer that this are removed from the stream.
    :param gaps: If True, the tokenizer *splits* on the expression, rather
        than matching on the expression.
    :return: analyzer to be used with a whoosh_utils index
    """
    chain = NERTokenizer(ne_types, expression, gaps)
    chain |= LowercaseFilter()
    chain |= StopFilter(stoplist=stoplist, minsize=minsize, maxsize=maxsize)

    return chain


# Conforming to camelCase convention used for analyzer functions in whoosh_utils
# noinspection PyPep8Naming
def QAAnalyzer(
    ner_tokenize=False,
    ne_types=QA_NE_TYPES,
    expression=default_pattern,
    stoplist=QA_STOPWORDS,
    minsize=2,
    maxsize=None,
    gaps=False,
):
    """Custom whoosh_utils analyzer for processing Named Entities and using custom stoplist

    :param ner_tokenize: Should NERAnalyzer be used instead of StandardAnalyzer
    :param ne_types: list/set of named entities to keep
    :param expression: The regular expression pattern to use to extract tokens.
    :param stoplist: A list of stop words. Set this to None to disable
        the stop word filter.
    :param minsize: Words smaller than this are removed from the stream.
    :param maxsize: Words longer that this are removed from the stream.
    :param gaps: If True, the tokenizer *splits* on the expression, rather
        than matching on the expression.
    :return: analyzer to be used with a whoosh_utils index
    """
    if not ner_tokenize:
        return StemmingAnalyzer(expression, stoplist, minsize, maxsize, gaps)
    else:
        return NERAnalyzer(ne_types, expression, stoplist, minsize, maxsize, gaps)
