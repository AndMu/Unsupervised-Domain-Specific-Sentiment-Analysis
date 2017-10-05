import logging

from nltk import TreebankWordTokenizer

from embeddings.Builders import Word2VecBuilder
from utilities.TextHelper import TextHelper
from utilities.Lexicon import Lexicon
from utilities.TwitterTreebankWordTokenizer import TwitterTreebankWordTokenizer


def build_SemEval():
    lexicon = Lexicon(TwitterTreebankWordTokenizer())
    lexicon.remove_stopwords = False
    TextHelper.stem_words = False
    builder = Word2VecBuilder(lexicon)
    builder.build('SemEval/all.out', 'SemEval_min2', min_count=2, dynamic=False)


def build_Imdb():
    lexicon = Lexicon(TreebankWordTokenizer())
    lexicon.remove_stopwords = False
    TextHelper.stem_words = False
    builder = Word2VecBuilder(lexicon)
    builder.build('aclImdb/All', 'Imdb_min2', min_count=2, dynamic=False)


def build_Amazon():
    lexicon = Lexicon(TreebankWordTokenizer())
    lexicon.remove_stopwords = False
    TextHelper.stem_words = False
    builder = Word2VecBuilder(lexicon)
    builder.build('Amazon', 'Amazon', min_count=2, dynamic=False)


if __name__ == '__main__':
    logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
    build_Amazon()