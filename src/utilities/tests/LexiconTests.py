import unittest
from os import path

from nltk import TreebankWordTokenizer

from embeddings.VectorManagers import Word2VecManager
from utilities import Constants
from utilities.Lexicon import Lexicon


class LexiconTests(unittest.TestCase):

    def setUp(self):
        self.lexicon = Lexicon(TreebankWordTokenizer())

    def test_word_tokenize(self):
        tokens = self.lexicon.word_tokenize('My sample text')
        self.assertEquals(3, len(tokens))

    def test_review_to_wordlist(self):
        tokens = self.lexicon.review_to_wordlist('My the sample text')
        self.assertEquals(4, len(tokens))
        self.lexicon.remove_stopwords = True
        tokens = self.lexicon.review_to_wordlist('My the sample text')
        self.assertEquals(2, len(tokens))

    def test_review_to_sentences(self):
        tokens = self.lexicon.review_to_sentences('My the sample text')
        self.assertEquals(1, len(list(tokens)))


class Word2VecManagerTests(unittest.TestCase):

    def setUp(self):
        self.lexicon = Word2VecManager(path.join(Constants.DATASETS, 'word2vec/SemEval_min2.bin'))

    def test_construction(self):
        self.assertEquals('SemEval_min2', self.lexicon.name)