import unittest

from os import path

from Discovery import seeds
from Discovery.LexiconDiscover import LexiconDiscover
from embeddings.VectorManagers import Word2VecManager
from utilities import Constants


class LexiconDiscoverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.word2vec = Word2VecManager(path.join(Constants.DATASETS, 'word2vec/SemEval_min2.bin'), top=100000)

    def setUp(self):
        self.discover = LexiconDiscover(self.word2vec, seeds.turney_seeds)

    def test_discover(self):
        result = self.discover.discover()
        self.assertGreater(len(result.positive), 100)
        self.assertGreater(len(result.negative), 100)

    def test_discover_construct(self):
        result = self.discover.discover()
        dataset = self.discover.construct(result)
        dataset.to_csv('words.csv', index=False, header=False, encoding='UTF8')
        self.assertGreater(len(dataset), 300)
