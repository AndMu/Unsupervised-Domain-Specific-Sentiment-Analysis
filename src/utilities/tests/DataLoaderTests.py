import unittest

from ddt import ddt, data
from nltk import TreebankWordTokenizer
from os import path

from embeddings.VectorManagers import Word2VecManager
from embeddings.VectorSources import EmbeddingVecSource
from utilities import Constants
from utilities.DataLoaders import ImdbDataLoader
from utilities.Lexicon import Lexicon
from utilities.Utilities import ClassConvertor

@ddt
class ImdbDataLoaderTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        lexicon = Lexicon(TreebankWordTokenizer())
        word2vec = Word2VecManager(path.join(Constants.DATASETS, 'word2vec/SemEval_min2.bin'), top=10000)
        cls.source = EmbeddingVecSource(lexicon, word2vec)
        cls.convertor = ClassConvertor("Binary", {"0": 0, "1": 1})

    def test_get_data(self):
        loader = ImdbDataLoader(self.source, self.convertor, root=path.join(Constants.DATASETS, 'Test'))
        # delete records, so no clash between dual and single
        names, train_x, train_y = loader.get_data('train', delete=True)
        self.assertEquals(20, len(names))
        self.assertEquals(20, len(train_x))
        self.assertEquals(20, len(train_y))
        # Test loading
        names, train_x, train_y = loader.get_data('train', delete=False)
        self.assertEquals(20, len(train_x))

    def test_get_single_data(self):
        loader = ImdbDataLoader(self.source, self.convertor, root=path.join(Constants.DATASETS, 'Test'))
        # delete records, so no clash between dual and single
        names, train_x, train_y = loader.get_data('train/pos', delete=True, class_iter=False)
        self.assertEquals(10, len(names))
        self.assertEquals(10, len(train_x))
        self.assertEquals(10, len(train_y))
        # Test loading
        names, train_x, train_y = loader.get_data('train/pos', delete=False, class_iter=False)
        self.assertEquals(10, len(train_x))

    @data(True, False)
    def test_unknown_data(self, class_iter):
        loader = ImdbDataLoader(self.source, self.convertor, root=path.join(Constants.DATASETS, 'Test'))
        self.assertRaises(StandardError, loader.get_data, 'xxx', delete=True, class_iter=class_iter)
