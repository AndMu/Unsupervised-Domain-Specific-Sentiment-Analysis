import unittest

from keras.preprocessing import sequence
from nltk import TreebankWordTokenizer
from os import path

from embeddings.VectorSources import EmbeddingVecSource
from utilities.Utilities import ClassConvertor
from embeddings.VectorManagers import Word2VecManager
from learning.DeepLearning import BaselineLSTM, WeightsLSTM
from utilities import Constants
from utilities.DataLoaders import ImdbDataLoader
from utilities.Lexicon import Lexicon


class WeightsLSTMTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        lexicon = Lexicon(TreebankWordTokenizer())
        word2vec = Word2VecManager(path.join(Constants.DATASETS, 'word2vec/SemEval_min2.bin'), top=10000)
        source = EmbeddingVecSource(lexicon, word2vec)
        class_convertor = ClassConvertor("Binary", {"0": 0, "1": 1})
        cls.loader = ImdbDataLoader(source, class_convertor, root=path.join(Constants.DATASETS, 'test'))
        pass

    def test_acceptance(self):
        first = WeightsLSTM(self.loader, 'AcceptanceTest', 100)
        name, train_x, train_y = self.loader.get_data('train', delete=True)
        train_x = sequence.pad_sequences(train_x, maxlen=100)
        first.fit(train_x, train_y)
        name, test_x, test_y = self.loader.get_data('test', delete=True)
        test_x = sequence.pad_sequences(test_x, maxlen=100)
        y = first.predict(test_x)
        self.assertEquals(20, len(y))
        self.assertGreater(sum(y > 0.5), 0)
        self.assertGreater(sum(y < 0.5), 0)
