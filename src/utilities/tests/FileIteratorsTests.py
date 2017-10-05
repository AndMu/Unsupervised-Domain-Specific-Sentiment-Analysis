import unittest

from os import path

from ddt import ddt, data, unpack
from mock import patch
from nltk import TreebankWordTokenizer

from embeddings.VectorManagers import Word2VecManager
from utilities import Constants
from utilities.FileIterators import ClassDataIterator, SemEvalDataIterator
from utilities.Lexicon import Lexicon
from utilities.Utilities import ClassConvertor
from embeddings.VectorSources import EmbeddingVecSource


class DataIteratorTests(unittest.TestCase):

    def setUp(self):
        with patch('embeddings.VectorSources.EmbeddingVecSource') as mock:
            instance = mock.instance
            instance.word2vec.name = 'name'
            self.iterator = ClassDataIterator(instance, 'root', 'test')

    def test_bin_location(self):
        self.assertEquals('C:/Temp/Sentiment\\bin\\root\\test\\name', self.iterator.bin_location)


@ddt
class SemEvalDataIteratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lexicon = Lexicon(TreebankWordTokenizer())
        word2vec = Word2VecManager(path.join(Constants.DATASETS, 'word2vec/Imdb_min2.bin'), top=10000)
        cls.source = EmbeddingVecSource(lexicon, word2vec)

    @data([2, 8047, 5690], [3, 14885, 5690])
    @unpack
    def test_parsing(self, num_class, expected, expected_pos):
        covertor = ClassConvertor("Binary", {"positive": 1, "negative": 0})
        if num_class == 3:
            covertor = ClassConvertor("Three", {"positive": 2, "negative": 0, "neutral": 1})
        iterator = SemEvalDataIterator(self.source, path.join(Constants.DATASETS, 'Test'), 'SemEval', covertor)
        iterator.delete_cache()
        data, names_data, type_data = iterator.get_data()
        self.assertEquals(expected, len(data))
        data, names_data, type_data = iterator.get_data()
        self.assertEquals(expected, len(data))
        class_id = num_class - 1
        self.assertEquals(expected_pos, sum(type_data == class_id))

    @data([2, 915, 575], [3, 1654, 739])
    @unpack
    def test_parsing_file(self, num_class, expected, expected_pos):

        covertor = ClassConvertor("Binary", {"positive": 1, "negative": 0})
        if num_class == 3:
            covertor = ClassConvertor("Three", {"positive": 2, "negative": 0, "neutral": 1})

        iterator = SemEvalDataIterator(self.source, path.join(Constants.DATASETS, 'Test'),
                                       'SemEval/twitter-2013dev-A.txt', covertor)
        iterator.delete_cache()
        data, names_data, type_data = iterator.get_data()
        self.assertEquals(expected, len(data))
        data, names_data, type_data = iterator.get_data()
        self.assertEquals(expected, len(data))
        self.assertEquals(expected_pos, sum(type_data == 1))

    def test_parsing_multiclass_file(self):

        covertor = ClassConvertor("Multi ", {"-2": 0, "-1": 0, "0": 1, "1": 2, "2": 2})
        iterator = SemEvalDataIterator(self.source, path.join(Constants.DATASETS, 'Test'),
                                       'SemEval/twitter-2016devtest-CE.out', covertor)
        iterator.delete_cache()
        data, names_data, type_data = iterator.get_data()
        self.assertEquals(2000, len(data))
        self.assertEquals(264, sum(type_data == 0))
        self.assertEquals(583, sum(type_data == 1))
        self.assertEquals(1153, sum(type_data == 2))
