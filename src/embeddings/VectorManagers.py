import abc

from keras.preprocessing.text import Tokenizer
from nltk import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

from learning import logger
from os import path
import numpy as np
import gensim
from gensim.models import Doc2Vec

from utilities.TextHelper import TextHelper


class BaseVecManager(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, total_words, word_index, index_word, word_vector_table, vector_size, word_vectors):
        self.name = name
        self.total_words = total_words
        self.word_index = word_index
        self.index_word = index_word
        self.word_vector_table = word_vector_table
        self.vector_size = vector_size
        self.word_vectors = word_vectors
        self.stemmer = PorterStemmer()
        self.emoticons = []
        self.hash_tags = []
        # prepare embedding matrix
        self.embedding_matrix = np.zeros((total_words + 1, vector_size))

        for word in word_vector_table.keys():
            embedding_vector = word_vector_table.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                i = word_index[word]
                self.embedding_matrix[i] = embedding_vector

            if TextHelper.is_emoticon(word):
                self.emoticons.append(word)
            if TextHelper.is_hash(word):
                self.hash_tags.append(word)

        logger.debug("Initialized")

    def get_vocabulary(self):
        return self.word_index

    def get_tokens(self, tokens):
        result = []
        for word in tokens:
            if word in self.word_index:
                result.append(self.word_vector_table[word])
            else:
                stemmed = self.stemmer.stem(word)
                if stemmed in self.word_index:
                    result.append(self.word_vector_table[stemmed])
        return result

    def construct_dataset(self, words):
        vectors = []
        for word in words:
            item = word.lower()
            vector = self.word_vector_table[item]
            vectors.append((item, vector))
        return vectors


class Word2VecManager(BaseVecManager):

    def __init__(self, file_name, is_binary=False, top=10000):
        name = path.splitext(path.split(file_name)[-1])[0]
        self.is_binary = is_binary
        w2vModel = self.construct(file_name)
        logger.info('Sorting words')
        sorted_list = sorted(w2vModel.wv.vocab.items(), key=lambda t: t[1].count, reverse=True)[0:top]
        total_words = len(sorted_list)
        index = 1
        word_index = {}
        index_word = {}
        word_vector_table = {}
        vectors = []

        for wordKey in sorted_list:
            word = wordKey[0]
            if len(word) == 0:
                continue
            vector = w2vModel.wv[word]
            word_index[word] = index
            index_word[index] = word
            vectors.append(vector)
            word_vector_table[word] = vector
            index += 1

        vector_size = w2vModel.vector_size
        word_vectors = np.array(vectors)
        super(Word2VecManager, self).__init__(name, total_words, word_index, index_word, word_vector_table, vector_size, word_vectors)

    def construct(self, file_name):
        logger.info('Loading Word2Vec...')
        if self.is_binary:
            logger.info('Loading binary version')
            return gensim.models.KeyedVectors.load_word2vec_format(file_name, binary=True)
        return gensim.models.Word2Vec.load(file_name)


# takes generic embeding class to load vectors
class EmbeddingManager(BaseVecManager):
    def __init__(self, embeddings):
        total_words = len(embeddings.vocabulary)
        word_index = embeddings.word_index
        index_word = embeddings.index_word
        vector_size = embeddings.dim
        word_vector_table = embeddings.word_vector_table
        word_vectors = embeddings.vectors
        super(EmbeddingManager, self).__init__(embeddings.name, total_words, word_index, index_word, word_vector_table,
                                               vector_size, word_vectors)