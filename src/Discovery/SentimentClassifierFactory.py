import abc

import numpy as np
from Discovery.SentimentTable import SentimentTable
from Discovery import logger


class WordVectorConstructor(object):

    def __init__(self, vector):
        self.vector = vector

    def get_vectors(self, words):
        selected_words = [item for item in words if item in self.vector.get_vocabulary()]
        vectors = self.vector.construct_dataset(selected_words)
        vectors_result = []
        for i in range(len(vectors)):
            vectors_result.append(vectors[i][1])
        return selected_words, np.array(vectors_result)


class SentimentClassifier(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, vector, classifier):
        self.vector = WordVectorConstructor(vector)
        self.classifier = classifier

    def get_words(self, sentiment):
        negative_words, negative_vectors = self.vector.get_vectors(sentiment.negative)
        positive_words, positive_vectors = self.vector.get_vectors(sentiment.positive)

        logger.info('Constructing classifier with {} positive and {} negative...'.format(len(negative_vectors),
                                                                                         len(positive_vectors)))
        y = []
        for i in range(0, len(negative_vectors)):
            y.append(0)

        for i in range(0, len(positive_vectors)):
            y.append(1)

        x = np.vstack((negative_vectors, positive_vectors))
        y = np.array(y)

        return x, y

    def fit(self, sentiment):
        x, y = self.get_words(sentiment)
        self.classifier.fit(x, y)

    def classify_sentiment(self, sentiments):
        logger.info('Starting <{}> classification...'.format(len(sentiments)))
        sentiments_words = sentiments.keys()
        words, x = self.vector.get_vectors(sentiments_words)
        logger.info('After filtering we have {}'.format(len(x)))
        y = self.classifier.predict_proba(x)
        result = SentimentTable()
        for i in range(len(y)):
            value = y[i]
            if value[0] > value[1]:
                value = -value[0]
            else:
                value = value[1]
            result.add_word(words[i], value)

        return result