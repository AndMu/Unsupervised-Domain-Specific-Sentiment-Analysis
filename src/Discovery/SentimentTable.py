import numpy as np
import pandas as pd
from Discovery import logger


class SentimentTable(object):
    """List of sentiment words"""

    def __init__(self, data=None):
        self.positive = []
        self.negative = []
        self.original_data = []
        self.y = []
        if data is not None:
            for item in data:
                self.add_word(item[0], float(item[1]))
            self.original_data = data

    def words(self):
        return list(self.get_frame().Word)

    def to_dic(self, normalize=False):
        frame = self.get_frame()
        if normalize:
            max = frame.Sentiment.abs().max()
            frame.Sentiment /= max
        return frame.set_index('Word')['Sentiment'].to_dict()

    def add_word(self, word, sentiment):
        word = word.lower()
        self.y.append(sentiment)
        if sentiment > 0:
            self.positive.append(word)

        if sentiment < 0:
            self.negative.append(word)

        self.original_data.append((word, sentiment))

    def balance(self):
        total_positive = len(self.positive)
        total_negative = len(self.negative)
        take = total_negative
        if total_positive < total_negative:
            take = total_positive

        return self.take_top(take)

    def take_top(self, top):
        logger.info("Taking top {}".format(top))
        positive = [x for x in self.original_data if float(x[1]) > 0]
        negative = [x for x in self.original_data if float(x[1]) < 0]
        positive = sorted(positive, key=lambda x: float(x[1]), reverse=True)[:top]
        negative = sorted(negative, key=lambda x: float(x[1]), reverse=False)[:top]
        return SentimentTable(positive + negative)

    def get_frame(self):
        data = pd.DataFrame([[item[0].lower(), float(item[1])] for item in self.original_data],
                            columns=['Word', 'Sentiment'])
        if len(data.Word) == 0:
            return data

        return data.groupby(['Word'], as_index=False)['Sentiment'].mean()

    def save_csv(self, filename, baseline=None):
        logger.info("Save CSV <{}>...".format(filename))
        frame = self.get_frame()
        if baseline is not None:
            frame.join(baseline.get_frame(), rsuffix='_baseline')
        frame.to_csv(filename, index=False, header=False)

    @staticmethod
    def construct_frame(data):
        subset = data[['Word', 'Sentiment']]
        items = [tuple(x) for x in subset.values]
        return SentimentTable(items)

    @staticmethod
    def construct_from_dict(data):
        list_key_value = [[k, v] for k, v in data.items()]
        return SentimentTable(list_key_value)

    @staticmethod
    def construct_bootstrapper(data):
        items = []
        for i in range(len(data)):
            items.append((data[i][0], data[i][1].value))
        return SentimentTable(items)

    @staticmethod
    def construct_positive_negative(positive, negative):
        items = []
        for item in positive:
            items.append((item, 1))
        for item in negative:
            items.append((item, -1))
        return SentimentTable(items)


class SentimentHelper(object):
    """Helper class to perform various functions"""

    @staticmethod
    def calculate_sentiment_value(positive, negative):
        coefficient = 2
        if positive == 0 and negative == 0:
            return 0

        min_value = 0.1
        positive += min_value
        negative += min_value
        rating = np.log2(positive / negative)

        if rating < -coefficient:
            rating = -coefficient
        elif rating > coefficient:
            rating = coefficient

        rating /= coefficient
        return np.round(rating, 2)


class SentimentValue(object):
    """Sentiment value holder"""

    def __init__(self, positive, negative):
        self.positive = positive
        self.negative = negative

        if positive < 0 and negative < 0:
            self.value = 0
            return

        if positive < 0:
            # increase negative
            negative -= positive
            positive = 0
        elif negative < 0:
            # increase positive
            positive -= negative
            negative = 0

        self.value = SentimentHelper.calculate_sentiment_value(positive, negative)
