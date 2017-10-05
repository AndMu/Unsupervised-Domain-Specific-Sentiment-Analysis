from Discovery import seeds, logger
from Discovery.SentimentClassifierFactory import SentimentClassifier
import pandas as pd

from learning.BasicLearning import LinerClassifier
from utilities.Lexicon import LexiconHelper


class LexiconDiscover(object):

    def __init__(self, vector_manager, domain_seeds):
        self.lexicon = LexiconHelper.load_lexicons(["inquirer", "bingliu"], remove_neutral=True)
        self.vector = vector_manager
        self.sentiment_seeds = seeds.construct(domain_seeds)
        self.fiter_out = None
        self.balanced = None
        self.take = None
        for item in self.vector.emoticons:
             self.lexicon[item] = 0

    def discover(self):
        classifier = SentimentClassifier(self.vector, LinerClassifier())
        logger.info("Fit main classifier to find bests parameters")
        classifier.fit(self.sentiment_seeds)
        logger.info("Finding sentiment")
        result = classifier.classify_sentiment(self.lexicon)
        return result

    def construct(self, result):
        frame = pd.DataFrame(result.original_data)
        if self.fiter_out is not None:
            frame = frame[(frame[1] > self.fiter_out) | (frame[1] < -self.fiter_out)]
        elif self.take is not None:
            logger.info("Taking %i words into dictionary", self.take)
            frame = self.filter_frame(frame, self.take / 2)

        if self.balanced == True and self.take is None:
            total_negative = len(frame[frame[1] < 0])
            total_positive = len(frame[frame[1] > 0])
            if total_negative > total_positive:
                total = total_positive
            else:
                total = total_negative

            total = int(total)
            frame = self.filter_frame(frame, total)

        frame[1] *= 2
        logger.info("Final size - %i", len(frame.index))
        return frame

    def filter_frame(self, frame, total):
        cutoff_pos = frame.nlargest(total, 1).min()[1]
        cutoff_neg = frame.nsmallest(total, 1).max()[1]
        frame = frame[(frame[1] >= cutoff_pos) | (frame[1] <= cutoff_neg)]
        return frame