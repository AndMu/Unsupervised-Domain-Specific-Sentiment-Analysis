import nltk
from os import path
from nltk.corpus import stopwords
import re

from utilities import Constants
from utilities.TextHelper import TextHelper
from utilities.Utilities import Utilities


class Lexicon:

    def __init__(self, tokenizer, remove_stopwords=False):
        self.word_tokenizer = tokenizer.tokenize
        self.stops = set(stopwords.words("english"))
        self.stops.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '``', '\'\'',
                      '#'])  # remove it if you need punctuation
        self.apostrophes = {"'s": " is", "'re": " are", "'nt": " not"}  ## Need a huge
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.dictionary = None
        self.cleanr = re.compile('<.*?>')
        self.remove_stopwords = remove_stopwords

    def word_tokenize(self, text):
        return self.word_tokenizer(text)

    def review_to_wordlist(self, review):
        cleantext = re.sub(self.cleanr, '', review.lower().strip())
        words = []
        for sentence in self.tokenizer.tokenize(cleantext):
            for word in self.get_words(sentence):
                words.append(word)
        return words

    def review_to_sentences(self, review):
        # Function to split a review into parsed sentences. Returns a
        # list of sentences, where each sentence is a list of words
        #
        # 1. Use the NLTK tokenizer to split the paragraph into sentences
        cleantext = re.sub(self.cleanr, '', review.lower().strip())
        raw_sentences = self.tokenizer.tokenize(cleantext)

        for raw_sentence in raw_sentences:
            # If a sentence is empty, skip it
            if len(raw_sentence) > 3:
                # Otherwise, call review_to_wordlist to get a list of words
                sencence = []
                for word in self.get_words(raw_sentence):
                    sencence.append(word)
                yield sencence

    def get_words(self, sentence):
        for word in self.word_tokenize(sentence):
            if self.remove_stopwords and (word in self.stops or len(word) <= 1):
                continue
            if word in self.apostrophes:
                yield self.apostrophes[word]
            else:
                yield TextHelper.get_raw_word(word)


class LexiconHelper(object):
    @staticmethod
    def load_lexicons(names, remove_neutral=True):
        result = {}
        for name in names:
            lexicon = LexiconHelper.load_lexicon(name, remove_neutral)
            for word, value in lexicon.iteritems():
                if word in result:
                    value = (result[word] + value) / 2
                result[word] = value
        return result

    @staticmethod
    def load_lexicon(name, remove_neutral=True):
        lexicon = Utilities.load_json(LexiconHelper.get_lexicon(name))
        return {TextHelper.get_raw_word(w): p for w, p in lexicon.items() if p != 0} if remove_neutral else lexicon

    @staticmethod
    def get_lexicon(lexicon_type):
        return path.join(Constants.PROCESSED_LEXICONS, lexicon_type + '.json')

