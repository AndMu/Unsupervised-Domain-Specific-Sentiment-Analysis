import random

from Discovery.SentimentTable import SentimentTable
from utilities.Lexicon import LexiconHelper

from Discovery import logger

"""
Seed words for propagating polarity scores.
"""

# From Turney and Littman (2003), probably not ideal for historical data
POSITIVE_TURNEY = ["good", "nice", "excellent", "positive", "fortunate", "correct", "superior"]
NEGATIVE_TURNEY = ["bad", "terrible", "poor", "negative", "unfortunate", "wrong", "inferior"]

POSITIVE_FINANCE = ["successful", "excellent", "profit", "beneficial", "improving", "improved", "success", "gains", "positive"]
NEGATIVE_FINANCE = ["negligent", "loss", "volatile", "wrong", "losses", "damages", "bad", "litigation", "failure", "down", "negative"]

POSITIVE_TWEET = ["love", "loved", "loves", "awesome",  "nice", "amazing", "best", "fantastic", "correct", "happy", "EMOTICON_joy", "EMOTICON_heart_eyes", "EMOTICON_smiley", "EMOTICON_smile", "EMOTICON_slightly_smiling_face", "EMOTICON_kissing_smiling_eyes", "EMOTICON_couplekiss", "EMOTICON_kissing", "EMOTICON_hearts", "EMOTICON_couple_with_heart", "EMOTICON_heartbeat", "EMOTICON_two_hearts", "EMOTICON_sparkling_heart", "EMOTICON_cupid", "EMOTICON_gift_heart", "EMOTICON_revolving_hearts", "EMOTICON_angel", "EMOTICON_clap", "EMOTICON_hugging_face", "EMOTICON_kissing_heart", "EMOTICON_relaxed"]
NEGATIVE_TWEET = ["hate", "hated", "hates", "terrible",  "nasty", "awful", "worst", "horrible", "wrong", "sad", "EMOTICON_broken_heart", "EMOTICON_crying_cat_face", "EMOTICON_cry", "EMOTICON_weary", "EMOTICON_slightly_frowning_face", "EMOTICON_sob", "EMOTICON_disappointed", "EMOTICON_angry", "EMOTICON_worried", "EMOTICON_rage", "EMOTICON_no_good", "EMOTICON_middle_finger", "EMOTICON_unamused", "EMOTICON_hankey", "EMOTICON_persevere", "EMOTICON_anguished", "EMOTICON_fearful", "EMOTICON_scream", "EMOTICON_frowning", "EMOTICON_confused", "EMOTICON_white_frowning_face", "EMOTICON_confounded", "EMOTICON_person_frowning"]

POSITIVE_HIST = ["good", "lovely", "excellent", "fortunate", "pleasant", "delightful", "perfect", "loved", "love", "happy"]
NEGATIVE_HIST = ["bad", "horrible", "poor",  "unfortunate", "unpleasant", "disgusting", "evil", "hated", "hate", "unhappy"]

POSITIVE_ADJ = ["good", "lovely", "excellent", "fortunate", "pleasant", "delightful", "perfect", "happy"]
NEGATIVE_ADJ = ["bad", "horrible", "poor",  "unfortunate", "unpleasant", "disgusting", "evil", "unhappy"]

POSITIVE_ADJ_MOVIE = ["good", "excellent", "perfect", "happy", "interesting", "amazing", "unforgettable", "genius", "gifted", "incredible"]
NEGATIVE_ADJ_MOVIE = ["bad", "bland", "horrible", "disgusting", "poor", "banal", "shallow", "disappointed", "disappointing", "lifeless", "simplistic", "bore"]


def movies_seeds():
    logger.info('movies_seeds')
    return POSITIVE_ADJ_MOVIE, NEGATIVE_ADJ_MOVIE


def amazon_seeds():
    logger.info('amazon_seeds')
    return POSITIVE_ADJ_MOVIE + POSITIVE_TURNEY, NEGATIVE_ADJ_MOVIE + NEGATIVE_TURNEY


def twitter_seeds():
    logger.info('twitter_seeds')
    return POSITIVE_TWEET, NEGATIVE_TWEET


def finance_seeds():
    logger.info('finance_seeds')
    return POSITIVE_FINANCE, NEGATIVE_FINANCE


def turney_seeds():
    logger.info('turney_seeds')
    return POSITIVE_TURNEY, NEGATIVE_TURNEY


def adj_seeds():
    logger.info('adj_seeds')
    return POSITIVE_ADJ, NEGATIVE_ADJ


def hist_seeds():
    logger.info('hist_seeds')
    return POSITIVE_HIST, NEGATIVE_HIST


def random_seeds(words, lexicon, num):
    logger.info('random_seeds')
    sample_set = list(set(words).intersection(lexicon))
    seeds = random.sample(sample_set, num)
    return [s for s in seeds if lexicon[s] == 1], [s for s in seeds if lexicon[s] == -1]


def GI_seeds():
    lexicon = LexiconHelper.load_lexicon("inquirer", remove_neutral=False)
    negative = [word for word in lexicon if lexicon[word] < 0]
    positive = [word for word in lexicon if lexicon[word] > 0]
    return positive, negative


def construct(seeds_func):
    positive, negative = seeds_func()
    logger.info("SEED Positive:%i Negative:%i", len(positive), len(negative))
    return SentimentTable.construct_positive_negative(positive, negative)
