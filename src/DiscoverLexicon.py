from optparse import OptionParser

import sys

from Discovery import seeds
import logging

from os import path

from Discovery.LexiconDiscover import LexiconDiscover
from embeddings.VectorManagers import Word2VecManager
from utilities import Constants
logger = logging.getLogger(__name__)
take = None
cutoff = None
balanced = None


def evaluation(word2vec, seed, output_file):

    discovery = LexiconDiscover(word2vec, seed)
    discovery.balanced = balanced
    discovery.fiter_out = cutoff
    discovery.take = take
    result = discovery.discover()
    data_set = discovery.construct(result)
    data_set.to_csv(output_file, index=False, header=False, encoding='utf-8')


def extract_lexicon_semeval():
    top_words = 20000
    word2vec = Word2VecManager(path.join(Constants.DATASETS, 'word2vec/SemEval_min2.bin'), is_binary=False, top=top_words)
    evaluation(word2vec, seeds.twitter_seeds, 'words_semval.csv')


def extract_lexicon_imdb():
    top_words = 20000
    word2vec = Word2VecManager(path.join(Constants.DATASETS, 'word2vec/Imdb_min2.bin'), top=top_words)
    evaluation(word2vec, seeds.movies_seeds, 'words_imdb.csv')


def extract_lexicon_amazon(name):
    top_words = 20000
    word2vec = Word2VecManager(path.join(Constants.DATASETS, 'word2vec/{}.bin'.format(name)), top=top_words)
    evaluation(word2vec, seeds.amazon_seeds, 'words_{}.csv'.format(name))


def extract_lexicon(name):
    top_words = 20000
    word2vec = Word2VecManager(path.join(Constants.DATASETS, 'word2vec/{}.bin'.format(name)), top=top_words)
    evaluation(word2vec, seeds.turney_seeds, 'words_{}.csv'.format(name))


if __name__ == '__main__':
    logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
    parser = OptionParser()
    parser.add_option("-d", dest="domain", default='semeval')
    parser.add_option("-b", dest="balanced", action="store_true", default=False)
    parser.add_option("-c", dest="cutoff", type="float")
    parser.add_option("-t", dest="take", type="int")

    (options, args) = parser.parse_args(sys.argv)
    balanced = options.balanced
    if options.cutoff is not None:
        cutoff = options.cutoff
        if options.take is not None:
            raise ValueError("Can't use take and cutoff together")
    elif options.take is not None:
        take = options.take
    logger.info("Extracting lexicon for [%s] with balanced: [%s] and cutoff: [%s]",
                options.domain,
                balanced,
                cutoff)

    if options.domain == "semeval":
        extract_lexicon_semeval()
    elif options.domain == "imdb":
        extract_lexicon_imdb()
    elif options.domain == "amazon":
        extract_lexicon_amazon()
    else:
        extract_lexicon(options.domain)
