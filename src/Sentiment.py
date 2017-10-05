import logging
import sys
from os import path

from keras.preprocessing import sequence
from keras.utils import np_utils
from nltk import TreebankWordTokenizer
from sklearn.calibration import CalibratedClassifierCV
from cntk.train.distributed import Communicator
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline

from utilities.Utilities import ClassConvertor, Utilities
from embeddings.VectorManagers import Word2VecManager
from learning.BasicLearning import LinerClassifier
from learning.DistributedLearner import DistributedLearner
from learning.DeepLearning import WeightsLSTM, ConvNetAlgorithm, DeepSklearnWrapper

from embeddings.VectorSources import EmbeddingVecSource
from utilities import Constants
from utilities.DataLoaders import SemEvalDataLoader, ImdbDataLoader
from utilities.Lexicon import Lexicon
import pandas as pd
import numpy as np
from utilities.TwitterTreebankWordTokenizer import TwitterTreebankWordTokenizer

from optparse import OptionParser

seed = 7
np.random.seed(seed)
logger = logging.getLogger(__name__)
top_words = 100000
do_adjustment = None
is_distributed = False


def get_word_convertor(embeddings):
    def analyzer(doc):
        doc_back = [embeddings.index_word[item] for item in doc]
        return doc_back

    return analyzer


def eval_routine(domain, num_classes, algo):

    algo = algo.upper()
    logger.info("Processing [%s] with [%i] classes using [%s]", domain, num_classes, algo)

    if domain == "semeval":
        time_steps = 40
        lstm = 150
        training_source = 'train.txt'
        lexicon = Lexicon(TwitterTreebankWordTokenizer())
        word2vec_name = 'word2vec/SemEval_min2.bin'
    elif domain == 'imdb':
        time_steps = 400
        lstm = 400
        training_source = 'out'
        testing_source = 'all/test'
        lexicon = Lexicon(TreebankWordTokenizer())
        word2vec_name = 'word2vec/Imdb_min2.bin'
    elif domain == 'amazon':
        time_steps = 400
        lstm = 400
        training_source = 'out'
        testing_source = 'test'
        lexicon = Lexicon(TreebankWordTokenizer())
        word2vec_name = 'word2vec/amazon.bin'
    else:
        raise ValueError("Invalid domain: " + domain)

    if num_classes == 2:
        class_convertor = ClassConvertor("Binary", {"negative": 0, "positive": 1})
        if domain == "semeval":
            ClassConvertor.ignore_error = True
            testing_source = 'SemEval2017-task4-test.subtask-BD.english.out'
    elif num_classes == 5:
        if domain == "semeval":
            testing_source = 'SemEval2017-task4-test.subtask-CE.english.out'
        class_convertor = ClassConvertor("Three", {"negative": 0, "neutral": 1, "positive": 2})
    else:
        raise ValueError("Invalid number of classes: " + str(num_classes))

    word2vec = Word2VecManager(path.join(Constants.DATASETS, word2vec_name), top=top_words)

    source = EmbeddingVecSource(lexicon, word2vec)

    if domain == "semeval":
        loader = SemEvalDataLoader(source, class_convertor, root=path.join(Constants.DATASETS, 'SemEval'))
    elif domain == "imdb":
        loader = ImdbDataLoader(source, class_convertor, root=path.join(Constants.DATASETS, 'aclImdb'))
    else:
        loader = ImdbDataLoader(source, class_convertor, root=path.join(Constants.DATASETS, domain))

    name, train_x, train_y = loader.get_data(training_source, delete=True)
    train_x, train_y = Utilities.unison_shuffled_copies(train_x, train_y)

    if num_classes == 5:
        loader.convertor = ClassConvertor("Five_class", {"-2": 0, "-1": 0, "0": 1, "1": 2, "2": 2})

    name, test_x, test_y = loader.get_data(testing_source, delete=True)
    test_y_orig = test_y

    logger.info("Changing X Vectors")
    if algo != "SVM":
        train_x = sequence.pad_sequences(train_x, maxlen=time_steps)
        test_x = sequence.pad_sequences(test_x, maxlen=time_steps)
        if num_classes > 2:
            test_y = np_utils.to_categorical(test_y)
            train_y = np_utils.to_categorical(train_y)

    algo = algo.upper()
    run_name = domain
    if is_distributed:
        run_name = domain + "_Rank_" + str(Communicator.rank())

    if algo == "LSTM":
        classifier = WeightsLSTM(loader, run_name, time_steps, top_words=top_words, lstm=lstm)
    elif algo == "CONV":
        classifier = ConvNetAlgorithm(loader, run_name, time_steps, top_words=top_words)
    elif algo == "SVM":
        classifier = Pipeline([
            ('vect', CountVectorizer(analyzer=get_word_convertor(word2vec))),
            ('tfidf', TfidfTransformer()),
            ('sel', SelectKBest(chi2, k=1000)),
            ('clf', LinerClassifier()),
        ])
        if is_distributed:
            raise ValueError("Can't run distributed SVM")
    else:
        raise ValueError("Invalid algo: " + algo)

    if do_adjustment is not None and algo != "SVM" and not is_distributed:
        wrapper = DeepSklearnWrapper(classifier)
        classifier = CalibratedClassifierCV(wrapper, method=do_adjustment)
        logger.info("Performing %s adjustment", do_adjustment)
        train_y = Utilities.make_single_dimension(train_y)

    if is_distributed:
        logger.info("Using distributed")
        distributed = DistributedLearner(classifier)
        distributed.create_distributed_trainer()
        distributed.train_distributed(train_x, train_y)
    else:
        classifier.fit(train_x, train_y)

    if is_distributed and not distributed.rank != 0:
        logger.info("Stopping execution")
        return

    result_y_prob = classifier.predict_proba(test_x)
    result_y = Utilities.make_single_dimension(result_y_prob)
    if num_classes == 2:
        result_y_prob_single = loader.convertor.make_single(result_y_prob)
        Utilities.measure_performance_auc(test_y, result_y, result_y_prob_single)
    Utilities.measure_performance(test_y_orig, result_y)
    result = np.c_[name, result_y_prob, result_y]
    vector_frame = pd.DataFrame(data=result)
    vector_frame.sort_values(by=[0], ascending=[True])
    vector_frame.to_csv('result_{}_{}_{}_{}.csv'.format(domain, algo, num_classes, do_adjustment), index=False, header=False)


if __name__ == '__main__':
    logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
    parser = OptionParser()
    parser.add_option("-p", dest="platt", action="store_true", default=False)
    parser.add_option("-i", dest="isotonic", action="store_true", default=False)
    parser.add_option("-m", dest="multi", action="store_true", default=False)
    parser.add_option("-d", dest="domain", default='semeval')
    parser.add_option("-a", dest="algo", default='LSTM')
    parser.add_option("-n", dest="num_classes", type="int", default=2)

    parser.add_option("-w", dest="use_emb_vectors", action="store_true", default=False)
    parser.add_option("-t", dest="train_batch_size", type="int", default=10)
    parser.add_option("-l", dest="test_batch_size", type="int", default=10)
    (options, args) = parser.parse_args(sys.argv)
    Constants.TRAINING_BATCH = options.train_batch_size
    Constants.TESTING_BATCH = options.test_batch_size
    use_emb_vectors = options.use_emb_vectors
    if options.multi:
        is_distributed = True
        logger.info("Using distributed training")

    if (options.platt or options.isotonic) and options.multi:
        raise ValueError("Can't do adjustment with distributed learner")

    if options.platt:
        logger.info("With Platt calibration")
        do_adjustment = "sigmoid"
    elif options.isotonic:
        logger.info("With Isotonic calibration")
        do_adjustment = "isotonic"

    eval_routine(options.domain, options.num_classes, options.algo)
