from os import path, makedirs
from random import shuffle

import gensim
from pathlib2 import Path

from utilities import Constants
from utilities.DocumentExtractors import SingleFileLineSentence, MultiFileLineSentence, MultiFileLineDocument
from embeddings import logger


class Word2VecBuilder:
    def __init__(self, lexicon, location=Constants.DATASETS):
        self.location_out = path.join(location, "word2vec/")
        self.lexicon = lexicon

    def build(self, source, name, size=500, window=5, min_count=2, dynamic=True):

        source = path.join(Constants.DATASETS, source)
        logger.info("Building word2vec %s...", source)
        model = gensim.models.Word2Vec(size=size, window=window, min_count=min_count, workers=60)

        source_path = Path(source)
        if not source_path.exists():
            raise ValueError("Path not found " + source_path)
        if source_path.is_file():
            sentences = SingleFileLineSentence(self.lexicon, source)
        else:
            sentences = MultiFileLineSentence(self.lexicon, source)

        if not dynamic:
            logger.info("Reading using static list...")
            sentences = list(sentences)

        logger.info("Building vocab...")
        model.build_vocab(sentences)
        logger.info("Training with %i sentences...", len(sentences))
        model.train(sentences, total_examples=len(sentences), epochs=15)

        model.save(path.join(self.location_out, name + ".bin"))
        return model


class Doc2VecBuilder:
    def __init__(self, lexicon, location=Constants.DATASETS):
        self.location_out = path.join(location, "doc2vec/")
        self.lexicon = lexicon

    def build(self, source, name, size=300):

        source = path.join(Constants.DATASETS, source)
        logger.info("Building word2vec %s...", source)
        model = gensim.models.Doc2Vec(size=size, window=20, min_count=2, workers=20)
        logger.info('Iterate documents')
        document_iterator = MultiFileLineDocument(self.lexicon, source)
        list_docs = list(document_iterator)
        logger.info('Found %i documents', len(list_docs))
        model.build_vocab(list_docs)

        # this could improve but we have to know how many we have reviews and s
        for epoch in range(10):
            logger.info("Echo %i", epoch)
            shuffle(list_docs)
            model.train(list_docs)
        if not path.exists(self.location_out):
            makedirs(self.location_out)
        model.save(path.join(self.location_out, name + ".bin"))
        return model
