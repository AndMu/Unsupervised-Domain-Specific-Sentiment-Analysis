import numpy as np

from utilities import logger


class EmbeddingVecSource(object):

    def __init__(self, lexicon, word2vec):
        self.lexicon = lexicon
        self.word2vec = word2vec

    def get_vector(self, file_name):
        with open(file_name) as myfile:
            text = myfile.read()
            text = text.replace('\n', '')
            text = text.decode("utf-8").encode("ascii", "ignore")
            tokens = self.lexicon.review_to_wordlist(text)
            return self.get_vector_from_tokens(tokens)

    def get_vector_from_review(self, text):
        tokens = self.lexicon.review_to_wordlist(text)
        return self.get_vector_from_tokens(tokens)

    def get_vector_from_tokens(self, tokens):
        data = list(self.word2vec.word_index[word] for word in tokens if word in self.word2vec.word_index)
        if 0 in data:
            raise ValueError("Can't have zero")
        data = np.array(data)
        if len(data) == 0:
            logger.debug("No Tokens Found")
            return None
        return data
