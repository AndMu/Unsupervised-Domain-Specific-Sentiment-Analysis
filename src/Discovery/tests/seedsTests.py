import unittest

from Discovery import seeds


class SentimentTableTests(unittest.TestCase):

    def test_twitter_seeds(self):
        result = seeds.twitter_seeds()
        self.assertEquals(31, len(result[0]))
        self.assertEquals(33, len(result[1]))

    def test_construct(self):
        table = seeds.construct(seeds.twitter_seeds)
        self.assertEquals(31, len(table.positive))
        self.assertEquals(33, len(table.negative))