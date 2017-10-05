import unittest

from Discovery.SentimentTable import SentimentTable
from Discovery.SentimentTable import SentimentValue
from utilities.Lexicon import LexiconHelper


class SentimentTableTests(unittest.TestCase):
    def test_construction(self):
        table = self.table_default
        self.assertEqual("good", table.positive[0])
        self.assertEqual("bad", table.negative[0])
        self.assertEqual(2, len(table.words()))
        self.assertEqual(1, table.to_dic()["good"])

    def get_frame(self):
        table = self.table_default
        frame = table.get_frame()
        self.assertEqual(2, len(frame.index))
        table = SentimentTable.construct_frame(frame)
        self.assertEqual("good", table.positive[0])
        self.assertEqual("bad", table.negative[0])
        self.assertEqual(2, len(table.table))

    def test_add_word(self):
        table = self.table_default
        self.assertEqual(2, len(table.words()))
        table.add_word("x", -1)
        self.assertEqual(3, len(table.words()))

    def test_take_top(self):
        table = self.table_default
        table.add_word("super good", 10)
        table.add_word("so so good", 0.5)
        table.add_word("so so bad", -0.5)
        table.add_word("super bad", -10)
        self.assertEqual(6, len(table.words()))
        table = table.take_top(1)
        self.assertEqual(2, len(table.words()))
        self.assertEqual("super good", table.positive[0])
        self.assertEqual("super bad", table.negative[0])

    def test_csv_save(self):
        table = self.table_default
        table.save_csv("test.csv")
        table.save_csv("test_baseline.csv", table)

    def test_construct_bootstrapper(self):
        data = [["Good", SentimentValue(1, 0)], ["Bad", SentimentValue(0, 1)]]
        table = SentimentTable.construct_bootstrapper(data)
        self.assertEqual("good", table.positive[0])
        self.assertEqual("bad", table.negative[0])
        self.assertEqual(2, len(table.words()))

    def test_construct_positive_negative(self):
        table = SentimentTable.construct_positive_negative(["Good"], ["Bad"])
        self.assertEqual("good", table.positive[0])
        self.assertEqual("bad", table.negative[0])
        self.assertEqual(2, len(table.words()))

    def test_to_dic(self):
        data = [["Good", "2"], ["Bad", "-1"]]
        self.table_default = SentimentTable(data)
        dict = self.table_default.to_dic()
        self.assertEqual(2, len(dict))
        self.assertEqual(2, dict['good'])
        dict = self.table_default.to_dic(True)
        self.assertEqual(1, dict['good'])
        self.assertEqual(-0.5, dict['bad'])

    def test_construct_from_dict(self):
        table = LexiconHelper.load_lexicon("inquirer")
        sentiment = SentimentTable.construct_from_dict(table)
        self.assertEquals(3457, len(sentiment.original_data))
        self.assertEquals(1565, len(sentiment.positive))
        self.assertEquals(1892, len(sentiment.negative))

    # preparing to test
    def setUp(self):
        data = [["Good", "1"], ["Bad", "-1"]]
        self.table_default = SentimentTable(data)


class SentimentValueTests(unittest.TestCase):
    def test_construct(self):
        value = SentimentValue(0.5, 0.3)
        self.assertEqual(0.5, value.positive)
        self.assertEqual(0.3, value.negative)
        self.assertEqual(0.29, value.value)

    def test_cases(self):
        self.assertEqual(-0.79, SentimentValue(-0.1, 0.1).value)
        self.assertEqual(-0.61, SentimentValue(0.2, 0.6).value)


