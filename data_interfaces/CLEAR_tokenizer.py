from nltk.tokenize import RegexpTokenizer
import numpy as np
import json
import re


class CLEARTokenizer:
    """ """
    def __init__(self, dictionary_file):

        self.tokenizer = CLEARTokenizer.get_tokenizer_inst()

        with open(dictionary_file, 'r') as f:
            data = json.load(f)
            self.word2i = data['word2i']
            self.answer2i = data['answer2i']

        self.dictionary_file = dictionary_file

        self.i2word = {}
        for (k, v) in self.word2i.items():
            self.i2word[v] = k

        self.i2answer = {}
        for (k, v) in self.answer2i.items():
            self.i2answer[v] = k

        # Retrieve key values
        self.no_words = len(self.word2i)
        self.no_answers = len(self.answer2i)

        self.unknown_question_token = self.word2i["<unk>"]
        self.padding_token = self.word2i["<padding>"]

        #self.padding_answer = self.answer2i["<padding>"]
        self.unknown_answer = self.answer2i["<unk>"]

    @staticmethod
    def get_tokenizer_inst():
        tokenizer_patterns = r"""
                        (?:[^\W\d_](?:[^\W\d_]|['\-_])+[^\W\d_]) # Words with apostrophes or dashes.
                        |
                        (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
                        |
                        (?:[a-z]\#)                    # Musical Notes (Ex : D#, F#)
                        |
                        (?:[\w_]+)                     # Words without apostrophes or dashes.
                        |
                        (?:\.(?:\s*\.){1,})            # Ellipsis dots.
                        |
                        (?:\S)                         # Everything else that isn't whitespace.
                        """

        return RegexpTokenizer(tokenizer_patterns, flags=re.VERBOSE | re.I | re.UNICODE)

    """
    Input: String
    Output: List of tokens
    """
    def encode_question(self, question):
        tokens = []
        for token in self.tokenizer.tokenize(question):
            if token not in self.word2i:
                token = '<unk>'
            tokens.append(self.word2i[token])
        return tokens

    def decode_question(self, tokens):
        return ' '.join([self.i2word[tok] for tok in tokens])

    def encode_answer(self, answer):
        if answer not in self.answer2i:
            return self.answer2i['<unk>']
        return self.answer2i[answer]

    def decode_answer(self, answer_id):
        return self.i2answer[answer_id]

    def tokenize_question(self, question):
        return self.tokenizer.tokenize(question)

    @staticmethod
    def pad_tokens(list_of_tokens, padding_token=0, seq_length=None, max_seq_length=0):

        if seq_length is None:
            seq_length = np.array([len(q) for q in list_of_tokens], dtype=np.int32)

        if max_seq_length == 0:
            max_seq_length = seq_length.max()

        batch_size = len(list_of_tokens)

        padded_tokens = np.full(shape=(batch_size, max_seq_length), fill_value=padding_token)

        for i, seq in enumerate(list_of_tokens):
            seq = seq[:max_seq_length]
            padded_tokens[i, :len(seq)] = seq

        return padded_tokens, seq_length




