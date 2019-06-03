from nltk.tokenize import RegexpTokenizer
import re
import io
import json
import collections
import argparse

from aqa.data_interfaces.CLEAR_dataset import CLEARDataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Creating dictionary..', fromfile_prefix_chars='@')

    parser.add_argument("-data_dir", type=str, help="Path to VQA dataset")
    parser.add_argument("-dict_file", type=str, default="dict.json", help="Name of the dictionary file")
    parser.add_argument("-min_occ", type=int, default=1, help='Minimum number of occurences to add word to dictionary (for Human Clevr)')
    args = parser.parse_args()

    dataset = CLEARDataset(args.data_dir, which_set="train")
    games = dataset.games

    word2i = {'<padding>': 0,
              '<unk>': 1
              }

    answer2i = {'<padding>': 0,
                '<unk>': 1
                }

    answer2occ = dataset.answer_counter
    word2occ = collections.defaultdict(int)


    # Input words
    tokenizerPatterns = r"""
        (?:[^\W\d_](?:[^\W\d_]|['\-_])+[^\W\d_]) # Words with apostrophes or dashes.
        |
        (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
        |
        (?:[a-z]\#)                     # Notes (Ex : D#, F#)
        |
        (?:[\w_]+)                     # Words without apostrophes or dashes.
        |
        (?:\.(?:\s*\.){1,})            # Ellipsis dots.
        |
        (?:\S)                         # Everything else that isn't whitespace.
        """

    tokenizer = RegexpTokenizer(tokenizerPatterns, flags=re.VERBOSE | re.I | re.UNICODE)

    for game in games:
        input_tokens = tokenizer.tokenize(game.question)
        for tok in input_tokens:
            word2occ[tok] += 1

    # parse the questions
    for word, occ in word2occ.items():
        if occ >= args.min_occ:
            word2i[word] = len(word2i)

    # parse the answers
    for answer in answer2occ.keys():
        answer2i[answer] = len(answer2i)

    print("Number of words): {}".format(len(word2i)))
    print("Number of answers: {}".format(len(answer2i)))

    with io.open(args.dict_file, 'w', encoding='utf8') as f_out:
       data = json.dumps({'word2i': word2i, 'answer2i': answer2i})
       f_out.write(data)
