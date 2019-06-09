from nltk.tokenize import RegexpTokenizer
import os
import re
import io
import json
import collections
import argparse

from aqa.data_interfaces.CLEAR_dataset import CLEARDataset
from aqa.data_interfaces.CLEAR_tokenizer import CLEARTokenizer

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Creating dictionary..', fromfile_prefix_chars='@')

    parser.add_argument("-data_dir", type=str, help="Path to VQA dataset")
    parser.add_argument("-dict_file", type=str, default="dict.json", help="Name of the dictionary file")
    parser.add_argument("-min_occ", type=int, default=1, help='Minimum number of occurences to add word to dictionary (for Human Clevr)')
    args = parser.parse_args()

    dataset = CLEARDataset(args.data_dir, {'type': 'raw', 'dim': [224, 224, 3]}, 32, sets=['train'], tokenize_text=False)

    # FIXME : Should we use the whole dataset to create the dictionary ?
    games = dataset.games['train']

    word2i = {'<padding>': 0,
              '<unk>': 1
              }

    answer2i = {'<padding>': 0,
                '<unk>': 1
                }

    answer2occ = dataset.answer_counter['train']
    word2occ = collections.defaultdict(int)

    tokenizer = CLEARTokenizer.get_tokenizer_inst()

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

    preprocessed_folder_path = os.path.join(args.data_dir, 'preprocessed')
    dict_file_path = os.path.join(preprocessed_folder_path, 'dict.json')

    if not os.path.isdir(preprocessed_folder_path):
        os.mkdir(preprocessed_folder_path)

    with io.open(dict_file_path, 'w', encoding='utf8') as f_out:
       data = json.dumps({'word2i': word2i, 'answer2i': answer2i})
       f_out.write(data)
