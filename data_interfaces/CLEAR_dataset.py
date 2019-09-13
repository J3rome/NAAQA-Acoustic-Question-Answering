import os
import collections

import ujson
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from data_interfaces.CLEAR_image_loader import get_img_builder, CLEARImage
from utils import read_json, get_size_from_image_header

import multiprocessing
import ctypes

from nltk.tokenize import RegexpTokenizer
import re


class CLEAR_dataset(Dataset):

    def __init__(self, folder, version_name, image_config, set_type, questions=None,
                 preprocessed_folder_name="preprocessed", transforms=None, dict_file_path=None, tokenize_text=True):

        self.root_folder_path = "%s/%s" % (folder, version_name)
        self.version_name = version_name

        preprocessed_folder_path = '{}/{}'.format(self.root_folder_path, preprocessed_folder_name)

        if tokenize_text and dict_file_path is not None:
            self.tokenizer = CLEARTokenizer(dict_file_path)
        else:
            self.tokenizer = None
            tokenize_text = False

        self.set = set_type
        self.image_config = image_config
        self.image_builder = get_img_builder(image_config, self.root_folder_path,
                                             preprocessed_folder_name=preprocessed_folder_name, bufferize=None)
        self.transforms = transforms

        attributes = read_json(self.root_folder_path, 'attributes.json')

        self.answer_to_family = {"<unk>": "unknown"}  # FIXME : Quantify what is the impact of having an unknown answer

        for family, answers in attributes.items():
            for answer in answers:
                # If there is duplicated answers, they will be assigned to the first occurring family
                if answer not in self.answer_to_family:
                    self.answer_to_family[answer] = family

        # Questions can either be read from file or provided as an array
        if questions is None:
            question_file_path = '{}/questions/CLEAR_{}_questions.json'.format(self.root_folder_path, self.set)

            with open(question_file_path) as question_file:
                self.questions = ujson.load(question_file)["questions"]
        else:
            self.questions = questions

        nb_games = len(self.questions)
        self.games = multiprocessing.Array(ctypes.c_wchar_p, [""]*nb_games)
        self.scenes = {}
        self.answer_counter = collections.Counter()
        self.answers = []

        for i, sample in enumerate(self.questions):
            question_id = int(sample["question_index"])
            question = self.tokenizer.encode_question(sample["question"]) if tokenize_text else sample['question']

            answer = sample.get("answer", None)  # None for test set
            if answer is not None:
                answer = str(answer) if type(answer) == int else answer
                answer = self.tokenizer.encode_answer(answer) if tokenize_text else answer

            if 'scene_index' in sample:
                image_id = int(sample["scene_index"])
            else:
                # Backward compatibility with older CLEVR format
                image_id = int(sample["image_index"])

            if "scene_filename" in sample:
                image_filename = sample["scene_filename"].replace('.wav', ".png")  # The clear dataset specify the filename to the scene wav file
            else:
                # Backward compatibility with older CLEVR format
                image_filename = sample["image_filename"].replace('AQA_', 'CLEAR_')

            self.games[i] = self.prepare_game({
                'id': question_id,
                'image': {'id': image_id, 'filename': image_filename, 'set': self.set},
                'question': question,
                'answer': answer
            })

            if image_id not in self.scenes:
                self.scenes[image_id] = image_filename

            self.answers.append(answer)

            self.answer_counter[answer] += 1

        # NOTE : The width might vary (Since we use the shape of the first example).
        #        To retrieve size of all images use self.get_all_image_sizes()
        self.input_shape = self[0]['image'].shape
            
    @classmethod
    def from_dataset_object(cls, dataset_obj, questions):
        folder_path = dataset_obj.root_folder_path.replace('/%s' % dataset_obj.version_name, '')
        return cls(folder_path, version_name=dataset_obj.version_name,
                   image_config=dataset_obj.image_config, set_type=dataset_obj.set,
                   questions=questions, dict_file_path=dataset_obj.tokenizer.dictionary_file,
                   transforms=dataset_obj.transforms)

    def get_game(self, idx, decode_tokens=False):
        game = ujson.loads(self.games[idx])
        if not decode_tokens:
            return game
        else:
            game['question'] = self.tokenizer.decode_question(game['question'])
            game['answer'] = self.tokenizer.decode_answer(game['answer'])

            return game

    def prepare_game(self, game):
        return ujson.dumps(game)

    def get_all_image_sizes(self):
        assert self.is_raw_img(), 'Config must be set to RAW img in order to retrieve images sizes'

        image_folder = "%s/%s" % (self.image_builder.img_dir, self.set)

        return {idx: get_size_from_image_header(image_folder, filepath) for idx, filepath in self.scenes.items()}

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        requested_game = self.get_game(idx)

        # Reference to H5py file must be shared between workers (when dataloader.num_workers > 0)
        # We create the image here since it will create the img_builder which contain the h5 file ref
        # See See https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/33
        image = CLEARImage(requested_game['image']['id'],
                           requested_game['image']['filename'],
                           self.image_builder,
                           requested_game['image']['set'])

        game_with_image = {
            'id': requested_game['id'],
            'image': image.get_image(),
            'question': np.array(requested_game['question']),
            'answer': np.array(requested_game['answer']),
            'scene_id': image.id
        }

        if self.transforms:
            game_with_image = self.transforms(game_with_image)

        return game_with_image

    def is_raw_img(self):
        return self.image_builder.is_raw_image()

    def get_token_counts(self):
        if self.tokenizer:
            return self.tokenizer.no_words, self.tokenizer.no_answers
        return None, None

    def get_padding_token(self):
        if self.tokenizer:
            return self.tokenizer.padding_token
        return 0

    def get_input_shape(self, channel_first=True):
        # Regular images : H x W x C
        # Torch images : C x H x W
        # Our input_shape is in the "Torch image" format
        if not channel_first:
            return self.input_shape[-2:] + [self.input_shape[0]]

        return self.input_shape

    def keep_1_game_per_scene(self):
        id_list = collections.defaultdict(lambda: False)
        unique_scene_games = []
        for game_idx in range(len(self.games)):
            game = self.get_game(game_idx)
            if not id_list[game['image']['id']]:
                unique_scene_games.append(self.prepare_game(game))
                id_list[game['image']['id']] = True

        self.games = unique_scene_games


class CLEARTokenizer:
    """ """
    def __init__(self, dictionary_file):

        self.tokenizer = CLEARTokenizer.get_tokenizer_inst()

        with open(dictionary_file, 'r') as f:
            data = ujson.load(f)
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


# FIXME : We should probably pad the sequence using torch methods
# FIXME : Investigate the Packing Method.
# See https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
#     https://discuss.pytorch.org/t/simple-working-example-how-to-use-packing-for-variable-length-sequence-inputs-for-rnn/2120/33
class CLEAR_collate_fct(object):
    def __init__(self, padding_token):
        self.padding_token = padding_token

    def __call__(self, batch):
        batch_questions = [b['question'] for b in batch]

        padded_questions, seq_lengths = CLEARTokenizer.pad_tokens(batch_questions, padding_token=self.padding_token)

        image_dims = [sample['image'].shape[1:] for sample in batch]
        max_image_width = max(image_dims, key=lambda x: x[1])[1]
        max_image_height = max(image_dims, key=lambda x: x[0])[0]

        # FIXME : Investigate why this doesnt work
        #seq_lengths = torch.tensor([len(q) for q in batch_questions])
        #padded_questions = torch.nn.utils.rnn.pad_sequence(batch_questions, batch_first=True)

        for sample, padded_question, seq_length in zip(batch, padded_questions, seq_lengths):
            sample['question'] = padded_question
            sample['seq_length'] = seq_length

            width_to_pad = max_image_width - sample['image'].shape[2]
            height_to_pad = max_image_height - sample['image'].shape[1]

            if width_to_pad + height_to_pad > 0:
                sample['image'] = F.pad(sample['image'], [0, width_to_pad, 0, height_to_pad])

        return torch.utils.data.dataloader.default_collate(batch)

if __name__ == "__main__":
    print("Please use main.py")
