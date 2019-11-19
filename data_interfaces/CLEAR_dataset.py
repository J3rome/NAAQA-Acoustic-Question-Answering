import os
import collections

import ujson
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms

from data_interfaces.CLEAR_image_loader import get_img_builder, CLEARImage
from utils.file import read_json, get_size_from_image_header
from utils.generic import get_answer_to_family_map
from data_interfaces.transforms import ResizeImgBasedOnHeight, ResizeImgBasedOnWidth

import multiprocessing
import ctypes

from nltk.tokenize import RegexpTokenizer
import re


class CLEAR_dataset(Dataset):

    def __init__(self, folder, version_name, input_image_type, set_type, questions=None, transforms=None,
                 dict_file_path=None, load_question_program=False, preprocessed_folder_name="preprocessed",
                 tokenize_text=True):

        self.root_folder_path = "%s/%s" % (folder, version_name)
        self.version_name = version_name

        if tokenize_text and dict_file_path is not None:
            self.tokenizer = CLEARTokenizer(dict_file_path)
        else:
            self.tokenizer = None
            tokenize_text = False

        self.set = set_type
        self.input_image_type = input_image_type
        self.image_builder = get_img_builder(input_image_type, self.root_folder_path,
                                             preprocessed_folder_name=preprocessed_folder_name, bufferize=None)
        self.transforms = transforms
        self.input_shape = None
        self.all_image_sizes = None

        self.answer_to_family = get_answer_to_family_map('%s/%s' % (self.root_folder_path, 'attributes.json'))
        self.answer_families = list(set(self.answer_to_family.values()))

        # Questions can either be read from file or provided as an array
        if questions is None:
            question_file_path = '{}/questions/CLEAR_{}_questions.json'.format(self.root_folder_path, self.set)

            self.questions = read_json(question_file_path)["questions"]
        else:
            self.questions = questions

        scene_file_path = '{}/scenes/CLEAR_{}_scenes.json'.format(self.root_folder_path, self.set)
        if os.path.isfile(scene_file_path):
            self.scenes_def = {int(s['scene_index']): s for s in read_json(scene_file_path)['scenes']}
        else:
            self.scenes_def = None

        nb_games = len(self.questions)
        self.games = multiprocessing.Array(ctypes.c_wchar_p, [""]*nb_games)
        self.scenes = {}
        self.answer_counter = collections.Counter()
        self.answers = []
        self.longest_question_length = 0

        for i, sample in enumerate(self.questions):
            question_id = int(sample["question_index"])
            question = self.tokenizer.encode_question(sample["question"]) if tokenize_text else sample['question']

            if tokenize_text:
                # Remove the <start> and <end> tokens from the count
                nb_word_in_question = len(question) - 2 if self.tokenizer.start_token else len(question)
            else:
                nb_word_in_question = len(question.split(' '))

            if nb_word_in_question > self.longest_question_length:
                self.longest_question_length = nb_word_in_question

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

            game = {
                'id': question_id,
                'image': {'id': image_id, 'filename': image_filename, 'set': self.set},
                'question': question,
                'answer': answer
            }

            if load_question_program:
                game['program'] = sample['program'] if 'program' in sample else []

            self.games[i] = self.prepare_game(game)

            if image_id not in self.scenes:
                self.scenes[image_id] = {
                    'filename': image_filename,
                    'question_idx': [i]
                }

                if self.scenes_def:
                    self.scenes[image_id]['definition'] = self.scenes_def[image_id]

            else:
                self.scenes[image_id]['question_idx'].append(i)

            self.answers.append(answer)

            self.answer_counter[answer] += 1
            
    @classmethod
    def from_dataset_object(cls, dataset_obj, questions):
        folder_path = dataset_obj.root_folder_path.replace('/%s' % dataset_obj.version_name, '')
        return cls(folder_path, version_name=dataset_obj.version_name,
                   input_image_type=dataset_obj.input_image_type, set_type=dataset_obj.set,
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

        if self.all_image_sizes is None:
            image_folder = "%s/%s" % (self.image_builder.img_dir, self.set)
            self.all_image_sizes = {idx: get_size_from_image_header(image_folder, s['filename'])
                                    for idx, s in self.scenes.items()}

        return self.all_image_sizes

    def get_game_id_for_scene(self, scene_id):
        assert scene_id in self.scenes

        return self.scenes[scene_id]['question_idx'][0]

    def get_resize_config(self):
        value = None
        resize_type = None

        if self.transforms:
            for transform in self.transforms.transforms:
                if type(transform) is ResizeImgBasedOnHeight:
                    value = transform.output_height
                    resize_type = 'height'
                elif type(transform) is ResizeImgBasedOnWidth:
                    value = transform.output_width
                    resize_type = 'width'

        return value, resize_type

    def get_min_width_image_dims(self, return_scene_id=False):
        return self._get_minmax_width_image_dims(return_scene_id, minmax_fct=min)

    def get_max_width_image_dims(self, return_scene_id=False):
        return self._get_minmax_width_image_dims(return_scene_id, minmax_fct=max)

    def _get_minmax_width_image_dims(self, return_scene_id=False, minmax_fct=max):
        image_sizes = self.get_all_image_sizes()

        height, max_width = minmax_fct(image_sizes.values(), key=lambda x: x[1])

        if return_scene_id:
            to_return = [(game_idx,) for game_idx, dim in image_sizes.items()
                         if dim[0] == height and dim[1] == max_width][0]
        else:
            to_return = tuple()

        resized_val, resized_dim = self.get_resize_config()

        if resized_val:
            if resized_dim == 'height':
                max_width = int(resized_val * max_width / height)
                height = resized_val
            elif resized_dim == 'width':
                height = int(resized_val * height / max_width)
                max_width = resized_val

        return to_return + (height, max_width)

    def add_transform(self, transform):
        # Create a new Compose object because initially, the object is shared between all dataset instances
        self.transforms = transforms.Compose(self.transforms.transforms + [transform])

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

        if 'program' in requested_game:
            game_with_image['program'] = requested_game['program']

        if self.transforms:
            game_with_image = self.transforms(game_with_image)

        if 'image_padding' not in game_with_image:
            game_with_image['image_padding'] = torch.tensor([0, 0], dtype=torch.int)

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
        if self.input_shape is None:
            # NOTE : The width might vary (Since we use the shape of the first example).
            #        To retrieve size of all images use self.get_all_image_sizes()
            self.input_shape = self[0]['image'].shape

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

        if "<start>" in self.word2i and "<end>" in self.word2i:
            self.start_token = self.word2i['<start>']
            self.end_token = self.word2i['<end>']
        else:
            self.start_token = None
            self.end_token = None

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
    def encode_question(self, question, to_lowercase=True):
        if to_lowercase:
            question = ' '.join([w.lower() for w in question.split(' ')])

        tokens = []
        for token in self.tokenizer.tokenize(question):
            if token not in self.word2i:
                token = '<unk>'
            tokens.append(self.word2i[token])

        if self.start_token:
            tokens = [self.start_token] + tokens + [self.end_token]

        return tokens

    def decode_question(self, tokens, remove_padding=False):
        if remove_padding:
            tokens = [tok for tok in tokens if tok != self.padding_token]

        if self.start_token:
            tokens = tokens[1:-1]

        return ' '.join([self.i2word[tok] for tok in tokens])

    def encode_answer(self, answer, to_lowercase=True):
        if to_lowercase:
            answer = answer.lower()

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

            if 'image_padding' not in sample:
                sample['image_padding'] = torch.tensor([height_to_pad, width_to_pad], dtype=torch.int)
            else:
                sample['image_padding'][0] += height_to_pad
                sample['image_padding'][1] += width_to_pad

        return torch.utils.data.dataloader.default_collate(batch)

if __name__ == "__main__":
    print("Please use main.py")
