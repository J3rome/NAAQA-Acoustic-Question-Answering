import os
import collections
import multiprocessing as mp
import random

import orjson
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms as viz_transforms

from data_interfaces.CLEAR_image_loader import get_img_builder, CLEARImage
from utils.file import read_json, get_size_from_image_header
from utils.generic import get_answer_to_family_map
from data_interfaces.transforms import ResizeTensorBasedOnMaxWidth, PadTensor

import multiprocessing
import ctypes

from nltk.tokenize import RegexpTokenizer
import re


class CLEAR_dataset(Dataset):

    def __init__(self, folder, version_name, input_image_type, set_type, questions=None, transforms=None,
                 dict_file_path=None, preprocessed_folder_name="preprocessed", tokenize_text=True, extra_stats=False,
                 use_cache=True, synchronized_cache=False, max_cache_size=5000):

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

        # If questions were provided, no need to load questions from file.
        # Only happen when cloning the dataset via CLEAR_dataset.from_dataset_object()
        if questions is None:
            question_file_path = '{}/questions/CLEAR_{}_questions.json'.format(self.root_folder_path, self.set)

            questions = read_json(question_file_path)["questions"]

        scene_file_path = '{}/scenes/CLEAR_{}_scenes.json'.format(self.root_folder_path, self.set)
        if os.path.isfile(scene_file_path):
            scenes_def = {int(s['scene_index']): s for s in read_json(scene_file_path)['scenes']}
        else:
            scenes_def = None

        nb_games = len(questions)
        self.games = multiprocessing.Array(ctypes.c_wchar_p, [""]*nb_games)
        self.games_per_family = collections.defaultdict(list)
        self.scenes = {}
        self.nb_scene = 0
        self.answer_counter = collections.Counter()
        self.answers = []
        self.longest_question_length = 0

        for i, sample in enumerate(questions):
            # For some reason this keep a reference to the questions object (Which take huge amount of memory)
            # Adding 0 ensure that a new object is created.. Kinda ugly hack but it works
            # (copy.deepcopy not working with atomic types)
            question_id = int(sample["question_index"]) + 0
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

            if image_id not in self.scenes:
                self.scenes[image_id] = {
                    'filename': image_filename,
                    'question_idx': []
                }

                self.nb_scene += 1

                if scenes_def:
                    self.scenes[image_id]['definition'] = scenes_def[image_id]

            game = {
                'id': question_id,
                'image': {'id': image_id, 'filename': image_filename, 'set': self.set},
                'question': question,
                'answer': answer
            }

            self.scenes[image_id]['question_idx'].append(question_id)

            if extra_stats:
                self.games_per_family[self.answer_to_family[str(sample['answer']).lower()]].append(question_id)
                game['program'] = sample['program'] if 'program' in sample else []

            self.games[question_id] = self.prepare_game(game)

            self.answers.append(answer)
            self.answer_counter[answer] += 1

        # Initialise image cache
        # We need shared memory and a Lock because this will be updated by each dataloader workers
        self.use_cache = use_cache
        self.synchronized_cache = synchronized_cache

        if self.synchronized_cache:
            self.multiprocessing_manager = mp.Manager()     # FIXME : Using mp.Array is probably more efficient
            self.cache_lock = mp.Lock()
            # FIXME: This is broken, won't focus on it since it doesn't help right now
            self.image_cache = {
                'indexes': self.multiprocessing_manager.list(),
                'images': self.multiprocessing_manager.dict()
            }
        else:
            self.image_cache = {
                'indexes': set(),
                'images': {},
                'size': 0,
                'max_size': min(max_cache_size, self.nb_scene)        # TODO : Set max cache size according to RAM
            }

    @classmethod
    def from_dataset_object(cls, dataset_obj, games=None):
        # Doesn't support cloning a dataset without a tokenizer
        folder_path = dataset_obj.root_folder_path.replace('/%s' % dataset_obj.version_name, '')

        if games:
            questions = []
            for game in games:
                game = orjson.loads(game)
                question = {
                    'question_index': game['id'],
                    'scene_index': game['image']['id'],
                    'scene_filename': game['image']['filename'].replace('.png', '.wav'),
                    'question': dataset_obj.tokenizer.decode_question(game['question']),
                    'answer': dataset_obj.tokenizer.decode_answer(game['answer'])
                }

                if 'program' in game:
                    question['program'] = game["program"]

                questions.append(question)
        else:
            questions = None

        return cls(folder_path, version_name=dataset_obj.version_name,
                   input_image_type=dataset_obj.input_image_type, set_type=dataset_obj.set,
                   questions=questions, dict_file_path=dataset_obj.tokenizer.dictionary_file,
                   transforms=dataset_obj.transforms, max_cache_size=dataset_obj.image_cache['max_size'])

    def get_game(self, idx, decode_tokens=False):
        game = orjson.loads(self.games[idx])
        if not decode_tokens:
            return game
        else:
            game['question'] = self.tokenizer.decode_question(game['question'])
            game['answer'] = self.tokenizer.decode_answer(game['answer'])

            return game

    def prepare_game(self, game):
        return orjson.dumps(game).decode('utf-8')    # orjson dumps return a byte string, we need to convert it to string

    def get_all_image_sizes(self):
        # FIXME : This is broken now that is_raw_img() return True for H5 file.
        assert self.is_raw_img(), 'Config must be set to RAW img in order to retrieve images sizes'

        if self.all_image_sizes is None:
            image_folder = "%s/%s" % (self.image_builder.img_dir, self.set)
            self.all_image_sizes = {idx: get_size_from_image_header(image_folder, s['filename'])
                                    for idx, s in self.scenes.items()}

        return self.all_image_sizes

    def get_random_game_per_family(self, return_game=False):
        assert len(self.games_per_family.keys()) > 0, 'Dataset.games_per_family empty. Need Dataset.extra_stats == True'
        to_return = {}

        for family, games_id in self.games_per_family.items():
            game_id = random.choice(games_id)

            if return_game:
                to_return[family] = self[game_id]
            else:
                to_return[family] = game_id

        return to_return


    def get_random_game(self, return_game=False):
        game_id = np.random.randint(0, len(self.games) - 1)

        if return_game:
            return self[game_id]

        return game_id

    def get_random_game_per_family_for_scene(self, scene_id, return_game=False):
        assert scene_id in self.scenes, f'Scene id {scene_id} is not loaded'

        games_for_scene = self.scenes[scene_id]['question_idx']
        game_per_family = {}

        for game_id in games_for_scene:
            game = self.get_game(game_id, decode_tokens=True)
            family = self.answer_to_family[game['answer']]

            if family in game_per_family:
                # We already have an example of this family
                continue

            if return_game:
                game_per_family[family] = self[game_id]
            else:
                game_per_family[family] = game_id

        for fam in set(self.answer_families) - set(game_per_family.keys()):
            if fam == 'unknown':
                continue
            game_per_family[fam] = None

        return game_per_family

    def get_random_game_for_scene(self, scene_id, return_game=False):
        assert scene_id in self.scenes, f'Scene id {scene_id} is not loaded'

        games_for_scene = self.scenes[scene_id]['question_idx']

        game_id = games_for_scene[np.random.randint(0, len(games_for_scene) - 1)]

        if return_game:
            return self[game_id]

        return game_id

    def get_transformed_dims(self, height, width):
        if self.transforms:
            # Transforms are reversed here because the last transform that affect dimensions contain the output size
            for transform in reversed(self.transforms.transforms):
                if type(transform) is ResizeTensorBasedOnMaxWidth:
                    return transform.get_resized_dim(height, width)
                elif type(transform) is PadTensor:
                    return transform.output_shape

        return height, width

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

        height, max_width = self.get_transformed_dims(height, max_width)

        return to_return + (height, max_width)

    def add_transform(self, transform):
        # Create a new Compose object because initially, the object is shared between all dataset instances
        self.transforms = viz_transforms.Compose(self.transforms.transforms + [transform])

    def __len__(self):
        return len(self.games)

    def load_image_from_cache(self, scene_id, image_filename):
        if self.synchronized_cache:
            # We need to lock the cache to prevent writing race condition from multiple dataloader processes
            self.cache_lock.acquire()

        if scene_id not in self.image_cache['indexes']:
            # Cache bust
            if self.image_cache['size'] >= self.image_cache['max_size']:
                # Deleting oldest entry (Not guaranteed, pop() set doesn't guarantee order)
                cache_idx = self.image_cache['indexes'].pop()
                del self.image_cache['images'][cache_idx]
                self.image_cache['size'] -= 1

            # Load image & Add to cache
            self.image_cache['indexes'].add(scene_id)
            image = CLEARImage(scene_id,
                               image_filename,
                               self.image_builder,
                               self.set).get_image()
            self.image_cache['images'][scene_id] = image
            self.image_cache['size'] += 1

        if self.synchronized_cache:
            # FIXME : Do we loose the advantages of multiprocessing if we lock on reading ? It basically work as if we only had 1 worker ?
            self.cache_lock.release()

        return self.image_cache['images'][scene_id]

    def __getitem__(self, idx):
        requested_game = self.get_game(idx)

        if self.use_cache:
            image = self.load_image_from_cache(requested_game['image']['id'], requested_game['image']['filename'])
        else:
            # Reference to H5py file must be shared between workers (when dataloader.num_workers > 0)
            # We create the image here since it will create the img_builder which contain the h5 file ref
            # See See https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/33
            image = CLEARImage(requested_game['image']['id'],
                               requested_game['image']['filename'],
                               self.image_builder,
                               requested_game['image']['set']).get_image()

        game_with_image = {
            'id': requested_game['id'],
            'image': image,
            'question': torch.tensor(requested_game['question'], dtype=torch.int32),
            'answer': torch.tensor(requested_game['answer']),
            'scene_id': requested_game['image']['id']
        }

        if 'program' in requested_game:
            game_with_image['program'] = requested_game['program']

        # FIXME : Transformed images should be saved in cache instead of the original one..Kinda defeat the purpose of the cache
        if len(self.transforms.transforms) > 0:
            game_with_image = self.transforms(game_with_image)

        if 'image_padding' not in game_with_image:
            # FIXME: We loose padding information when loading from h5
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

        dict_data = read_json(dictionary_file)
        self.word2i = dict_data['word2i']
        self.answer2i = dict_data['answer2i']

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
