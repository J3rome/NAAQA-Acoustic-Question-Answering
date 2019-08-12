import os
import collections

import ujson
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from data_interfaces.CLEAR_tokenizer import CLEARTokenizer
from data_interfaces.CLEAR_image_loader import get_img_builder, CLEARImage
from utils import read_json

import multiprocessing
import ctypes


class CLEAR_dataset(Dataset):

    def __init__(self, folder, version_name, image_config, set, questions=None,
                 transforms=None, dict_file_path=None, tokenize_text=True):

        self.root_folder_path = "%s/%s" % (folder, version_name)
        self.version_name = version_name

        preprocessed_folder_path = '{}/preprocessed'.format(self.root_folder_path)

        if tokenize_text:
            if dict_file_path is None:
                dict_file_path = '{}/dict.json'.format(preprocessed_folder_path)

            self.tokenizer = CLEARTokenizer(dict_file_path)
        else:
            self.tokenizer = None

        self.set = set
        self.image_builder = get_img_builder(image_config, self.root_folder_path, bufferize=None)
        self.transforms = transforms

        feature_shape_filename = "feature_shape.json"

        if self.image_builder.is_raw_image():
            self.input_shape = image_config['dim']
        elif os.path.isfile("{}/{}".format(preprocessed_folder_path, feature_shape_filename)):
            self.input_shape = read_json(preprocessed_folder_path, feature_shape_filename)['extracted_feature_shape']

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

            self.answers.append(answer)

            self.answer_counter[answer] += 1

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
        # Our input_shape is in the "regular image" format
        if channel_first:
            return [self.input_shape[2]] + self.input_shape[:2]

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

        # FIXME : Investigate why this doesnt work
        #seq_lengths = torch.tensor([len(q) for q in batch_questions])
        #padded_questions = torch.nn.utils.rnn.pad_sequence(batch_questions, batch_first=True)

        for sample, padded_question, seq_length in zip(batch, padded_questions, seq_lengths):
            sample['question'] = padded_question
            sample['seq_length'] = seq_length

        return torch.utils.data.dataloader.default_collate(batch)

if __name__ == "__main__":
    print("Please use main.py")
