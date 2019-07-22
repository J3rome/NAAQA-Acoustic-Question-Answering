import os
import collections

import ujson
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision.transforms.functional as F

from data_interfaces.CLEAR_tokenizer import CLEARTokenizer
from data_interfaces.CLEAR_image_loader import get_img_builder, CLEARImage
from utils import read_json


# FIXME : Move transforms out of here
# Transforms
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because (RGB/BGR)
        # numpy image: H x W x C
        # torch image: C X H X W
        image = np.array(sample['image']).transpose((2, 0, 1))      # FIXME : Transpose in pytorch
        return {
            'image': torch.from_numpy(image).float().div(255),      # FIXME : Normalization should be done in an independent transform
            'question': torch.from_numpy(sample['question']).int(),
            'answer': torch.from_numpy(sample['answer']),
            'id': sample['id'],             # Not processed by the network, no need to transform to tensor.. Seems to be transfered to tensor in collate function anyways
            'scene_id': sample['scene_id']  # Not processed by the network, no need to transform to tensor
        }


class ResizeImg(object):
    def __init__(self, output_shape):
        self.output_shape = output_shape

    def __call__(self, sample):
        sample['image'] = F.resize(sample['image'], self.output_shape)
        return sample


class CLEAR_dataset(Dataset):

    def __init__(self, folder, image_config, set, questions=None, transforms=None, dict_file_path=None, tokenize_text=True):
        preprocessed_folder_path = '{}/preprocessed'.format(folder)

        if tokenize_text:
            if dict_file_path is None:
                dict_file_path = '{}/dict.json'.format(preprocessed_folder_path)

            self.tokenizer = CLEARTokenizer(dict_file_path)
        else:
            self.tokenizer = None

        self.root_folder_path = folder
        self.set = set
        self.image_builder = get_img_builder(image_config, folder, bufferize=None)
        self.transforms = transforms

        feature_shape_filename = "feature_shape.json"

        if self.image_builder.is_raw_image():
            self.input_shape = image_config['dim']
        elif os.path.isfile("{}/{}".format(preprocessed_folder_path, feature_shape_filename)):
            self.input_shape = read_json(preprocessed_folder_path, feature_shape_filename)['extracted_feature_shape']

        attributes = read_json(folder, 'attributes.json')

        self.answer_to_family = {"<unk>": "unknown"}  # FIXME : Quantify what is the impact of having an unknown answer

        for family, answers in attributes.items():
            for answer in answers:
                # If there is duplicated answers, they will be assigned to the first occurring family
                if answer not in self.answer_to_family:
                    self.answer_to_family[answer] = family

        self.games = []
        self.answer_counter = collections.Counter()
        self.answers = []

        # Questions can either be read from file or provided as an array
        if questions is None:
            question_file_path = '{}/questions/CLEAR_{}_questions.json'.format(folder, self.set)

            with open(question_file_path) as question_file:
                samples = ujson.load(question_file)["questions"]
        else:
            samples = questions

        for sample in samples:
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

            self.games.append({
                'id': question_id,
                'image': {'id': image_id, 'filename': image_filename, 'set': self.set},
                'question': question,
                'answer': answer
            })

            self.answers.append(answer)

            self.answer_counter[answer] += 1

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        requested_game = self.games[idx]

        # Reference to H5py file must be shared between workers (when dataloader.num_workers > 0)
        # We create the image here since it will create the img_builder which contain the h5 file ref
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

    # FIXME : We should probably pad the sequence using torch methods
    # FIXME : Investigate the Packing Method.
    # See https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
    #     https://discuss.pytorch.org/t/simple-working-example-how-to-use-packing-for-variable-length-sequence-inputs-for-rnn/2120/33
    def CLEAR_collate_fct(self, batch):
        batch_questions = [b['question'] for b in batch]

        padded_questions, seq_lengths = self.tokenizer.pad_tokens(batch_questions)

        for sample, padded_question, seq_length in zip(batch, padded_questions, seq_lengths):
            sample['question'] = padded_question
            sample['seq_length'] = seq_length

        return torch.utils.data.dataloader.default_collate(batch)


if __name__ == "__main__":
    image_config = {
            "type": "raw",
            "dim": [224, 224, 3]
        }

    test_dataset = CLEAR_dataset('data/v2.0.0_1k_scenes_1_inst_per_scene', image_config, 'train', transforms=ToTensor())

    dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=test_dataset.CLEAR_collate_fct)


    batch = next(iter(dataloader))

    print("done")
