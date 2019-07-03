import json
import math
import random
import collections
import numpy as np
import time
from tqdm import tqdm
import os

from data_interfaces.CLEAR_tokenizer import CLEARTokenizer
from data_interfaces.CLEAR_image_loader import CLEARImage, get_img_builder

class Game(object):
    def __init__(self, id, image, question, answer):
        self.id = id
        self.image = image
        self.question = question
        self.answer = answer

    def __str__(self):
        return "[#q:{}, #p:{}] {} - {}".format(self.id, self.image.id, self.question, self.answer)


class CLEARDataset(object):
    """Loads the CLEAR dataset."""

    def __init__(self, folder, image_config, batch_size, sets=None, dict_file_path=None, tokenize_text=True):
        if sets is None:
            self.sets = ['train', 'val', 'test']
        else:
            self.sets = sets

        if tokenize_text:
            if dict_file_path is None:
                dict_file_path = '{}/preprocessed/dict.json'.format(folder)

            self.tokenizer = CLEARTokenizer(dict_file_path)
        else:
            self.tokenizer = None

        self.root_folder_path = folder
        self.batch_size = batch_size
        self.image_builder = get_img_builder(image_config, folder, bufferize=None)    # TODO : Figure out buffersize

        preprocessed_feature_shape_json_path = "{}/preprocessed/feature_shape.json".format(folder)

        if self.image_builder.is_raw_image():
            self.input_shape = image_config['dim']
        elif os.path.isfile(preprocessed_feature_shape_json_path):
            with open(preprocessed_feature_shape_json_path) as f:
                self.input_shape = json.load(f)['extracted_feature_shape']

        with open("{}/attributes.json".format(folder)) as f:
            attributes = json.load(f)

        self.answer_to_family = {"<unk>": "unknown"}       # FIXME : Quantify what is the impact of having an unknown answer

        for family, answers in attributes.items():
            for answer in answers:
                # If there is duplicated answers, they will be assigned to the first occurring family
                if answer not in self.answer_to_family:
                    self.answer_to_family[answer] = family

        self.games = {}
        self.answer_counter = {}
        self.batchifiers = {}

        for set_type in self.sets:
            self.games[set_type] = []

            question_file_path = '{}/questions/CLEAR_{}_questions.json'.format(folder, set_type)

            self.answer_counter[set_type] = collections.Counter()
            with open(question_file_path) as question_file:
                data = json.load(question_file)
                info = data["info"]
                samples = data["questions"]#[:8*6]

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
                        image_filename = sample["scene_filename"].replace('.wav', ".png")       # The clear dataset specify the filename to the scene wav file
                    else:
                        # Backward compatibility with older CLEVR format
                        image_filename = sample["image_filename"].replace('AQA_', 'CLEAR_')

                    self.games[set_type].append(Game(id=question_id,
                                      image=CLEARImage(image_id, image_filename, self.image_builder, set_type),
                                      question=question,
                                      answer=answer))

                    self.answer_counter[set_type][answer] += 1

        print("Successfully Loaded CLEAR v{} ({}) - {} games loaded.".format(info["version"], ",".join(self.sets), len(self.games)))

    def get_batches(self, set_type, shuffled=True):
        self.batchifiers[set_type] = CLEARBatchifier(self.games[set_type], self.batch_size, self.tokenizer, shuffle=shuffled)
        return self.batchifiers[set_type]

    def get_data(self, indices=[]):
        if len(indices) > 0:
            return [self.games[i] for i in indices]
        else:
            return self.games

    def keep_1_game_per_scene(self):
        for set_type in self.sets:
            id_list = collections.defaultdict(lambda: False)
            unique_scene_games = []
            for game in self.games[set_type]:
                if not id_list[game.image.id]:
                    unique_scene_games.append(game)
                    id_list[game.image.id] = True

            self.games[set_type] = unique_scene_games

    def is_raw_img(self):
        return self.image_builder.is_raw_image()

    def __len__(self):
        return len(self.games)


def create_semaphore_iterator(obj_list, semaphores):
    for obj in obj_list:
        semaphores.acquire()
        yield obj

class CLEARBatchifier(object):
    """Provides an generic multithreaded iterator over the dataset."""

    def __init__(self, games, batch_size, tokenizer, shuffle= True, pad_batches=True):

        self.tokenizer = tokenizer

        if shuffle:
            random.shuffle(games)

        self.games = games
        self.n_examples = len(games)
        self.batch_size = batch_size

        self.n_batches = int(math.ceil(self.n_examples / self.batch_size))

        # Split batches
        i = 0
        batches = []

        while i <= len(games):
            end = min(i + batch_size, len(games))
            batches.append(games[i:end])
            i += batch_size

        # FIXME : Do we really need this ?
        self.nb_padded_in_last_batch = 0
        if pad_batches:
            # Pad last batch if needed
            last_batch_len = len(batches[-1])
            if last_batch_len < batch_size:
                no_missing = batch_size - last_batch_len
                batches[-1] += batches[0][:no_missing]
                self.nb_padded_in_last_batch = no_missing

        self.batches = batches
        self.batch_index = 0

    def get_nb_games(self):
        return sum([len(b) for b in self.batches])

    def load_batch(self, games):
        start_time = time.time()

        batch = collections.defaultdict(list)
        batch_size = len(games)

        assert batch_size > 0

        for i, game in enumerate(games):

            batch["raw"].append(game)

            batch['question'].append(game.question)
            batch['answer'].append(game.answer)

            img_load_start_time = time.time()
            # retrieve the image source type
            img = game.image.get_image()    # FIXME : This is the thing that should be parallelize in a CPU Pool..
            if "image" not in batch: # initialize an empty array for better memory consumption
                batch["image"] = np.zeros((batch_size,) + img.shape, dtype=np.float32)
            batch["image"][i] = img

            #tqdm.write("Image loaded in %f" % (time.time() - img_load_start_time))

        if self.tokenizer:
            batch['question'], batch['seq_length'] = self.tokenizer.pad_tokens(batch['question'])

        tqdm.write("Batch loaded in %f" % (time.time() - start_time))

        return batch

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_index >= self.n_batches:
            raise StopIteration()

        to_return = self.load_batch(self.batches[self.batch_index])
        self.batch_index += 1
        return to_return

    # trick for python 2.X
    def next(self):
        return self.__next__()
