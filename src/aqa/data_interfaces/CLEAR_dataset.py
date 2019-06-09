import json
import math
import random
import collections
import numpy as np
from multiprocessing import Semaphore
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool
import os

from aqa.data_interfaces.CLEAR_tokenizer import CLEARTokenizer
from aqa.data_interfaces.CLEAR_image_loader import CLEARImage, get_img_builder

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

        self.image_builder = get_img_builder(image_config, folder, bufferize=None)    # TODO : Figure out buffersize

        self.games = {}
        self.answer_counter = {}
        self.batchifiers = {}

        for set in self.sets:
            self.games[set] = []

            question_file_path = '{}/questions/CLEAR_{}_questions.json'.format(folder, set)

            self.answer_counter[set] = collections.Counter()
            with open(question_file_path) as question_file:
                data = json.load(question_file)
                info = data["info"]
                samples = data["questions"]

                for sample in samples:

                    question_id = int(sample["question_index"])
                    question = self.tokenizer.encode_question(sample["question"]) if tokenize_text else sample['question']

                    answer = sample.get("answer", None)  # None for test set
                    if answer is not None:
                        answer = self.tokenizer.encode_answer(answer) if tokenize_text else answer


                    image_id = int(sample["scene_index"])
                    image_filename = sample["scene_filename"].replace('.wav', ".png")       # The clear dataset specify the filename to the scene wav file

                    self.games[set].append(Game(id=question_id,
                                      image=CLEARImage(image_id, image_filename, self.image_builder, set),
                                      question=question,
                                      answer=answer))

                    self.answer_counter[set][answer] += 1

            self.batchifiers[set] = CLEARBatchifier(self.games[set], batch_size, self.tokenizer)

        print("Successfully Loaded CLEAR v{} ({}) - {} games loaded.".format(info["version"], ",".join(self.sets), len(self.games)))

    def get_batches(self, set):
        return self.batchifiers[set]

    def get_data(self, indices=[]):
        if len(indices) > 0:
            return [self.games[i] for i in indices]
        else:
            return self.games

    def __len__(self):
        return len(self.games)


class CLEARBatchifier(object):
    """Provides an generic multithreaded iterator over the dataset."""

    def __init__(self, games, batch_size, tokenizer, nb_thread=3,   # FIXME : Make nb_thread param pop up
                 shuffle= True, no_semaphore= 40, pad_batches=True):         # FIXME :No semaphore seem to be the same as batch size

        # Define CPU pool
        # CPU/GPU option
        # h5 requires a Tread pool while raw images requires to create new process

        #if image_builder.is_raw_image():
        #    cpu_pool = Pool(nb_thread, maxtasksperchild=1000)
        pool = ThreadPool(nb_thread)
        pool._maxtasksperchild = 1000

        self.tokenizer = tokenizer

        if shuffle:
            random.shuffle(games)

        self.n_examples = len(games)
        self.batch_size = batch_size

        self.n_batches = int(math.ceil(1. * self.n_examples / self.batch_size))

        # Split batches
        i = 0
        batches = []

        while i <= len(games):
            end = min(i + batch_size, len(games))
            batches.append(games[i:end])
            i += batch_size

        # FIXME : Do we really need this ?
        if pad_batches:
            # Pad last batch if needed
            last_batch_len = len(batches[-1])
            if last_batch_len < batch_size:
                no_missing = batch_size - last_batch_len
                batches[-1] += batches[0][:no_missing]


        # Multi_proc iterator
        def create_semaphore_iterator(obj_list, semaphores):
            for obj in obj_list:
                semaphores.acquire()
                yield obj

        self.semaphores = Semaphore(no_semaphore)
        batches = create_semaphore_iterator(batches, self.semaphores)
        self.process_iterator = pool.imap(self.load_batch, batches)

    def load_batch(self, games):

        batch = collections.defaultdict(list)
        batch_size = len(games)

        assert batch_size > 0

        for i, game in enumerate(games):

            #batch["raw"].append(game)

            batch['question'].append(game.question)
            batch['answer'].append(game.answer)

            # retrieve the image source type
            img = game.image.get_image()    # FIXME : This is the thing that should be parallelize in a CPU Pool..
            if "image" not in batch: # initialize an empty array for better memory consumption
                batch["image"] = np.zeros((batch_size,) + img.shape, dtype=np.float32)
            batch["image"][i] = img

        batch['question'], batch['seq_length'] = self.tokenizer.pad_tokens(batch['question'])

        return batch

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        return self

    def __next__(self):
        self.semaphores.acquire()
        ret_val = self.process_iterator.next()
        self.semaphores.release()
        return ret_val

    # trick for python 2.X
    def next(self):
        return self.__next__()
