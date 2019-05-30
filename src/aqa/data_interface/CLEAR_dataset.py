import json

import collections
import os


# TODO : Rename to Spectrogram ?
class Image:
    def __init__(self, id, filename, image_builder, which_set):
        self.id = id
        self.filename = filename

        if image_builder is not None:
            self.image_loader = image_builder.build(id, filename=filename, which_set=which_set)

    def get_image(self, **kwargs):
        return self.image_loader.get_image(**kwargs)


class Game(object):
    def __init__(self, id, image, question, answer, question_family_index):
        self.id = id
        self.image = image
        self.question = question
        self.answer = answer
        self.question_family_index = question_family_index

    def __str__(self):
        return "[#q:{}, #p:{}] {} - {} ({})".format(self.id, self.image.id, self.question, self.answer, self.question_family_index)


class CLEARDataset(object):
    """Loads the CLEAR dataset."""

    def __init__(self, folder, which_set, image_builder=None):

        question_file_path = '{}/questions/CLEAR_{}_questions.json'.format(folder, which_set)

        self.games = []
        self.question_family_index = collections.Counter()
        self.answer_counter = collections.Counter()

        with open(question_file_path) as question_file:
            print("Loading questions...")
            data = json.load(question_file)
            info = data["info"]
            samples = data["questions"]

            assert info["set_type"] == which_set

            print("Successfully Loaded AQA v{} ({})".format(info["version"], which_set))

            for sample in samples:

                question_id = int(sample["question_index"])
                question = sample["question"]
                question_family_index = sample.get("question_family_index", -1)  # -1 for test set

                answer = sample.get("answer", None)  # None for test set

                image_id = int(sample["scene_index"])
                image_filename = sample["scene_filename"].replace('.wav', ".png")
                image_filename = os.path.join(which_set, image_filename)

                self.games.append(Game(id=question_id,
                                  image=Image(image_id, image_filename, image_builder, which_set),
                                  question=question,
                                  answer=answer,
                                  question_family_index=question_family_index))

                self.question_family_index[question_family_index] += 1
                self.answer_counter[answer] += 1

        print('{} games loaded...'.format(len(self.games)))

    def get_data(self, indices=[]):
        if len(indices) > 0:
            return [self.games[i] for i in indices]
        else:
            return self.games

    def __len__(self):
        return len(self.games)
