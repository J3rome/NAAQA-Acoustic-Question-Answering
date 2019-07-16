import os
import collections
import copy
import time

import ujson
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision

from torchvision import transforms

from data_interfaces.CLEAR_tokenizer import CLEARTokenizer
from data_interfaces.CLEAR_image_loader import get_img_builder, CLEARImage
from utils import read_json


# FIXME : We should probably pad the sequence using torch methods
# FIXME : Investigate the Packing Method.
# See https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
#     https://discuss.pytorch.org/t/simple-working-example-how-to-use-packing-for-variable-length-sequence-inputs-for-rnn/2120/33
def create_collate_fct(tokenizer):

    def CLEAR_collate(batch):
        batch_questions = [b['question'] for b in batch]

        padded_questions, seq_length = tokenizer.pad_tokens(batch_questions)

        for sample, padded_question in zip(batch, padded_questions):
            sample['question'] = padded_question



        return torch.utils.data.dataloader.default_collate(batch)

    return CLEAR_collate


# Transforms
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because (RGB/BGR)
        # numpy image: H x W x C
        # torch image: C X H X W
        image = sample['image'].transpose((2, 0, 1))
        return {
            'image': torch.from_numpy(image).float().div(255),
            #'image': torchvision.transforms.functional.to_tensor(image),
            'question': torch.from_numpy(sample['question']),
            'answer': torch.from_numpy(sample['answer'])
        }


class CLEAR_dataset(Dataset):

    def __init__(self, folder, image_config, batch_size, set, transforms=None, dict_file_path=None, tokenize_text=True):
        # TODO : I think we should have 1 dataset object for each sets (Train. val test)

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

        question_file_path = '{}/questions/CLEAR_{}_questions.json'.format(folder, self.set)

        with open(question_file_path) as question_file:
            data = ujson.load(question_file)
            info = data["info"]
            samples = data["questions"]  # [:8*6]

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
                    image_filename = sample["scene_filename"].replace('.wav',
                                                                      ".png")  # The clear dataset specify the filename to the scene wav file
                else:
                    # Backward compatibility with older CLEVR format
                    image_filename = sample["image_filename"].replace('AQA_', 'CLEAR_')

                self.games.append({
                    'id': question_id,
                    'image': CLEARImage(image_id, image_filename, self.image_builder, self.set),
                    'question': question,
                    'answer': answer
                })

                self.answers.append(answer)

                self.answer_counter[answer] += 1

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        requested_game = self.games[idx]

        # FIXME : should the answer be included here ? (Seems to be in the .classes object)
        # FIXME : Transform should be applied here

        #self.tokenizer.pad_tokens(requested_game['question'])

        game_with_image = {
            'id': requested_game['id'],
            'image': requested_game['image'].get_image(),
            'question': np.array(requested_game['question']),
            'answer': np.array(requested_game['answer'])
        }

        if self.transforms:
            game_with_image = self.transforms(game_with_image)

        return game_with_image

    def is_raw_img(self):
        return self.image_builder.is_raw_image()


def train_model(device, model, dataloader, criterion=None, optimizer=None, scheduler=None, num_epochs=25):
    since = time.time()

    #best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    dataset_size = len(dataloader.dataset)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        #scheduler.step()
        #model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        for batch in dataloader:
            images = batch['image'].to(device)
            questions = batch['question'].to(device)
            answers = batch['answer'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):  # FIXME : Do we need to set this to false when evaluating validation ?
                outputs = model(inputs)     # FIXME : Input are image & question
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, answers)

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * dataloader.batch_size
            running_corrects += torch.sum(preds == answers.data)

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects.double() / dataset_size

        print('{} Loss: {:.4f} Acc: {:.4f}'.format('Train', epoch_loss, epoch_acc))


        # TODO : Save model
        # TODO : Save gamma & beta
        # TODO : Visualize gradcam
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model







import matplotlib.pyplot as plt

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = np.multiply(inp.numpy().transpose((1, 2, 0)), 255).astype(np.uint8)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


if __name__ == "__main__":
    image_config = {
            "type": "raw",
            "dim": [224, 224, 3]
        }

    test_dataset = CLEAR_dataset('data/v2.0.0_1k_scenes_1_inst_per_scene', image_config, 0, 'train', transforms=ToTensor())

    collate_fct = create_collate_fct(test_dataset.tokenizer)

    dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_fct)

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    train_model(device, None, dataloader)

    #batch = next(iter(dataloader))
    #grid = torchvision.utils.make_grid(batch['image'], padding=3)
    #imshow(grid)
    #plt.show()

    print("done")
