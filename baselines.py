import random

from tqdm import tqdm

from utils.file import save_json
from runner import process_dataloader

def random_answer_baseline(dataloader, output_folder=None):
    answers = list(dataloader.dataset.answer_counter.keys())
    set_type = dataloader.dataset.set
    correct_answer = 0
    incorrect_answer = 0
    for batch in tqdm(dataloader):
        ground_truths = batch['answer'].tolist()
        for ground_truth in ground_truths:
            random_answer = random.choice(answers)
            if random_answer == ground_truth:
                correct_answer += 1
            else:
                incorrect_answer += 1

    accuracy = correct_answer / (correct_answer + incorrect_answer)

    print(f"Random answer baseline accuracy for set {set_type} : {accuracy}")

    if output_folder:
        save_json({'accuracy': accuracy}, output_folder, f'{set_type}_results.json')


def random_weight_baseline(model, device, dataloader, output_folder=None):

    set_type = dataloader.dataset.set
    loss, accuracy, _ = process_dataloader(False, device, model, dataloader)

    print(f"Random weight baseline for set {set_type}. Accuracy : {accuracy}  Loss : {loss}")

    if output_folder:
        save_json({'accuracy': accuracy}, output_folder, f'{set_type}_results.json')
