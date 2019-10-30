import random
import os
from shutil import rmtree as rmdir_tree
import re
import numpy as np
import subprocess
import ujson
import h5py
from collections import defaultdict
from tqdm import tqdm
from dateutil.parser import parse as date_parse

import torch


def get_config(config_path):
    with open(config_path) as f:
        return ujson.load(f)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_random_state():
    states = {
        'py': random.getstate(),
        'np': np.random.get_state(),
        'torch': torch.random.get_rng_state()
    }

    if torch.cuda.is_available():
        states['torch_cuda'] = torch.cuda.get_rng_state()

    return states


def set_random_state(states):
    random.setstate(states['py'])
    np.random.set_state(states['np'])
    torch.random.set_rng_state(states['torch'])

    if torch.cuda.is_available() and 'torch_cuda' in states:
        torch.cuda.set_rng_state(states['torch_cuda'])


def create_folder_if_necessary(folder_path, overwrite_folder=False):
    is_symlink = os.path.islink(folder_path)
    if not os.path.isdir(folder_path) and not is_symlink:
        os.mkdir(folder_path)
    elif overwrite_folder:
        if is_symlink and not os.path.exists(os.readlink(folder_path)):
            # Invalid symlink
            return  # FIXME : should we remove the broken symlink ?

        for file in os.listdir(folder_path):
            file_path = "%s/%s" % (folder_path, file)
            if os.path.isdir(file_path):
                rmdir_tree(file_path)
            elif os.path.isfile(file_path):
                os.remove(file_path)


def create_symlink_to_latest_folder(experiment_folder, dated_folder_name, symlink_name='latest'):
    symlink_path = "%s/%s" % (experiment_folder, symlink_name)
    if os.path.isdir(symlink_path) or (os.path.exists(symlink_path) and not os.path.exists(os.readlink(symlink_path))):
        # Remove the previous symlink before creating a new one (We readlink to recover in case of broken symlink)
        os.remove(symlink_path)

    subprocess.run('cd %s && ln -s %s %s' % (experiment_folder, dated_folder_name, symlink_name), shell=True)


def process_predictions(dataset, predictions, ground_truths, questions_id, scenes_id, predictions_probs, images_padding):
    processed_predictions = []

    iterator = zip(predictions, ground_truths, questions_id, scenes_id, predictions_probs, images_padding)
    for prediction, ground_truth, question_id, scene_id, prediction_probs, image_padding in iterator:
        # TODO : Add is_relation
        scene_objects = dataset.scenes[scene_id]['definition']['objects']

        decoded_prediction = dataset.tokenizer.decode_answer(prediction)
        decoded_ground_truth = dataset.tokenizer.decode_answer(ground_truth)
        prediction_answer_family = dataset.answer_to_family[decoded_prediction]
        ground_truth_answer_family = dataset.answer_to_family[decoded_ground_truth]
        processed_predictions.append({
            'question_id': question_id,
            'scene_id': scene_id,
            'scene_length': len(scene_objects),
            'correct': bool(prediction == ground_truth),
            'correct_answer_family': bool(prediction_answer_family == ground_truth_answer_family),
            'prediction': decoded_prediction,
            'prediction_id': prediction,
            'ground_truth': decoded_ground_truth,
            'ground_truth_id': ground_truth,
            'confidence': prediction_probs[prediction],
            'prediction_answer_family': prediction_answer_family,
            'ground_truth_answer_family': ground_truth_answer_family,
            'prediction_probs': prediction_probs,
            'image_padding': image_padding
        })

    return processed_predictions


def process_gamma_beta(processed_predictions, gamma_vectors_per_resblock, beta_vectors_per_resblock):
    processed_gamma_beta_vectors = []

    if gamma_vectors_per_resblock[0].is_cuda:
        op_to_apply = lambda x: x.cpu().numpy()
    else:
        op_to_apply = lambda x: x.numpy()

    gamma_vectors_per_resblock = [op_to_apply(v) for v in gamma_vectors_per_resblock]
    beta_vectors_per_resblock = [op_to_apply(v) for v in beta_vectors_per_resblock]

    for result_index, processed_prediction in enumerate(processed_predictions):
        question_index = processed_prediction['question_id']
        processed_gamma_beta_vector = defaultdict(lambda : {})
        for resblock_index, gamma_vectors, beta_vectors in zip(range(len(gamma_vectors_per_resblock)), gamma_vectors_per_resblock, beta_vectors_per_resblock):

            processed_gamma_beta_vector['question_index'] = question_index
            processed_gamma_beta_vector['resblock_%d' % resblock_index]['gamma_vector'] = gamma_vectors[result_index]
            processed_gamma_beta_vector['resblock_%d' % resblock_index]['beta_vector'] = beta_vectors[result_index]

            # TODO : Add more attributes ? Could simply be cross loaded via question.json

        processed_gamma_beta_vectors.append(processed_gamma_beta_vector)

    return processed_gamma_beta_vectors


def sort_stats(stats, reverse=False):
    return sorted(stats, key=lambda e: float(e['val_loss']), reverse=reverse)


def save_training_stats(stats_output_file, epoch_nb, train_accuracy, train_loss, val_accuracy,
                        val_loss, epoch_train_time):
    """
    Will read the stats file from disk and append new epoch stats (Will create the file if not present)
    """
    if os.path.isfile(stats_output_file):
        with open(stats_output_file, 'r') as f:
            stats = ujson.load(f)
    else:
        stats = []

    stats.append({
        'epoch': "Epoch_%.2d" % epoch_nb,
        'train_time': str(epoch_train_time),
        'train_acc': '%.5f' % train_accuracy,
        'train_loss': '%.5f' % train_loss,
        'val_acc': '%.5f' % val_accuracy,
        'val_loss': '%.5f' % val_loss
    })

    stats = sort_stats(stats)

    with open(stats_output_file, 'w') as f:
        ujson.dump(stats, f, indent=2, sort_keys=True)

    return stats


def calc_mean_and_std(dataloader, device='cpu'):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    assert dataloader.dataset.is_raw_img(), "Config must be set to RAW img to calculate images stats"

    print("Calculating mean and std from dataset")

    cnt = 0
    fst_moment = torch.empty(3, device=device)
    snd_moment = torch.empty(3, device=device)

    for batched_data in tqdm(dataloader):
        images = batched_data['image'].to(device)
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    snd_moment = torch.sqrt(snd_moment - fst_moment ** 2)

    return fst_moment.tolist(), snd_moment.tolist()


def update_mean_in_config(mean, std, config_file_path, key='clear_stats', current_config=None):
    if current_config:
        config = current_config
    else:
        config = read_json(config_file_path)

    if 'preprocessing' not in config:
        config['preprocessing'] = {}

    if key not in config['preprocessing']:
        config['preprocessing'][key] = {}

    config['preprocessing'][key]['mean'] = mean
    config['preprocessing'][key]['std'] = std

    print(f"Saving mean ({mean}) and std ({std}) in '{config_file_path}'")

    save_json(config, config_file_path, indented=True)


def save_git_revision(output_folder, filename='git.revision'):
    output_path = '%s/%s' % (output_folder, filename)

    command = "git rev-parse HEAD > %s" % output_path
    command += " && git status | grep '\.py' >> %s" % output_path
    command += " && git diff '*.py' >> %s" % output_path

    subprocess.run(command, shell=True)


def save_json(results, output_folder, filename=None, indented=True):
    if filename is None:
        # First parameter is full path
        path = output_folder
    else:
        path = '%s/%s' % (output_folder, filename)

    with open(path, 'w') as f:
        ujson.dump(results, f, indent=2 if indented else None, escape_forward_slashes=False)


def read_json(folder, filename=None):
    if filename is None:
        # First parameter is full path
        path = folder
    else:
        path = '%s/%s' % (folder, filename)

    with open(path, 'r') as f:
        return ujson.load(f)


def get_size_from_image_header(folder, filename=None):
    if filename is None:
        # First parameter is full path
        path = folder
    else:
        path = '%s/%s' % (folder, filename)

    with open(path, 'rb') as f:
        image_header = f.read(25)

    assert b'PNG' in image_header[:8], 'Image must be a PNG'

    width = int.from_bytes(image_header[16:20], byteorder='big')
    height = int.from_bytes(image_header[20:24], byteorder='big')

    return height, width


def read_gamma_beta_h5(filepath):
    gammas_betas = []

    with h5py.File(filepath, 'r') as f:
        nb_val = f['question_index'].shape[0]
        set_type = f['question_index'].attrs['set_type']

        for idx in range(nb_val):
            gamma_beta = {
                'question_index': f['question_index'][idx]
            }

            for resblock_key in f['gamma']:
                gamma_beta[resblock_key] = {
                    'gamma_vector': f['gamma'][resblock_key][idx],
                    'beta_vector': f['beta'][resblock_key][idx]
                }

            gammas_betas.append(gamma_beta)

    return set_type, gammas_betas


def save_gamma_beta_h5(gammas_betas, set_type, folder, filename=None, nb_vals=None, start_idx=0):
    """
    This is a PATCH, couldn't write huge JSON files.
    The data structure could be better, just a quick hack to make it work without changing the structure
    """

    if nb_vals is None:
        nb_vals = len(gammas_betas)

    if filename is None:
        # First parameter is full path
        path = folder
    else:
        path = '%s/%s' % (folder, filename)

    resblock_keys = list(set(gammas_betas[0].keys()) - {'question_index'})
    nb_dim_resblock = len(gammas_betas[0]['resblock_0']['gamma_vector'])

    file_exist = os.path.isfile(path)

    with h5py.File(path, 'a', libver='latest') as f:

        if not file_exist:
            # Create datasets
            f.create_dataset('question_index', (nb_vals,), dtype='i')

            f['question_index'].attrs['set_type'] = set_type

            for group_name in ['gamma', 'beta']:
                group = f.create_group(group_name)

                for resblock_key in resblock_keys:
                    group.create_dataset(resblock_key, (nb_vals, nb_dim_resblock), dtype='f')

            start_idx = 0

        nb_val_to_write = len(gammas_betas)
        vals = defaultdict(lambda : defaultdict(lambda : []))
        vals['question_index'] = []

        # Extract all values so we can write them all at once
        for gamma_beta in gammas_betas:
            vals['question_index'].append(gamma_beta['question_index'])

            for resblock_key in resblock_keys:
                vals['gamma'][resblock_key].append(gamma_beta[resblock_key]['gamma_vector'])
                vals['beta'][resblock_key].append(gamma_beta[resblock_key]['beta_vector'])

        # Write data to H5 file
        f['question_index'][start_idx:nb_val_to_write] = vals['question_index']

        for resblock_key in resblock_keys:
            f['gamma'][resblock_key][start_idx:nb_val_to_write,:] = vals['gamma'][resblock_key]
            f['beta'][resblock_key][start_idx:nb_val_to_write,:] = vals['beta'][resblock_key]

        return nb_val_to_write


def get_answer_to_family_map(attributes_filepath, to_lowercase=True, reduced_text=False):
    attributes = read_json(attributes_filepath)

    answer_to_family = {"<unk>": "unknown"}  # FIXME : Quantify what is the impact of having an unknown answer

    for family, answers in attributes.items():
        for answer in answers:
            the_answer = answer
            if reduced_text and 'of the scene' in the_answer:
                the_answer = str(answer.split(' ')[0][:3])

            if to_lowercase:
                the_answer = the_answer.lower()

            # If there is duplicated answers, they will be assigned to the first occurring family
            if the_answer not in answer_to_family:
                answer_to_family[the_answer] = family

    return answer_to_family


def is_date_string(string):
    return re.match(r'\d+-\d+-\d+_\d+h\d+', string) != None


def close_tensorboard_writers(tensorboard_writers):
    for key, writer in tensorboard_writers.items():
        writer.close()


def visualize_cam(mask, img):
    """ Taken from https://github.com/vickyliin/gradcam_plus_plus-pytorch
    Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]

    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    heatmap = (255 * mask.squeeze()).type(torch.uint8).cpu().numpy()
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])

    result = heatmap+img.cpu()
    result = result.div(result.max()).squeeze()

    return heatmap, result
