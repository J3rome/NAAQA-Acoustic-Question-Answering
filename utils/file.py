from collections import defaultdict
import os
import subprocess
import shutil

import h5py
import ujson


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


def create_folder_if_necessary(folder_path, overwrite_folder=False):
    is_symlink = os.path.islink(folder_path)
    if not os.path.isdir(folder_path) and not is_symlink:
        os.mkdir(folder_path)
    # FIXME : What if broken symlink and not overwriting folder ?
    elif overwrite_folder:
        if is_symlink and not os.path.exists(os.readlink(folder_path)):
            # Invalid symlink
            return  # FIXME : should we remove the broken symlink ?

        for file in os.listdir(folder_path):
            file_path = "%s/%s" % (folder_path, file)
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            elif os.path.isfile(file_path):
                os.remove(file_path)


def create_symlink_to_latest_folder(experiment_folder, dated_folder_name, symlink_name='latest'):
    symlink_path = "%s/%s" % (experiment_folder, symlink_name)
    # FIXME : Doesn't work as intended. Break when 'latest' link is broken
    if os.path.isdir(symlink_path) or (os.path.exists(symlink_path) and not os.path.exists(os.readlink(symlink_path))):
        # Remove the previous symlink before creating a new one (We readlink to recover in case of broken symlink)
        os.remove(symlink_path)

    subprocess.run('cd %s && ln -s %s %s' % (experiment_folder, dated_folder_name, symlink_name), shell=True)


def fix_best_epoch_symlink_if_necessary(output_dated_folder, film_model_weight_path):
    best_epoch_symlink_path = f"{output_dated_folder}/best"
    best_epoch_symlink_value = os.readlink(best_epoch_symlink_path)
    full_linked_path = f"{output_dated_folder}/{best_epoch_symlink_value}"
    if not os.path.exists(full_linked_path):
        # 'best' epoch symlink is broken. This will only happen if we --continue_training
        # and we can't beat the val loss of the last experiment
        previous_experiment_date = film_model_weight_path.split('/')[-3]
        subprocess.run(f"ln -snf ../{previous_experiment_date}/best {best_epoch_symlink_path}", shell=True)


def save_git_revision(output_folder, filename='git.revision'):
    output_path = '%s/%s' % (output_folder, filename)

    command = "git rev-parse HEAD > %s" % output_path
    command += " && git status | grep '\.py' >> %s" % output_path
    command += " && git diff '*.py' >> %s" % output_path

    subprocess.run(command, shell=True)


def create_folders_save_args(args, paths):
    create_folder_if_necessary(args['output_root_path'])
    create_folder_if_necessary(paths["output_task_folder"])
    create_folder_if_necessary(paths["output_experiment_folder"])
    create_folder_if_necessary(paths["output_dated_folder"])

    # Save arguments & config to output folder
    save_json(args, paths["output_dated_folder"], filename="arguments.json")
    save_git_revision(paths["output_dated_folder"])


def save_model_config(args, paths, model_config):
    save_json(model_config, paths["output_dated_folder"], filename='config_%s_input.json' % args['input_image_type'])

    # Copy dictionary file used
    shutil.copyfile(args['dict_file_path'], "%s/dict.json" % paths["output_dated_folder"])
