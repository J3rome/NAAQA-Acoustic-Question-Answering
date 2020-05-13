import os
from datetime import datetime, timedelta
import re

import pandas as pd
import numpy as np

from utils.file import read_json


def to_float(string):
    # Conversion with None handling
    return float(string) if string else None


def to_int(string):
    # Conversion with None handling
    return int(string) if string else None


def get_experiments(experiment_result_path, prefix=None):
    experiments = []

    for exp_folder in os.listdir(experiment_result_path):
        exp_folder_path = f'{experiment_result_path}/{exp_folder}'

        if not os.path.isdir(exp_folder_path):
            continue

        if prefix and prefix not in exp_folder:
            continue

        for date_folder in os.listdir(exp_folder_path):
            exp_dated_folder_path = f'{exp_folder_path}/{date_folder}'

            if date_folder == 'latest':
                # We skip the 'latest' symlink
                continue

            if 'best' not in os.listdir(exp_dated_folder_path):
                # Failed experiment. Was stopped before first epoch could be saved
                continue

            # Load arguments
            arguments = read_json(f"{exp_dated_folder_path}/arguments.json")

            # Retrieve Prefix, nb_scene and nb_question_per_scene from version name
            matches = re.match('(.*)_(\d+)k_(\d+)_inst_1024_win_50_overlap', arguments['version_name'])

            if not matches:
                continue

            matches = matches.groups()

            experiment = {
                'prefix': matches[0],
                'nb_scene': to_int(matches[1]) * 1000,
                'nb_q_per_scene': to_int(matches[2]),
                'config': arguments['config_path'].replace('config/','').replace('/','_').replace('.json', ''),
                'nb_epoch': arguments['nb_epoch'],
                'stop_accuracy': arguments['stop_at_val_acc'],
                'random_seed': arguments['random_seed'],
                'date': datetime.strptime(date_folder, '%Y-%m-%d_%Hh%M'),
                'folder': exp_folder
            }

            additional_note = arguments['output_name_suffix'].replace(f'_{experiment["nb_epoch"]}_epoch', '').replace(experiment['config'], '').replace(f'_{experiment["random_seed"]}', '').replace(f"_stop_at_{experiment['stop_accuracy']}", '').replace('_resnet', '')

            # Trim note
            if len(additional_note) > 0:
                if additional_note[0] == '_':
                    additional_note = additional_note[1:]

                if additional_note[-1] == '_':
                    additional_note = additional_note[:-1]
            else:
                additional_note = None

            experiment['note'] = additional_note

            experiment['nb_sample'] = experiment['nb_scene'] * experiment['nb_q_per_scene']

            # Load experiment stats
            epoch_stats = read_json(f'{exp_dated_folder_path}/stats.json')

            experiment['nb_epoch_runned'] = len(epoch_stats)
            experiment['nb_epoch_trained'] = to_int(epoch_stats[0]['epoch'].split('_')[-1])
            experiment['best_val_acc'] = to_float(epoch_stats[0]['val_acc'])
            experiment['best_val_loss'] = to_float(epoch_stats[0]['val_loss'])
            experiment['train_acc'] = to_float(epoch_stats[0]['train_acc'])
            experiment['train_loss'] = to_float(epoch_stats[0]['train_loss'])

            epoch_stats_chronological = sorted(epoch_stats, key=lambda x: to_int(x['epoch'].split('_')[1]))
            experiment['all_train_acc'] = []
            experiment['all_train_loss'] = []
            experiment['all_val_acc'] = []
            experiment['all_val_loss'] = []
            experiment['train_time'] = timedelta(0)
            epoch_times = []

            for stat in epoch_stats_chronological:
                experiment['all_train_acc'].append(to_float(stat['train_acc']))
                experiment['all_train_loss'].append(to_float(stat['train_loss']))
                experiment['all_val_acc'].append(to_float(stat['val_acc']))
                experiment['all_val_loss'].append(to_float(stat['val_loss']))

                parsed_time = datetime.strptime(stat['train_time'], "%H:%M:%S.%f")
                epoch_time = timedelta(hours=parsed_time.hour, minutes=parsed_time.minute, seconds=parsed_time.second,
                                       microseconds=parsed_time.microsecond)
                epoch_times.append(epoch_time)
                experiment['train_time'] += epoch_time

            experiment['mean_epoch_time'] = np.mean(epoch_times)

            # Load test set results
            test_result_filepath = f"{exp_dated_folder_path}/test_stats.json"
            if os.path.isfile(test_result_filepath):
                test_stats = read_json(f"{exp_dated_folder_path}/test_stats.json")
                experiment['test_version'] = test_stats['version_name']
                experiment['test_acc'] = to_float(test_stats['accuracy'])
                experiment['test_loss'] = to_float(test_stats['loss'])
            else:
                experiment['test_version'] = None
                experiment['test_acc'] = None
                experiment['test_loss'] = None

            experiment['0.6_at_epoch'] = None
            experiment['0.7_at_epoch'] = None
            experiment['0.8_at_epoch'] = None
            experiment['0.9_at_epoch'] = None

            epochs_stats_reversed = reversed(epoch_stats)
            for epoch_stat in epochs_stats_reversed:
                epoch_idx = to_int(epoch_stat['epoch'].split('_')[1])
                val_acc = to_float(epoch_stat['val_acc'])

                if experiment['0.6_at_epoch'] is None and val_acc >= 0.6:
                    experiment['0.6_at_epoch'] = epoch_idx
                elif experiment['0.7_at_epoch'] is None and val_acc >= 0.7:
                    experiment['0.7_at_epoch'] = epoch_idx
                elif experiment['0.8_at_epoch'] is None and val_acc >= 0.8:
                    experiment['0.8_at_epoch'] = epoch_idx
                elif experiment['0.9_at_epoch'] is None and val_acc >= 0.9:
                    experiment['0.9_at_epoch'] = epoch_idx

            if experiment['nb_epoch_runned'] < experiment['nb_epoch']:
                # TODO : Check stopped_early.json
                if experiment['stop_accuracy'] and experiment['best_val_acc'] >= experiment['stop_accuracy']:
                    experiment['stopped_early'] = 'stop_threshold'
                elif experiment['test_acc'] is None:
                    experiment['stopped_early'] = 'RUNNING'
                else:
                    experiment['stopped_early'] = 'not_learning'
            else:
                experiment['stopped_early'] = 'NO'

            # Load number of params from model_summary
            experiment['total_nb_param'], experiment['nb_trainable_param'], experiment['nb_non_trainable_param'] = get_nb_param_from_summary(f'{exp_dated_folder_path}/model_summary.txt')

            experiment['batch_size'] = arguments['batch_size']
            experiment['resnet_features'] = arguments['conv_feature_input']

            img_arguments = arguments

            if arguments['h5_image_input']:
                preprocessed_data_path = f"{arguments['data_root_path']}/{arguments['version_name']}/{arguments['preprocessed_folder_name']}"

                if os.path.exists(preprocessed_data_path):
                    img_arguments = read_json(preprocessed_data_path, 'arguments.json')

            experiment['pad_to_largest'] = img_arguments['pad_to_largest_image']
            experiment['resized_height'] = to_int(img_arguments['img_resize_height']) if img_arguments['resize_img'] else None
            experiment['resized_width'] = to_int(img_arguments['img_resize_width']) if img_arguments['resize_img'] else None

            # Load dict
            exp_dict = read_json(f'{exp_dated_folder_path}/dict.json')
            experiment['nb_answer'] = len(exp_dict['answer2i'])


            # Load timing

            # Load git-revision
            with open(f'{exp_dated_folder_path}/git.revision', 'r') as f:
                experiment['git_revision'] = f.readlines()[0].replace('\n', '')

            # Load config
            config_filepath = f'config/{experiment["config"]}.json'

            if not os.path.exists(config_filepath):
                # Config file doesn't exist on local instance, use the one in the exp_dated_folder
                config_filename = [f for f in os.listdir(exp_dated_folder_path) if 'config' in f and '.json' in f]

                if len(config_filename) == 0:
                    raise FileNotFoundError(f"No config file in '{exp_dated_folder_path}'")

                config_filepath = f"{exp_dated_folder_path}/{config_filename[0]}"

            config = read_json(config_filepath)

            experiment['word_embedding_dim'] = to_int(config['question']['word_embedding_dim'])
            experiment['rnn_state_size'] = to_int(config['question']['rnn_state_size'])
            experiment['extractor_type'] = config['image_extractor']['type']
            experiment['extractor_out_chan'] = to_int(config['image_extractor']['out'][-1])
            experiment['stem_out_chan'] = to_int(config['stem']['conv_out'])
            experiment['nb_resblock'] = len(config['resblock']['conv_out'])
            experiment['resblocks_out_chan'] = to_int(config['resblock']['conv_out'][-1])
            experiment['classifier_projection_out'] = to_int(config['classifier']['projection_size'])
            experiment['classifier_type'] = config['classifier']['type']
            experiment['classifier_global_pool'] = config['classifier']['global_pool_type']
            experiment['optimizer_type'] = config['optimizer']['type']
            experiment['optimizer_lr'] = to_float(config['optimizer']['learning_rate'])
            experiment['optimizer_weight_decay'] = to_float(config['optimizer']['weight_decay'])
            experiment['dropout_drop_prob'] = to_float(config['optimizer']['dropout_drop_prob'])

            if 'conv_out' not in config['classifier'] or experiment['classifier_type'] == 'conv':
                experiment['classifier_conv_out'] = None
            else:
                experiment['classifier_conv_out'] = to_int(config['classifier']['conv_out'])

            # Gpu name
            gpu_name_filepath = f"{exp_dated_folder_path}/gpu.json"
            if os.path.exists(gpu_name_filepath):
                experiment['gpu_name'] = read_json(gpu_name_filepath)['gpu_name']
            else:
                experiment['gpu_name'] = None

            experiments.append(experiment)

    experiments_df = pd.DataFrame(experiments,
                                  columns=['prefix', 'nb_sample', 'nb_scene', 'nb_q_per_scene', 'config', 'nb_epoch',
                                           'nb_epoch_runned', 'stop_accuracy', 'best_val_acc', 'best_val_loss',
                                           'test_acc', 'test_loss', 'train_acc', 'train_loss', 'stopped_early',
                                           '0.6_at_epoch', '0.7_at_epoch', '0.8_at_epoch', '0.9_at_epoch',
                                           'batch_size', 'resnet_features', 'nb_trainable_param', 'test_version',
                                           'random_seed', 'date', 'total_nb_param', 'nb_non_trainable_param',
                                           'word_embedding_dim', 'rnn_state_size', 'extractor_type', 'stem_out_chan',
                                           'nb_resblock', 'resblocks_out_chan', 'classifier_conv_out_chan',
                                           'classifier_type', 'classifier_global_pool', 'optimizer_type', 'nb_answer',
                                           'optimizer_lr', 'optimizer_weight_decay', 'dropout_drop_prob',
                                           'git_revision', 'pad_to_largest', 'resized_height', 'resized_width',
                                           'all_train_acc', 'all_train_loss', 'all_val_acc', 'all_val_loss',
                                           'train_time', 'mean_epoch_time', 'gpu_name', 'folder', 'note'
                                           ]
                                  )
    return experiments_df


def get_nb_param_from_summary(summary_filepath):
    with open(summary_filepath, 'r') as f:
        summary_lines = f.readlines()

        # Retrive lines containing 'params'. First is total params, second trainable params, third non-trainable
        nb_params = [int(l.split(':')[1].strip().replace(',', '')) for l in summary_lines if 'params' in l]

    return tuple(nb_params)
