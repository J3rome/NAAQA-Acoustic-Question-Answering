import os
from datetime import datetime
import re

import pandas as pd

from utils.file import read_json


def get_experiments(data_path, prefix=None):
    experiments = []

    for exp_folder in os.listdir(data_path):
        exp_folder_path = f'{data_path}/{exp_folder}'

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

            # Parse experiment date
            date = datetime.strptime(date_folder, '%Y-%m-%d_%Hh%M')

            # Retrieve info from experiment name
            matches = re.match('(.*)_(\d+)k_(\d+)_inst_1024_win_50_overlap_(.*)_(\d+)_epoch_stop_at_(.*)_(\d+)', exp_folder)

            if not matches:
                continue

            matches = matches.groups()

            experiment = {
                'prefix': matches[0],
                'nb_scene': int(matches[1]) * 1000,
                'nb_q_per_scene': int(matches[2]),
                'config': matches[3],
                'nb_epoch': int(matches[4]),
                'stop_accuracy': float(matches[5]),
                'random_seed': matches[6],
                'date': date,

            }

            experiment['nb_sample'] = experiment['nb_scene'] * experiment['nb_q_per_scene']

            # Load experiment stats
            epoch_stats = read_json(f'{exp_dated_folder_path}/stats.json')

            experiment['nb_epoch_runned'] = len(epoch_stats)
            experiment['best_val_acc'] = float(epoch_stats[0]['val_acc'])
            experiment['best_val_loss'] = float(epoch_stats[0]['val_loss'])
            experiment['train_acc'] = float(epoch_stats[0]['train_acc'])
            experiment['train_loss'] = float(epoch_stats[0]['train_loss'])

            if experiment['nb_epoch_runned'] < experiment['nb_epoch']:
                # TODO : Check stopped_early.json
                if experiment['best_val_acc'] >= experiment['stop_accuracy']:
                    experiment['stopped_early'] = 'stop_threshold'
                else:
                    experiment['stopped_early'] = 'not_learning'
            else:
                experiment['stopped_early'] = 'NO'

            # Load number of params from model_summary
            experiment['total_nb_param'], experiment['nb_trainable_param'], experiment['nb_non_trainable_param'] = get_nb_param_from_summary(f'{exp_dated_folder_path}/model_summary.txt')

            # Load test set results
            test_result_filepath = f"{exp_dated_folder_path}/test_stats.json"
            if os.path.isfile(test_result_filepath):
                test_stats = read_json(f"{exp_dated_folder_path}/test_stats.json")
                experiment['test_version'] = test_stats['version_name']
                experiment['test_acc'] = test_stats['accuracy']
                experiment['test_loss'] = test_stats['loss']
            else:
                experiment['test_version'] = None
                experiment['test_acc'] = None
                experiment['test_loss'] = None

            # Load arguments
            arguments = read_json(f"{exp_dated_folder_path}/arguments.json")
            experiment['batch_size'] = arguments['batch_size']
            experiment['resnet_features'] = arguments['conv_feature_input']
            # TODO : Retrieve img_size, pad_to_largest. Those are only exposed when preparing/extracting features. We could write some json in output folder

            # Load timing

            # Load git-revision
            with open(f'{exp_dated_folder_path}/git.revision', 'r') as f:
                experiment['git_revision'] = f.readlines()[0].replace('\n' ,'')

            # Load config
            config = read_json(f'config/{experiment["config"]}.json')

            experiment['word_embedding_dim'] = config['question']['word_embedding_dim']
            experiment['rnn_state_size'] = config['question']['rnn_state_size']
            experiment['extractor_type'] = config['image_extractor']['type']
            experiment['stem_out_chan'] = config['stem']['conv_out']
            experiment['nb_resblock'] = len(config['resblock']['conv_out'])
            experiment['resblocks_out_chan'] = config['resblock']['conv_out'][-1]
            experiment['classifier_conv_out_chan'] = config['classifier']['conv_out']
            experiment['classifier_type'] = config['classifier']['type']
            experiment['classifier_global_pool'] = config['classifier']['global_pool_type']
            experiment['optimizer_type'] = config['optimizer']['type']
            experiment['optimizer_lr'] = config['optimizer']['learning_rate']
            experiment['optimizer_weight_decay'] = config['optimizer']['weight_decay']
            experiment['dropout_drop_prob'] = config['optimizer']['dropout_drop_prob']

            experiments.append(experiment)

    experiments_df = pd.DataFrame(experiments,
                                  columns=['prefix', 'nb_sample', 'nb_scene', 'nb_q_per_scene', 'config', 'nb_epoch',
                                           'nb_epoch_runned', 'stop_accuracy', 'best_val_acc', 'best_val_loss',
                                           'test_acc', 'test_loss', 'train_acc', 'train_loss', 'stopped_early',
                                           'batch_size', 'resnet_features', 'nb_trainable_param', 'test_version',
                                           'random_seed',  'date', 'total_nb_param', 'nb_non_trainable_param',
                                           'word_embedding_dim', 'rnn_state_size', 'extractor_type', 'stem_out_chan',
                                           'nb_resblock', 'resblocks_out_chan', 'classifier_conv_out_chan',
                                           'classifier_type', 'classifier_global_pool', 'optimizer_type',
                                           'optimizer_lr', 'optimizer_weight_decay', 'dropout_drop_prob', 'git_revision'
                                           ]
                                  )
    return experiments_df


def get_nb_param_from_summary(summary_filepath):
    with open(summary_filepath, 'r') as f:
        summary_lines = f.readlines()

        # Retrive lines containing 'params'. First is total params, second trainable params, third non-trainable
        nb_params = [int(l.split(':')[1].strip().replace(',', '')) for l in summary_lines if 'params' in l]

    return tuple(nb_params)