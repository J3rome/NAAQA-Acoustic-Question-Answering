import os
from datetime import datetime, timedelta
import re
from copy import deepcopy

import pandas as pd
import numpy as np

from utils.file import read_json, save_json
from utils.generic import get_answer_to_family_map


def to_float(string):
    # Conversion with None handling
    return float(string) if string or string == 0 else None


def to_int(string):
    # Conversion with None handling
    return int(string) if string or string == 0 else None


def get_max_freq(exp):
    if exp['input_type'] in ['audio', 'RGB', '1D']:
        sample_rate = exp['resample_audio'] if exp['resample_audio'] else 48000   # Acoustic scenes are sampled at 48kHz

        max_freq = sample_rate // 2
        if exp['keep_freq_point'] and exp['keep_freq_point'] < exp['n_fft']:
            max_freq = max_freq * (exp['keep_freq_point']/exp['n_fft'])

        return max_freq

    return None


def load_test_questions(data_folder):
    question_path = f"{data_folder}/questions/CLEAR_test_questions.json"
    if os.path.exists(question_path):
        return read_json(question_path)['questions']

    return None


def get_acc_per_q_type(exp_folder_path, exp, test_questions, answer_to_family_map, test_pred_path, force_recalc=False, cogent=False):
    per_q_acc_stats_path = f"{exp_folder_path}/test_stats_per_q.json"

    if "cogent" in test_pred_path:
        per_q_acc_stats_path = per_q_acc_stats_path.replace(".json", "_cogent.json")

    if os.path.exists(per_q_acc_stats_path) and not force_recalc:
        per_family_acc = read_json(per_q_acc_stats_path)

    else:
        print("Calculating per question accuracies")
        per_family_acc = {}

        test_preds = read_json(test_pred_path)       # FIXME: Will raise exception if file doesn't exist

        unique_filters = set()

        for pred in test_preds:
            # Assume questions are ordered by id
            question = test_questions[pred['question_id']]

            # for some reason, some 'ground_truth_answer_family' in loaded predictions are wrong... let's patch this...
            pred['ground_truth_answer_family'] = answer_to_family_map[str(question['answer']).lower()]

            last_program_node = question['program'][-1]['type']

            if last_program_node == "query_position_instrument":
                pred['ground_truth_answer_family'] = "position_rel"

            if pred['ground_truth_answer_family'] == 'count' and last_program_node == 'count_different_instrument':
                pred['ground_truth_answer_family'] = "count_diff"

            if pred['ground_truth_answer_family'] == 'boolean':
                if last_program_node in ['exist', 'or']:
                    pred['ground_truth_answer_family'] = "exist"
                elif last_program_node in ['equal_integer', 'greater_than', 'less_than']:
                    pred['ground_truth_answer_family'] = "count_compare"

            pred['relate_count'] = sum([1 for p in question['program'] if p['type'] == "relate"])
            pred['has_or'] = 'or' in question['question']

            filters = {p['type'] for p in question['program'] if 'filter' in p['type']}
            unique_filters.update(filters)
            pred['filters'] = ",".join(filters)


        # FIXME : THIS IS MOST PROBABLY NOT THE GOOD WAY OF DOING THIS... JUST NEED SOME QUICK DATA
        predictions_df = pd.DataFrame(test_preds,
                                      columns=['question_id', 'scene_id', 'scene_length', 'correct', 'prediction',
                                               'ground_truth', 'prediction_answer_family', 'ground_truth_answer_family',
                                               'confidence', 'relate_count', 'filters', 'has_or'])
        grouped = predictions_df.groupby(['ground_truth_answer_family', 'correct', 'relate_count', 'filters', 'has_or'],
                                         as_index=False)

        families = {k[0] for k in grouped.groups.keys()}

        grouped_count = grouped.count()

        # Global stats
        global_correct_entries = grouped_count[grouped_count['correct'] == True]
        global_incorrect_entries = grouped_count[grouped_count['correct'] == False]

        # No relate
        nb_global_correct_no_relate = global_correct_entries[global_correct_entries['relate_count'] == 0]['prediction'].sum()
        nb_global_incorrect_no_relate = global_incorrect_entries[global_incorrect_entries['relate_count'] == 0]['prediction'].sum()
        nb_global_total_no_relate = nb_global_correct_no_relate + nb_global_incorrect_no_relate

        per_family_acc[f'all_no_rel_test_acc'] = nb_global_correct_no_relate / nb_global_total_no_relate if nb_global_total_no_relate > 0 else -1

        # No relate with filter
        for f in unique_filters:
            nb_filter_correct_no_relate = global_correct_entries[(global_correct_entries['relate_count'] == 0) & (global_correct_entries['filters'].str.contains(f))]['prediction'].sum()
            nb_filter_incorrect_no_relate = global_incorrect_entries[(global_incorrect_entries['relate_count'] == 0) & (global_incorrect_entries['filters'].str.contains(f))]['prediction'].sum()
            nb_filter_total_no_relate = nb_filter_correct_no_relate + nb_filter_incorrect_no_relate

            per_family_acc[f'all_no_rel_with_{f}_test_acc'] = nb_filter_correct_no_relate / nb_filter_total_no_relate if nb_filter_total_no_relate > 0 else -1

        # With relate
        nb_global_correct_with_relate = global_correct_entries[global_correct_entries['relate_count'] > 0]['prediction'].sum()
        nb_global_incorrect_with_relate = global_incorrect_entries[global_incorrect_entries['relate_count'] > 0]['prediction'].sum()
        nb_global_total_with_relate = nb_global_correct_with_relate + nb_global_incorrect_with_relate

        per_family_acc[f'all_with_rel_test_acc'] = nb_global_correct_with_relate / nb_global_total_with_relate if nb_global_total_with_relate > 0 else -1

        # With OR
        nb_global_correct_with_or = global_correct_entries[global_correct_entries['has_or'] == True]['prediction'].sum()
        nb_global_incorrect_with_or = global_incorrect_entries[global_incorrect_entries['has_or'] == True]['prediction'].sum()
        nb_global_total_with_or = nb_global_correct_with_or + nb_global_incorrect_with_or

        per_family_acc[f'all_with_or_test_acc'] = nb_global_correct_with_or / nb_global_total_with_or if nb_global_total_with_or > 0 else -1

        # No OR --- FIXME : This global stat is not fair, we only have OR in 'compare','count' or 'exist' question
        nb_global_correct_no_or = global_correct_entries[global_correct_entries['has_or'] == False]['prediction'].sum()
        nb_global_incorrect_no_or = global_incorrect_entries[global_incorrect_entries['has_or'] == False]['prediction'].sum()
        nb_global_total_no_or = nb_global_correct_no_or + nb_global_incorrect_no_or

        per_family_acc[f'all_no_or_test_acc'] = nb_global_correct_no_or / nb_global_total_no_or if nb_global_total_no_or > 0 else -1

        # Per family stats
        for family in families:
            correct_entries = global_correct_entries[(global_correct_entries['ground_truth_answer_family'] == family)]
            incorrect_entries = global_incorrect_entries[(global_incorrect_entries['ground_truth_answer_family'] == family)]

            # Global acc by family
            nb_correct = correct_entries['prediction'].sum()
            nb_incorrect = incorrect_entries['prediction'].sum()
            nb_total = nb_correct + nb_incorrect

            per_family_acc[f'{family}_test_acc'] = nb_correct / nb_total if nb_total > 0 else -1

            # No relation acc by family
            nb_correct_no_relate = correct_entries[correct_entries['relate_count'] == 0]['prediction'].sum()
            nb_incorrect_no_relate = incorrect_entries[incorrect_entries['relate_count'] == 0]['prediction'].sum()
            nb_total_no_relate = nb_correct_no_relate + nb_incorrect_no_relate

            per_family_acc[f'{family}_no_rel_test_acc'] = nb_correct_no_relate / nb_total_no_relate if nb_total_no_relate > 0 else -1

            # No relation acc -- with filter by family
            for f in unique_filters:
                nb_filter_correct_no_relate = global_correct_entries[(global_correct_entries['relate_count'] == 0) & (global_correct_entries['filters'].str.contains(f))]['prediction'].sum()
                nb_filter_incorrect_no_relate = global_incorrect_entries[(global_incorrect_entries['relate_count'] == 0) & (global_incorrect_entries['filters'].str.contains(f))]['prediction'].sum()
                nb_filter_total_no_relate = nb_filter_correct_no_relate + nb_filter_incorrect_no_relate

                per_family_acc[f'{family}_no_rel_with_{f}_test_acc'] = nb_filter_correct_no_relate / nb_filter_total_no_relate if nb_filter_total_no_relate > 0 else -1

            # With relation acc by family
            nb_correct_with_relate = correct_entries[correct_entries['relate_count'] > 0]['prediction'].sum()
            nb_incorrect_with_relate = incorrect_entries[incorrect_entries['relate_count'] > 0]['prediction'].sum()
            nb_total_with_relate = nb_correct_with_relate + nb_incorrect_with_relate

            per_family_acc[f'{family}_with_rel_test_acc'] = nb_correct_with_relate / nb_total_with_relate if nb_total_with_relate > 0 else -1

            # With OR
            nb_correct_with_or = correct_entries[correct_entries['has_or'] == True]['prediction'].sum()
            nb_incorrect_with_or = incorrect_entries[incorrect_entries['has_or'] == True]['prediction'].sum()
            nb_total_with_or = nb_correct_with_or + nb_incorrect_with_or

            per_family_acc[f'{family}_with_or_test_acc'] = nb_correct_with_or / nb_total_with_or if nb_total_with_or > 0 else -1

            # No OR
            nb_correct_no_or = correct_entries[correct_entries['has_or'] == False]['prediction'].sum()
            nb_incorrect_no_or = incorrect_entries[incorrect_entries['has_or'] == False]['prediction'].sum()
            nb_total_no_or = nb_correct_no_or + nb_incorrect_no_or

            per_family_acc[f'{family}_no_or_test_acc'] = nb_correct_no_or / nb_total_no_or if nb_total_no_or > 0 else -1

        # Writing stats to file so we don't recalculate them
        save_json(per_family_acc, per_q_acc_stats_path, sort_keys=True)

    # Merge dicts together
    exp = {**exp, **per_family_acc}

    return exp


def get_experiments(experiment_result_path, exp_prefix=None, data_folder="data/CLEAR_50k_4_inst_audio",
                    cogent_data_folder="data/CLEAR_cogent_50k_4_inst_audio", question_type_analysis=True, min_date=None,
                    cogent_analysis=False):
    experiments = []
    if question_type_analysis:

        if not cogent_analysis:
            test_questions = load_test_questions(data_folder)
        else:
            test_questions = load_test_questions(cogent_data_folder)

        answer_to_family_map = get_answer_to_family_map(f'{data_folder}/attributes.json')

        if test_questions is None:
            # Deactivate question type analysis if we can't load questions
            question_type_analysis = False

    if min_date:
        if '_' in min_date:
            format_str = '%Y-%m-%d_%Hh%M'
        else:
            format_str = '%Y-%m-%d'

        min_date = datetime.strptime(min_date, format_str)

    for exp_folder in os.listdir(experiment_result_path):
        exp_folder_path = f'{experiment_result_path}/{exp_folder}'

        do_question_type_analysis = question_type_analysis

        if not os.path.isdir(exp_folder_path):
            continue

        if exp_prefix and exp_prefix not in exp_folder:
            continue

        for date_folder in os.listdir(exp_folder_path):
            exp_dated_folder_path = f'{exp_folder_path}/{date_folder}'

            if date_folder == 'latest':
                # We skip the 'latest' symlink
                continue

            # A random id might be appended to the experiment folder when uploaded to drive, remove it
            date_folder = re.sub(r'-\d+$', '', date_folder)
            experiment_date = datetime.strptime(date_folder, '%Y-%m-%d_%Hh%M')

            if min_date and experiment_date < min_date:
                # Skip experiments older than min_date, reduce loading time
                continue

            file_list = os.listdir(exp_dated_folder_path)
            if 'best' not in file_list or 'test_predictions.json' not in file_list:
                # Failed experiment. Was stopped before first epoch could be saved
                print(f"Failed experiment -- {exp_dated_folder_path}")
                print('==========================')
                continue

            # Load arguments
            arguments = read_json(f"{exp_dated_folder_path}/arguments.json")

            # Retrieve Prefix, nb_scene and nb_question_per_scene from version name
            if 'DAQA' not in arguments['version_name']:
                matches = re.match('(.*)_(\d+)k_(\d+)_(.*)', arguments['version_name'])

                if not matches:
                    continue

                matches = matches.groups()

            else:
                matches = ['DAQA', 0, 0]

                do_question_type_analysis = False

            experiment = {
                'prefix': matches[0],
                'nb_scene': to_int(matches[1]) * 1000,
                'nb_q_per_scene': to_int(matches[2]),
                'config': arguments['config_path'].replace('config/','').replace('/','_').replace('.json', ''),
                'nb_epoch': arguments['nb_epoch'],
                'stop_accuracy': arguments['stop_at_val_acc'],
                'random_seed': arguments['random_seed'],
                'date': experiment_date,
                'folder': exp_folder,
                'folder_dated': f"{exp_folder}/{date_folder}",
                "version_name": arguments['version_name'],
                "output_name_suffix": arguments['output_name_suffix']
            }

            additional_note = arguments['output_name_suffix'].replace(f'_{experiment["nb_epoch"]}_epoch', '').replace(experiment['config'], '').replace(f'_{experiment["random_seed"]}', '').replace(f"_stop_at_{experiment['stop_accuracy']}", '').replace('_resnet_extractor', '').replace('config_', '')

            # Trim note
            if len(additional_note) > 0:
                if additional_note[0] == '_':
                    additional_note = additional_note[1:]

                if additional_note[-1] == '_':
                    additional_note = additional_note[:-1]
            else:
                additional_note = None

            experiment['note'] = additional_note

            #experiment['queue'] = experiment['note'].split("__")[-1]
            experiment['queue'] = re.split(r'epoch_\d+_', arguments['output_name_suffix'])[-1]

            if 'device' in arguments:
                experiment['device'] = arguments['device']
            else:
                experiment['device'] = None

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
            experiment['0.6_at_epoch'] = None
            experiment['0.7_at_epoch'] = None
            experiment['0.8_at_epoch'] = None
            experiment['0.9_at_epoch'] = None

            for stat in epoch_stats_chronological:
                experiment['all_train_acc'].append(to_float(stat['train_acc']))
                experiment['all_train_loss'].append(to_float(stat['train_loss']))
                experiment['all_val_acc'].append(to_float(stat['val_acc']))
                experiment['all_val_loss'].append(to_float(stat['val_loss']))

                parsed_time = datetime.strptime(stat['train_time'].split('day, ')[-1], "%H:%M:%S.%f")
                epoch_time = timedelta(hours=parsed_time.hour, minutes=parsed_time.minute, seconds=parsed_time.second,
                                       microseconds=parsed_time.microsecond)
                epoch_times.append(epoch_time)
                experiment['train_time'] += epoch_time

                epoch_idx = to_int(stat['epoch'].split('_')[1])
                val_acc = to_float(stat['val_acc'])

                if experiment['0.6_at_epoch'] is None and val_acc >= 0.6:
                    experiment['0.6_at_epoch'] = epoch_idx
                elif experiment['0.7_at_epoch'] is None and val_acc >= 0.7:
                    experiment['0.7_at_epoch'] = epoch_idx
                elif experiment['0.8_at_epoch'] is None and val_acc >= 0.8:
                    experiment['0.8_at_epoch'] = epoch_idx
                elif experiment['0.9_at_epoch'] is None and val_acc >= 0.9:
                    experiment['0.9_at_epoch'] = epoch_idx

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
            experiment['preprocessed_folder_name'] = arguments['preprocessed_folder_name']

            experiment['reduce_lr_on_plateau'] = arguments.get('reduce_lr_on_plateau', False)

            img_arguments = arguments

            if arguments['h5_image_input']:
                # Copy preprocessed arguments if not in results
                local_preprocessed_arguments = f"{exp_dated_folder_path}/preprocessed_arguments.json"
                preprocessed_argument_path = f"{arguments['data_root_path']}/{arguments['version_name']}/{arguments['preprocessed_folder_name']}/arguments.json"

                if os.path.exists(local_preprocessed_arguments):
                    # Preprocessed arguments stored in the results folder
                    img_arguments = read_json(local_preprocessed_arguments)
                elif os.path.exists(preprocessed_argument_path):
                    # Preprocessed arguments stored in the data folder
                    img_arguments = read_json(preprocessed_argument_path)

                    save_json(img_arguments, local_preprocessed_arguments)
                #else:
                    #print(f"Was unable to retrieve preprocessing arguments for version {arguments['version_name']}")

                # Copy preprocessed data stats (mean, std, min, max) if not in results
                local_preprocesses_stats = f"{exp_dated_folder_path}/preprocessed_data_stats.json"
                preprocessed_stats_path = f"{arguments['data_root_path']}/{arguments['version_name']}/{arguments['preprocessed_folder_name']}/clear_stats.json"

                if not os.path.exists(local_preprocesses_stats) and os.path.exists(preprocessed_stats_path):
                    preprocessed_stats = read_json(preprocessed_stats_path)

                    save_json(preprocessed_stats, local_preprocesses_stats)
                #else:
                    #print(f"Was unable to retrieve dataset statistics for {arguments['version_name']} -- {arguments['preprocessed_folder_name']}")

            experiment['input_type'] = img_arguments['input_image_type']
            experiment['spectrogram_rgb'] = img_arguments['spectrogram_rgb'] if 'spectrogram_rgb' in img_arguments else None
            experiment['n_fft'] = img_arguments['spectrogram_n_fft'] if 'spectrogram_n_fft' in img_arguments else None
            experiment['hop_length'] = img_arguments['spectrogram_hop_length'] if 'spectrogram_hop_length' in img_arguments else None
            experiment['keep_freq_point'] = img_arguments['spectrogram_keep_freq_point'] if 'spectrogram_keep_freq_point' in img_arguments else None
            experiment['n_mels'] = img_arguments['spectrogram_n_mels'] if 'mel_spectrogram' in img_arguments and 'spectrogram_n_mels' in img_arguments and img_arguments['mel_spectrogram'] else None
            experiment['resample_audio'] = img_arguments['resample_audio_to'] if 'resample_audio_to' in img_arguments else None
            experiment['max_freq'] = get_max_freq(experiment)
            experiment['spectrogram_repeat_channels'] = img_arguments['spectrogram_repeat_channels'] if 'spectrogram_repeat_channels' in img_arguments else None

            if experiment['input_type'] == 'audio':
                experiment['input_type'] = 'RGB' if experiment['spectrogram_rgb'] else '1D'

            experiment['RGB_colormap'] = None
            if experiment['spectrogram_rgb']:
                if experiment['folder_dated'] in get_blues_experiment_id():
                    experiment['RGB_colormap'] = 'Blues'
                else:
                    experiment['RGB_colormap'] = 'Viridis'

            experiment['norm_zero_one'] = img_arguments['normalize_zero_one']
            experiment['norm_zero_one_again'] = arguments['normalize_zero_one']
            experiment['norm_clear_stats'] = img_arguments['normalize_with_clear_stats']
            experiment['norm_imagenet_stats'] = img_arguments['normalize_with_imagenet_stats']
            experiment['normalisation'] = 'data_stats' if img_arguments['normalize_with_clear_stats'] else None
            experiment['normalisation'] = 'imagenet_stats' if img_arguments['normalize_with_imagenet_stats'] else experiment['normalisation']

            experiment['pad_to_largest'] = img_arguments['pad_to_largest_image']
            experiment['resized_height'] = to_int(img_arguments['img_resize_height']) if img_arguments['resize_img'] else None
            experiment['resized_width'] = to_int(img_arguments['img_resize_width']) if img_arguments['resize_img'] else None


            # Cogent test set results
            # TODO : Load spec test result
            # TODO : Load mel test result
            # TODO : Define cogent_test_same_type (Choose spec or mel depended on what was the training type)
            # TODO : Add predictions
            n_mels = experiment['n_mels'] if experiment['n_mels'] is not None else "128"
            cogent_spec_test_stats_filepath = f"{exp_dated_folder_path}/cogent_spec_test_stats.json"
            cogent_mel_test_stats_filepath = f"{exp_dated_folder_path}/cogent_mel_{n_mels}_test_stats.json"
            cogent_spec_test_predictions_filepath = f"{exp_dated_folder_path}/cogent_spec_test_predictions.json"
            cogent_mel_test_predictions_filepath = f"{exp_dated_folder_path}/cogent_mel_{n_mels}_test_predictions.json"

            experiment['cogent_spec_test_acc'] = None
            experiment['cogent_spec_test_loss'] = None
            experiment['cogent_mel_test_acc'] = None
            experiment['cogent_mel_test_loss'] = None
            experiment['cogent_test_acc'] = None
            experiment['cogent_test_loss'] = None

            if experiment['prefix'] in ["CLEAR_FINAL", "DAQA"]:
                experiment['cogent_test_acc'] = experiment['test_acc']
                experiment['cogent_test_loss'] = experiment['test_loss']

            else:
                if os.path.isfile(cogent_spec_test_stats_filepath):
                    test_stats = read_json(cogent_spec_test_stats_filepath)
                    experiment['cogent_spec_test_acc'] = to_float(test_stats['accuracy'])
                    experiment['cogent_spec_test_loss'] = to_float(test_stats['loss'])
                else:
                    print(f"[MISSING COGENT] -- {exp_dated_folder_path}")

                if os.path.isfile(cogent_mel_test_stats_filepath):
                    test_stats = read_json(cogent_mel_test_stats_filepath)
                    experiment['cogent_mel_test_acc'] = to_float(test_stats['accuracy'])
                    experiment['cogent_mel_test_loss'] = to_float(test_stats['loss'])
                else:
                    print(f"[MISSING COGENT] -- {exp_dated_folder_path}")

                if img_arguments['mel_spectrogram']:
                    experiment['cogent_test_acc'] = experiment['cogent_mel_test_acc']
                    experiment['cogent_test_loss'] = experiment['cogent_mel_test_loss']
                    cogent_test_predictions_filepath = cogent_mel_test_predictions_filepath
                else:
                    experiment['cogent_test_acc'] = experiment['cogent_spec_test_acc']
                    experiment['cogent_test_loss'] = experiment['cogent_spec_test_loss']
                    cogent_test_predictions_filepath = cogent_spec_test_predictions_filepath

            # Load dict
            exp_dict = read_json(f'{exp_dated_folder_path}/dict.json')
            experiment['nb_answer'] = len(exp_dict['answer2i'])


            # Load timing

            # Load git-revision
            with open(f'{exp_dated_folder_path}/git.revision', 'r') as f:
                experiment['git_revision'] = f.readlines()[0].replace('\n', '')

            # Load config
            #config_filepath = f'config/{experiment["config"]}.json'
            config_filepath = arguments['config_path']

            if not os.path.exists(config_filepath):
                # Config file doesn't exist on local instance, use the one in the exp_dated_folder
                config_filename = [f for f in os.listdir(exp_dated_folder_path) if 'config' in f and '.json' in f]

                if len(config_filename) == 0:
                    raise FileNotFoundError(f"No config file in '{exp_dated_folder_path}'")

                config_filepath = f"{exp_dated_folder_path}/{config_filename[0]}"

            config = read_json(config_filepath)

            experiment['malimo'] = arguments['malimo'] if 'malimo' in arguments else None

            experiment['extractor_spatial_location'] = config['image_extractor'].get('spatial_location', False)
            experiment['stem_spatial_location'] = config['stem']['spatial_location']
            experiment['resblock_spatial_location'] = config['resblock']['spatial_location']
            experiment['classifier_spatial_location'] = config['classifier']['spatial_location']

            for prefix in ['extractor', 'stem', 'resblock', 'classifier']:
                key = f'{prefix}_spatial_location'
                if isinstance(experiment[key], bool):
                    experiment[key] = [0, 1] if experiment[key] else []

                nb_dim = len(experiment[key])
                if nb_dim == 0:
                    experiment[key] = 'None'
                elif nb_dim == 2:
                    experiment[key] = 'Both'
                else:
                    # nb_dim == 1
                    experiment[key] = 'Time' if experiment[key][0] == 1 else 'Freq'

            experiment['word_embedding_dim'] = to_int(config['question']['word_embedding_dim'])
            experiment['rnn_state_size'] = to_int(config['question']['rnn_state_size'])

            experiment['extractor_type'] = config['image_extractor']['type']

            if 'resnet_feature_extractor' in arguments and arguments['resnet_feature_extractor']:
                experiment['extractor_type'] = 'resnet'

            if experiment['extractor_type'] == 'film_original':
                experiment['extractor_type'] = 'Conv_2d'
            elif 'separated' in experiment['extractor_type']:
                experiment['extractor_type'] = 'Parallel'
            elif 'interlaced' in experiment['extractor_type']:
                experiment['extractor_type'] = 'Interleaved'

                if config['image_extractor']['time_first']:
                    experiment['extractor_type'] += "_Time_First"
                else:
                    experiment['extractor_type'] += "_Freq_First"

            else:
                experiment['extractor_type'] = experiment['extractor_type'].capitalize()

            experiment['extractor_out_chan'] = to_int(config['image_extractor']['out'][-1]) if type(config['image_extractor']['out']) == list else config['image_extractor']['out']
            experiment['extractor_filters'] = config['image_extractor']['out']

            if experiment['extractor_type'] in ['Baseline', 'Conv', 'Conv_2d']:
                experiment['extractor_nb_block'] = len(config['image_extractor']['kernels'])
                experiment['extractor_projection_size'] = None
            elif not 'Resnet' in experiment['extractor_type']:
                experiment['extractor_nb_block'] = len(config['image_extractor']['freq_kernels'])

                if len(config['image_extractor']['out']) > experiment['extractor_nb_block']:
                    experiment['extractor_projection_size'] = to_int(config['image_extractor']['out'][-1])
                    experiment['extractor_filters'] = experiment['extractor_filters'][:-1]
                else:
                    experiment['extractor_projection_size'] = None

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

            # Question type analysis
            if do_question_type_analysis:
                # FIXME : Should probably add the family columns with NaN ?
                if experiment['prefix'] == "CLEAR_FINAL" or not cogent_analysis or not os.path.exists(cogent_test_predictions_filepath):
                    test_pred_path = f"{exp_dated_folder_path}/test_predictions.json"
                else:
                    test_pred_path = cogent_test_predictions_filepath

                experiment = get_acc_per_q_type(exp_dated_folder_path, experiment, test_questions,
                                                answer_to_family_map, test_pred_path)

            experiments.append(experiment)

    cols = {k for e in experiments for k in e.keys()}

    experiments_df = pd.DataFrame(experiments, columns=cols)

    # Round number params to the closest thousand to facilitate comparison
    experiments_df['nb_trainable_param_round'] = experiments_df['nb_trainable_param'].apply(lambda x: x // 1000 * 1000)
    #experiments_df['nb_trainable_param_million'] = experiments_df['nb_trainable_param'].apply(lambda x: f"{x / 1000000:.2f} M" if x >= 1000000 else f"{x // 1000} k")
    experiments_df['nb_trainable_param_million'] = experiments_df['nb_trainable_param'].apply(lambda x: x / 1000000)

    #convert_dict = { key: 'int32' for key, val in experiments[0].items() if type(val) == int }

    #experiments_df.astype(convert_dict)

    return experiments_df


def get_format_dicts():
    question_families = [
        'count_diff',
        'exist',
        'count_compare',
        'boolean',
        'brightness',
        'count',
        'instrument',
        'loudness',
        'note',
        'position_global',
        'position_rel',
        'position'
    ]

    def percent_format(x):
        return "{:.2f}".format(x * 100) if not pd.isnull(x) and x > 0 else "NaN"

    def std_format(x):
        if type(x) != str:
            return f"± {x * 100:.2f}"

        values = x.split(' ± ')
        nb_values = len(values)

        if nb_values == 2:
            # For some reason (probably because of float representation) 90.05 doesn't get rounded to 90.1, adding a small value to it make it work.
            return f"{(float(values[0]) + 0.000000001)* 100:.1f} ± {float(values[1]) * 100:.2f}"
        elif nb_values == 1:
            return f"± {float(values[0]) * 100:.2f}"
        else:
            raise Exception("Format error")

    format_dict = {
        'total_nb_param': "{:,d}".format,
        'nb_non_trainable_param': "{:,d}".format,
        'nb_trainable_param': "{:,d}".format,
        'nb_trainable_param_round': "$\\sim${:,d}".format,
        "nb_trainable_param_million": "{:.2f} M".format,
        'nb_sample': "{:,d}".format,
        'nb_scene': "{:,d}".format,
        'rnn_state_size': "{:,d}".format,
        'optimizer_lr': "{:.2e}".format,
        'optimizer_weight_decay': "{:.2e}".format,
        'best_val_acc': percent_format,
        'best_val_acc_std': std_format,
        'train_acc': percent_format,
        'train_acc_std': std_format,
        'test_acc': percent_format,
        'test_acc_std': std_format,
        'cogent_test_acc': percent_format,
        'cogent_test_acc_std': std_format,
        'mean_std': std_format,
        'n_mels': '{:.0f}'.format,
        '0.6_at_epoch': lambda x: x if not pd.isnull(x) and x != 0 else None,
        '0.7_at_epoch': lambda x: x if not pd.isnull(x) and x != 0 else None,
        '0.8_at_epoch': lambda x: x if not pd.isnull(x) and x != 0 else None,
        '0.9_at_epoch': lambda x: x if not pd.isnull(x) and x != 0 else None
    }

    for family in question_families:
        format_dict[f"{family}_test_acc"] = percent_format
        format_dict[f"{family}_test_acc_std"] = std_format

        format_dict[f"{family}_no_rel_test_acc"] = percent_format
        format_dict[f"{family}_no_rel_test_acc_std"] = std_format

        format_dict[f"{family}_with_rel_test_acc"] = percent_format
        format_dict[f"{family}_with_rel_test_acc_std"] = std_format

    latex_format_dict = deepcopy(format_dict)
    latex_format_dict['nb_sample'] = lambda x: "{:d}k".format(x // 1000)
    latex_format_dict['nb_scene'] = lambda x: "{:d}k".format(x // 1000)
    latex_format_dict['best_val_acc'] = percent_format
    latex_format_dict['train_acc'] = percent_format
    latex_format_dict['test_acc'] = percent_format
    latex_format_dict['classifier_type'] = lambda x: 'Fcn' if x == 'fcn' else 'Conv-Avg'
    latex_format_dict['classifier_conv_out'] = lambda x: '{:d}'.format(int(x)) if not pd.isnull(x) and x > 0 else "--"
    latex_format_dict['classifier_projection_out'] = lambda x: '{:d}'.format(int(x)) if not pd.isnull(x) and x > 0 else "--"
    latex_format_dict['extractor_nb_block'] = lambda x: '{:d}'.format(int(x)) if not pd.isnull(x) else "--"
    latex_format_dict['extractor_projection_size'] = lambda x: '{:d}'.format(int(x)) if not pd.isnull(x) and x > 0 else "--"
    latex_format_dict['extractor_filters'] = lambda x: str(x)[1:-1]  # Remove the '[' and ']' of str(array)
    latex_format_dict['extractor_type'] = lambda x: x.replace("_", " ").capitalize()

    latex_format_dict['extractor_spatial_location'] = lambda x: x if x != "None" else "--"
    latex_format_dict['stem_spatial_location'] = lambda x: x if x != "None" else "--"
    latex_format_dict['resblock_spatial_location'] = lambda x: x if x != "None" else "--"
    latex_format_dict['classifier_spatial_location'] = lambda x: x if x != "None" else "--"

    def normalisation_format(norm_type):
        if norm_type and norm_type == "imagenet_stats":
            return "ImageNet"
        elif norm_type == "data_stats":
            return "CLEAR"
        else:
            return None
    latex_format_dict['normalisation'] = normalisation_format



    return format_dict, latex_format_dict


def get_nb_param_from_summary(summary_filepath):
    with open(summary_filepath, 'r') as f:
        summary_lines = f.readlines()

        # Retrive lines containing 'params'. First is total params, second trainable params, third non-trainable
        nb_params = [int(l.split(':')[1].strip().replace(',', '')) for l in summary_lines if 'params' in l]

    return tuple(nb_params)


def get_delete_experiment_from_drive_script(df, dryrun=False):
    cmds = []
    for f, d in df[['folder', 'date']].values:
        res_path = f"{f}/{d.strftime('%Y-%m-%d_%Hh%M')}"
        cmd = f"rclone delete Drive:result/training/{res_path} -P"
        if dryrun:
            cmd += " --dry-run"
        cmds.append(cmd)

    return cmds


def get_full_sync_experiment_from_drive_script(df, dest, dryrun=False):
    cmds = []
    for f, d in df[['folder', 'date']].values:
        res_path = f"{f}/{d.strftime('%Y-%m-%d_%Hh%M')}"
        cmd = f"rclone copy Drive:result/training/{res_path} {dest}/{res_path} -l -P"
        if dryrun:
            cmd += " --dry-run"
        cmds.append(cmd)

    return cmds


def get_exp_folder_path(df):
    paths = []
    for f, d in df[['folder', 'date']].values:
        paths.append(f"{f}/{d.strftime('%Y-%m-%d_%Hh%M')}")

    return paths


def get_blues_experiment_id():
    return [
        'CLEAR_50k_4_inst_audio_win_512_hop_2048_keep_256_RGB_norm_zero_one_norm_clear_stats_extractor_slim_parallel_3_block_64_proj_40_epoch_876944_original_test/2020-10-29_12h33',
        'CLEAR_50k_4_inst_audio_win_512_hop_2048_keep_256_RGB_norm_zero_one_norm_clear_stats_extractor_slim_interleaved_3_block_32_proj_40_epoch_876944_original_test/2020-10-29_21h32',
        'CLEAR_50k_4_inst_audio_win_512_hop_2048_keep_256_RGB_norm_zero_one_norm_clear_stats_extractor_slim_parallel_3_block_64_proj_no_pool_40_epoch_876944_no_pooling_no_pooling/2020-11-20_14h42',
        'CLEAR_50k_4_inst_audio_win_512_hop_2048_keep_256_RGB_norm_zero_one_norm_imagenet_stats_resnet_film_original_40_epoch_876944_original_resnet/2020-10-31_18h38',
        'CLEAR_50k_4_inst_audio_win_512_hop_2048_keep_256_RGB_norm_zero_one_norm_imagenet_stats_resnet_resize_to_224_square_film_original_40_epoch_876944_resized_224_resized_224/2020-11-11_16h20',
        'CLEAR_50k_4_inst_audio_win_512_hop_2048_keep_256_RGB_norm_zero_one_norm_clear_stats_film_original_40_epoch_876944_original_test/2020-10-28_22h22',
        'CLEAR_50k_4_inst_audio_win_512_hop_2048_keep_256_RGB_norm_zero_one_norm_clear_stats_resnet_film_original_40_epoch_876944_original_test/2020-10-31_01h38',
        'CLEAR_50k_4_inst_audio_win_512_hop_2048_keep_256_RGB_norm_zero_one_norm_clear_stats_resnet_film_original_40_epoch_876944_original_resnet/2020-10-31_13h58',
        'CLEAR_50k_4_inst_audio_win_512_hop_2048_keep_256_RGB_norm_zero_one_norm_imagenet_stats_resnet_film_original_40_epoch_876944_original_resnet/2020-11-01_00h09'
    ]

