import argparse
from _datetime import datetime
import subprocess
import shutil
import os

from tqdm import tqdm

from utils import set_random_seed, create_folder_if_necessary, get_config, process_predictions, process_gamma_beta
from utils import create_symlink_to_latest_folder, save_training_stats, save_json, sort_stats, is_date_string
from utils import calc_mean_and_std, save_gamma_beta_h5, save_git_revision, get_random_state, set_random_state

from visualization import visualize_gamma_beta, grad_cam_visualization
from preprocessing import create_dict_from_questions, extract_features

# NEW IMPORTS
from models.CLEAR_film_model import CLEAR_FiLM_model
from data_interfaces.CLEAR_dataset import CLEAR_dataset, CLEAR_collate_fct
from data_interfaces.transforms import ToTensor, ImgBetweenZeroOne, ResizeImgBasedOnHeight, ResizeImgBasedOnWidth, PadTensor, NormalizeSample, ResizeTensor
from models.torchsummary import summary     # Custom version of torchsummary to fix bugs with input
import torch
import time
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
from tensorboardX import SummaryWriter


# TODO : Add option for custom test file --> Already available by specifying different inference_set ? The according dataset & dataloader should be created..
#       Maybe not a good idea to instantiate everything out of the "task" functions.. Or maybe we could just instantiate it inside for the test inference
parser = argparse.ArgumentParser('FiLM model for CLEAR Dataset (Acoustic Question Answering)', fromfile_prefix_chars='@')

parser.add_argument("--training", help="FiLM model training", action='store_true')
parser.add_argument("--inference", help="FiLM model inference", action='store_true')
parser.add_argument("--visualize_grad_cam", help="Class Activation Maps - GradCAM", action='store_true')
parser.add_argument("--visualize_gamma_beta", help="FiLM model parameters visualization (T-SNE)", action='store_true')
parser.add_argument("--feature_extract", help="Feature Pre-Extraction", action='store_true')
parser.add_argument("--create_dict", help="Create word dictionary (for tokenization)", action='store_true')

# Input parameters
parser.add_argument("--data_root_path", type=str, default='data', help="Directory with data")
parser.add_argument("--version_name", type=str, help="Name of the dataset version")
parser.add_argument("--film_model_weight_path", type=str, default=None, help="Path to Film pretrained weight file")
parser.add_argument("--config_path", type=str, default='config/film.json', help="Path to Film pretrained ckpt file")
parser.add_argument("--inference_set", type=str, default='test', help="Define on which set the inference will run")
parser.add_argument("--dict_file_path", type=str, default=None, help="Define what dictionnary file should be used")
parser.add_argument("--normalize_with_imagenet_stats", help="Will normalize input images according to"
                                                       "ImageNet mean & std (Only with RAW input)", action='store_true')
parser.add_argument("--normalize_with_clear_stats", help="Will normalize input images according to"
                                                         "CLEAR mean & std (Only with RAW input)", action='store_true')
parser.add_argument("--no_img_resize", help="Disable RAW image resizing", action='store_true')
parser.add_argument("--raw_img_resize_val", type=int, default=224,
                    help="Specify the size to which the image will be resized (when working with RAW img)"
                         "The width is calculated according to the height in order to keep the ratio")
parser.add_argument("--raw_img_resize_based_on_height", action='store_true',
                    help="If set (with --raw_img_resize_val), the width of the image will be calculated according to "
                         "the height in order to keep the ratio. [Default option if neither "
                         "--raw_img_resize_based_on_height and --raw_img_resize_based_on_width are set]")
parser.add_argument("--raw_img_resize_based_on_width", action='store_true',
                    help="If set (with --raw_img_resize_val), the height of the image will be calculated according to "
                         "the width in order to keep the ratio")


parser.add_argument("--keep_image_range", help="Will NOT scale the image between 0-1 (RAW img)", action='store_true')
parser.add_argument("--pad_to_largest_image", help="If set, images will be padded to meet the largest image in the set."
                                                   "All input will have the same size.", action='store_true')
parser.add_argument("--pad_to_square_images", help="If set, all images will be padded to make them square",
                    action='store_true')
parser.add_argument("--resize_to_square_images", help="If set, all images will be resized to make them square",
                    action='store_true')
parser.add_argument("--gamma_beta_path", type=str, default=None, help="Path where gamma_beta values are stored "
                                                                          "(when using --visualize_gamma_beta)")
parser.add_argument("--no_early_stopping", help="Override the early stopping config", action='store_true')


# Output parameters
parser.add_argument("--output_root_path", type=str, default='output', help="Directory with image")
parser.add_argument("--preprocessed_folder_name", type=str, default='preprocessed',
                    help="Directory where to store/are stored extracted features and token dictionary")
parser.add_argument("--output_name_suffix", type=str, default='', help="Suffix that will be appended to the version "
                                                                       "name (output & tensorboard)")
parser.add_argument("--dict_folder", type=str, default=None,
                    help="Directory where to store/retrieve generated dictionary. "
                         "If --dict_file_path is used, this will be ignored")
parser.add_argument("--tensorboard_folder", type=str, default='tensorboard',
                    help="Path where tensorboard data should be stored.")
parser.add_argument("--tensorboard_save_graph", help="Save model graph to tensorboard", action='store_true')
parser.add_argument("--perf_over_determinist", help="Will let torch use nondeterministic algorithms (Better "
                                                    "performance but less reproductibility)", action='store_true')


# Other parameters
parser.add_argument("--nb_epoch", type=int, default=15, help="Nb of epoch for training")
parser.add_argument("--nb_epoch_stats_to_keep", type=int, default=5, help="Nb of epoch stats to keep for training")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size (For training and inference)")
parser.add_argument("--continue_training", help="Will use the --film_model_weight_path as  training starting point",
                    action='store_true')
parser.add_argument("--random_seed", type=int, default=None, help="Random seed used for the experiment")
parser.add_argument("--use_cpu", help="Model will be run/train on CPU", action='store_true')
parser.add_argument("--force_dict_all_answer", help="Will make sure that all answers are included in the dict" +
                                                    "(not just the one appearing in the train set)" +
                                                    " -- Preprocessing option" , action='store_true')
parser.add_argument("--no_model_summary", help="Will hide the model summary", action='store_true')


# TODO : Interactive mode
def run_one_game(device, model, games, data_path, input_config, transforms_list=[]):

    # TODO : Should be able to run from a specific image (Not necessarly inside data/experiment_name/images/set_type)

    set_type = 'test'
    game = {
            "question_index" : 1,
            "question": "How many loud things can we hear ?",
            "answer": 3,
            "scene_index": 3
        }

    game['scene_filename'] = "CLEAR_%s_%06d.png" % (set_type, game['scene_index'])

    games = [
        game
    ]

    test_dataset = CLEAR_dataset(data_path, input_config, set_type, questions=games,
                                 dict_file_path=args.dict_file_path,
                                 transforms=transforms.Compose(transforms_list + [ToTensor()]))

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                  num_workers=1, collate_fn=test_dataset.CLEAR_collate_fct)

    _, accuracy, predictions, gamma_beta = process_dataloader(False, device, model, test_dataloader)

    print("Accuracy is : %f" % accuracy)


def set_inference(device, model, dataloader, output_folder, save_gamma_beta=True):
    set_type = dataloader.dataset.set

    if save_gamma_beta:
        gamma_beta_path = '%s/%s_gamma_beta.h5' % (output_folder, set_type)
    else:
        gamma_beta_path = None

    _, acc, predictions = process_dataloader(False, device, model, dataloader, gamma_beta_path=gamma_beta_path)

    save_json(predictions, output_folder, filename='%s_predictions.json' % set_type)

    print("%s Accuracy : %.5f" % (set_type, acc))


def train_model(device, model, dataloaders, output_folder, criterion=None, optimizer=None, scheduler=None,
                nb_epoch=25, nb_epoch_to_keep=None, start_epoch=0, tensorboard_writers=None):

    if tensorboard_writers is None:
        tensorboard_writers = {'train': None, 'val': None}
    else:
        assert 'train' in tensorboard_writers and 'val' in tensorboard_writers, 'Must provide all tensorboard writers.'

    stats_file_path = "%s/stats.json" % output_folder
    removed_epoch = []

    since = time.time()

    # Early stopping (Only enable when we are running at least 20 epoch)
    early_stopping = model.early_stopping is not None and nb_epoch > 20
    if early_stopping:
        if type(model.early_stopping['wait_first_n_epoch']) == float:
            wait_first_n_epoch = int(nb_epoch * model.early_stopping['wait_first_n_epoch'])
        else:
            wait_first_n_epoch = model.early_stopping['wait_first_n_epoch']

        if type(model.early_stopping['stop_threshold']) == float:
            stop_threshold = int(nb_epoch*model.early_stopping['stop_threshold'])
        else:
            stop_threshold = model.early_stopping['stop_threshold']

        wait_first_n_epoch += start_epoch           # Apply grace period even when continuing training
        stop_threshold = max(stop_threshold, 1)
        best_val_loss = 9999
        early_stop_counter = 0

    # TODO : Write hyperparams to tensorboard

    for epoch in range(start_epoch, start_epoch + nb_epoch):
        epoch_output_folder_path = "%s/Epoch_%.2d" % (output_folder, epoch)
        create_folder_if_necessary(epoch_output_folder_path)
        print('Epoch {}/{}'.format(epoch, start_epoch + nb_epoch - 1))
        print('-' * 10)

        epoch_time = datetime.now()
        train_loss, train_acc, train_predictions = process_dataloader(True, device, model,
                                                                      dataloaders['train'],
                                                                      criterion, optimizer, epoch_id=epoch,
                                                                      tensorboard_writer=tensorboard_writers['train'],
                                                                      gamma_beta_path="%s/train_gamma_beta.h5" % epoch_output_folder_path)
        epoch_train_time = datetime.now() - epoch_time

        print('\n{} Loss: {:.4f} Acc: {:.4f}'.format('Train', train_loss, train_acc))

        val_loss, val_acc, val_predictions = process_dataloader(False, device, model,
                                                                dataloaders['val'], criterion, epoch_id=epoch,
                                                                tensorboard_writer=tensorboard_writers['val'],
                                                                gamma_beta_path="%s/val_gamma_beta.h5" % epoch_output_folder_path)
        print('\n{} Loss: {:.4f} Acc: {:.4f}'.format('Val', val_loss, val_acc))

        stats = save_training_stats(stats_file_path, epoch, train_acc, train_loss, val_acc, val_loss, epoch_train_time)

        save_json(train_predictions, epoch_output_folder_path, filename="train_predictions.json")
        save_json(val_predictions, epoch_output_folder_path, filename="val_predictions.json")

        # Save training weights
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.get_cleaned_state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            'rng_state': get_random_state(),
        }

        torch.save(checkpoint, '%s/model.pt.tar' % epoch_output_folder_path)

        sorted_stats = sort_stats(stats)

        if nb_epoch_to_keep is not None:
            # FIXME : Probably not the most efficient way to do this
            epoch_to_remove = sorted_stats[nb_epoch_to_keep:]

            for epoch_stat in epoch_to_remove:
                if epoch_stat['epoch'] not in removed_epoch:
                    removed_epoch.append(epoch_stat['epoch'])

                    shutil.rmtree("%s/%s" % (output_folder, epoch_stat['epoch']))

        print("Epoch took %s" % str(datetime.now() - epoch_time))

        # Create a symlink to best epoch output folder
        best_epoch = sorted_stats[0]
        print("Best Epoch is %s" % best_epoch['epoch'])
        best_epoch_symlink_path = '%s/best' % output_folder
        subprocess.run("ln -snf %s %s" % (best_epoch['epoch'], best_epoch_symlink_path), shell=True)

        # Early Stopping
        if early_stopping:
            if val_loss < best_val_loss - model.early_stopping['min_step']:
                best_val_loss = val_loss
                early_stop_counter = 0
            elif epoch > wait_first_n_epoch:
                early_stop_counter += 1
                print("Early Stopping count : %d/%d" % (early_stop_counter, stop_threshold))

                if early_stop_counter >= stop_threshold:
                    print("Early Stopping at epoch %d on %d" % (epoch, nb_epoch))
                    break

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {}'.format(best_epoch['val_acc']))

    # TODO : load best model weights ?
    #model.load_state_dict(best_model_state)
    return model


def process_dataloader(is_training, device, model, dataloader, criterion=None, optimizer=None, gamma_beta_path=None,
                       write_to_file_every=500, epoch_id=0, tensorboard_writer=None):
    # Model should already have been copied to the GPU at this point (If using GPU)
    assert (is_training and criterion is not None and optimizer is not None) or not is_training

    if is_training:
        model.train()
    else:
        model.eval()

    dataset_size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    running_loss = 0.0
    running_corrects = 0

    processed_predictions = []
    processed_gammas_betas = []
    all_questions = []
    nb_written = 0

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        #mem_trace.report('Batch %d/%d - Epoch %d' % (i, dataloader.batch_size, epoch))
        images = batch['image'].to(device)
        questions = batch['question'].to(device)
        answers = batch['answer'].to(device)
        seq_lengths = batch['seq_length'].to(device)

        # Those are not processed by the network, only used to create statistics. Therefore, no need to copy to GPU
        questions_id = batch['id']
        scenes_id = batch['scene_id']
        images_padding = batch['image_padding']

        if is_training:
            # zero the parameter gradients
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            outputs, outputs_softmax = model(questions, seq_lengths, images, pack_sequence=True)
            _, preds = torch.max(outputs, 1)
            if criterion:
                loss = criterion(outputs, answers)

            if is_training:
                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

        batch_processed_predictions = process_predictions(dataloader.dataset, preds.tolist(), answers.tolist(),
                                                          questions_id.tolist(), scenes_id.tolist(),
                                                          outputs_softmax.tolist(), images_padding.tolist())

        processed_predictions += batch_processed_predictions

        # TODO : Add config to log only specific things
        if tensorboard_writer:
            # FIXME: Find a way to show original input images in tensorboard (Could save a list of scene ids and add them to tensorboard after the epoch, check performance cost -- Image loading etc)
            if dataloader.dataset.is_raw_img():
                # TODO : Tag img before adding to tensorboard ? -- This can be done via .add_image_with_boxes()
                tensorboard_writer.add_images('Inputs/images', batch['image'], epoch_id)

            all_questions += batch['question'].tolist()

        if gamma_beta_path is not None:
            gammas, betas = model.get_gammas_betas()
            processed_gammas_betas += process_gamma_beta(batch_processed_predictions, gammas, betas)

            if batch_idx % write_to_file_every == 0 and batch_idx != 0:
                nb_written += save_gamma_beta_h5(processed_gammas_betas, dataloader.dataset.set, gamma_beta_path,
                                                 nb_vals=dataset_size, start_idx=nb_written)
                processed_gammas_betas = []

        # statistics
        if criterion:
            running_loss += loss.item() * dataloader.batch_size
        running_corrects += torch.sum(preds == answers.data).item()

    nb_left_to_write = len(processed_gammas_betas)
    if gamma_beta_path is not None and nb_left_to_write > 0:
        save_gamma_beta_h5(processed_gammas_betas, dataloader.dataset.set, gamma_beta_path, nb_vals=dataset_size,
                           start_idx=nb_written)

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects / dataset_size

    # TODO : Add config to log only specific things
    if tensorboard_writer:
        log_text = ""
        for question, processed_prediction in zip(all_questions, processed_predictions):
            # FIXME: Tokenizer might not be instantiated --- We probably wouldn't be logging in tensorboard..
            decoded_question = dataloader.dataset.tokenizer.decode_question(question, remove_padding=True)
            log_text += f"{processed_prediction['correct']}//{processed_prediction['correct_answer_family']} "
            log_text += f"{decoded_question} -- {processed_prediction['ground_truth']} "
            log_text += f"[[{processed_prediction['prediction']} - {processed_prediction['confidence']}]]  \n"

        tensorboard_writer.add_text('Inputs/Text', log_text, epoch_id)

        tensorboard_writer.add_scalar('Results/Loss', epoch_loss, global_step=epoch_id)
        tensorboard_writer.add_scalar('Results/Accuracy', epoch_acc, global_step=epoch_id)

    return epoch_loss, epoch_acc, processed_predictions


def validate_arguments(args):
    
    mutually_exclusive_params = [args.training, args.inference, args.feature_extract,
                                 args.create_dict, args.visualize_gamma_beta, args.visualize_grad_cam]

    assert sum(mutually_exclusive_params) == 1, \
        "[ERROR] Can only do one task at a time " \
        "(--training, --inference, --visualize_gamma_beta, --create_dict, --feature_extract)"

    mutually_exclusive_params = [args.raw_img_resize_based_on_height, args.raw_img_resize_based_on_width]
    assert sum(mutually_exclusive_params) < 2, "[ERROR] Image resize can be either --raw_img_resize_based_on_height " \
                                               "or --raw_img_resize_based_on_width but not both"

    mutually_exclusive_params = [args.pad_to_square_images, args.resize_to_square_images]
    assert sum(mutually_exclusive_params) < 2, "[ERROR] Can either --pad_to_square_images or --resize_to_square_images"


def get_task_from_args(args):
    if args.training:
        return "train_film"
    elif args.inference:
        return "inference"
    elif args.visualize_gamma_beta:
        return "visualize_gamma_beta"
    elif args.visualize_grad_cam:
        return "visualize_grad_cam"
    elif args.feature_extract:
        return "feature_extract"
    elif args.create_dict:
        return "create_dict"

    assert False, "Arguments don't specify task"

def main(args):
    # Parameter validation
    validate_arguments(args)
    task = get_task_from_args(args)

    output_name = args.version_name + "_" + args.output_name_suffix if args.output_name_suffix else args.version_name
    print("Task '%s' for version '%s'\n" % (task.replace('_', ' ').title(), output_name))

    if args.random_seed is not None:
        set_random_seed(args.random_seed)

    # Paths
    data_path = "%s/%s" % (args.data_root_path, args.version_name)

    output_task_folder = "%s/%s" % (args.output_root_path, task)
    output_experiment_folder = "%s/%s" % (output_task_folder, output_name)
    current_datetime = datetime.now()
    current_datetime_str = current_datetime.strftime("%Y-%m-%d_%Hh%M")
    output_dated_folder = "%s/%s" % (output_experiment_folder, current_datetime_str)

    # Flags
    restore_model_weights = args.inference or (args.training and args.continue_training) or args.visualize_grad_cam
    create_output_folder = not args.create_dict and not args.feature_extract
    instantiate_model = not args.create_dict and 'gamma_beta' not in task
    use_tensorboard = 'train' in task

    args.dict_folder = args.preprocessed_folder_name if args.dict_folder is None else args.dict_folder
    if args.dict_file_path is None:
        args.dict_file_path = "%s/%s/dict.json" % (data_path, args.dict_folder)

    film_model_config = get_config(args.config_path)

    early_stopping = not args.no_early_stopping and film_model_config['early_stopping']['enable']
    film_model_config['early_stopping']['enable'] =  early_stopping

    if create_output_folder:
        # TODO : See if this is optimal file structure
        create_folder_if_necessary(args.output_root_path)
        create_folder_if_necessary(output_task_folder)
        create_folder_if_necessary(output_experiment_folder)
        create_folder_if_necessary(output_dated_folder)

        # Save arguments & config to output folder
        save_json(args, output_dated_folder, filename="arguments.json")
        save_git_revision(output_dated_folder)

        if instantiate_model:
            save_json(film_model_config, output_dated_folder, filename='config_%s.json' % film_model_config['input']['type'])

            # Copy dictionary file used
            shutil.copyfile(args.dict_file_path, "%s/dict.json" % output_dated_folder)

    if use_tensorboard:
        # FIXME : What happen with test set? I guess we don't really care, we got our own visualisations for test run
        # Create tensorboard writer
        base_writer_path = '%s/%s/%s' % (args.tensorboard_folder, output_name, current_datetime_str)

        # TODO : Add 'comment' param with more infos on run. Ex : Raw vs Conv
        tensorboard_writers = {
            'train': SummaryWriter('%s/train' % base_writer_path),
            'val': SummaryWriter('%s/val' % base_writer_path)
        }
    else:
        tensorboard_writers = None

    if args.no_img_resize or film_model_config['input']['type'].lower() != 'raw':
        args.raw_img_resize_val = None

    device = 'cuda:0' if torch.cuda.is_available() and not args.use_cpu else 'cpu'
    print("Using device '%s'" % device)

    ####################################
    #   Dataloading
    ####################################

    transforms_list = []

    # Bundle together ToTensor and ImgBetweenZeroOne, need to be one after the other for other transforms to work
    to_tensor_transform = [ToTensor()]
    if not args.keep_image_range:
        to_tensor_transform.append(ImgBetweenZeroOne())

    if film_model_config['input']['type'] == 'raw':
        feature_extractor_config = {'version': 101, 'layer_index': 6}   # Idx 6 -> Block3/unit22

        if args.raw_img_resize_val:
            if args.raw_img_resize_based_on_width:
                resize_transform = ResizeImgBasedOnWidth
            else:
                # By default, we resize according to height
                resize_transform = ResizeImgBasedOnHeight
            transforms_list.append(resize_transform(args.raw_img_resize_val))

        # TODO : Add data augmentation ?

        transforms_list += to_tensor_transform

        if args.normalize_with_imagenet_stats:
            imagenet_stats = film_model_config['feature_extractor']['imagenet_stats']
            transforms_list.append(NormalizeSample(mean=imagenet_stats['mean'], std=imagenet_stats['std'], inplace=True))
        elif args.normalize_with_clear_stats:
            assert False, "Normalization with CLEAR stats not implemented"

    else:
        transforms_list += to_tensor_transform
        feature_extractor_config = None

    transforms_to_apply = transforms.Compose(transforms_list)

    print("Creating Datasets")
    train_dataset = CLEAR_dataset(args.data_root_path, args.version_name, film_model_config['input'], 'train',
                                  dict_file_path=args.dict_file_path, transforms=transforms_to_apply,
                                  tokenize_text=not args.create_dict,
                                  preprocessed_folder_name=args.preprocessed_folder_name)

    val_dataset = CLEAR_dataset(args.data_root_path, args.version_name, film_model_config['input'], 'val',
                                dict_file_path=args.dict_file_path, transforms=transforms_to_apply,
                                tokenize_text=not args.create_dict,
                                preprocessed_folder_name=args.preprocessed_folder_name)

    test_dataset = CLEAR_dataset(args.data_root_path, args.version_name, film_model_config['input'], 'test',
                                 dict_file_path=args.dict_file_path, transforms=transforms_to_apply,
                                 tokenize_text=not args.create_dict,
                                 preprocessed_folder_name=args.preprocessed_folder_name)

    #trickytest_dataset = CLEAR_dataset(data_path, film_model_config['input'], 'trickytest',
    #                             dict_file_path=args.dict_file_path,
    #                             transforms=transforms.Compose(transforms_list + [ToTensor()]))

    if args.pad_to_largest_image or args.pad_to_square_images:
        # We need the dataset object to retrieve images dims so we have to manually add transforms
        max_train_img_dims = train_dataset.get_max_width_image_dims()
        max_val_img_dims = val_dataset.get_max_width_image_dims()
        max_test_img_dims = test_dataset.get_max_width_image_dims()

        if args.pad_to_largest_image:
            train_dataset.add_transform(PadTensor(max_train_img_dims))
            val_dataset.add_transform(PadTensor(max_val_img_dims))
            test_dataset.add_transform(PadTensor(max_test_img_dims))

        if args.pad_to_square_images or args.resize_to_square_images:
            train_biggest_dim = max(max_train_img_dims)
            val_biggest_dim = max(max_val_img_dims)
            test_biggest_dim = max(max_test_img_dims)

            if args.resize_to_square_images:
                to_square_transform = ResizeTensor
            else:
                to_square_transform = PadTensor

            train_dataset.add_transform(to_square_transform((train_biggest_dim, train_biggest_dim)))
            val_dataset.add_transform(to_square_transform((val_biggest_dim, val_biggest_dim)))
            test_dataset.add_transform(to_square_transform((test_biggest_dim, test_biggest_dim)))


    print("Creating Dataloaders")
    collate_fct = CLEAR_collate_fct(padding_token=train_dataset.get_padding_token())

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=4, collate_fn=collate_fct)

    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=4, collate_fn=collate_fct)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=4, collate_fn=collate_fct)

    #trickytest_dataloader = DataLoader(trickytest_dataset, batch_size=args.batch_size, shuffle=False,
    #                             num_workers=4, collate_fn=train_dataset.CLEAR_collate_fct)

    # This should only be used once to get the dataset mean & std. We could write it to the data folder in json format
    # FIXME : THIS IS AFFECTING THE RESULTS --> Should restore rng state if doing this
    #mean, std = calc_mean_and_std(train_test_dataloader, device=device)


    ####################################
    #   Model Definition
    ####################################
    if instantiate_model:
        print("Creating model")
        # Retrieve informations to instantiate model
        nb_words, nb_answers = train_dataset.get_token_counts()
        input_image_torch_shape = train_dataset.get_input_shape(channel_first=True)  # Torch size have Channel as first dimension
        padding_token = train_dataset.get_padding_token()

        film_model = CLEAR_FiLM_model(film_model_config, input_image_channels=input_image_torch_shape[0],
                                      nb_words=nb_words, nb_answers=nb_answers,
                                      sequence_padding_idx=padding_token,
                                      feature_extraction_config=feature_extractor_config)

        if restore_model_weights:
            assert args.film_model_weight_path is not None, 'Must provide path to model weights to ' \
                                                            'do inference or to continue training.'

            # If path specified is a date, we construct the path to the best model weights for the specified run
            base_path = "%s/train_film/%s/%s" % (args.output_root_path, output_name, args.film_model_weight_path)
            # Note : We might redo some epoch when continuing training because the 'best' epoch is not necessarely the last
            suffix = "best/model.pt.tar"

            if is_date_string(args.film_model_weight_path):
                args.film_model_weight_path = "%s/%s" % (base_path, suffix)
            elif args.film_model_weight_path == 'latest':
                # The 'latest' symlink will be overriden by this run (If continuing training).
                # Use real path of latest experiment
                symlink_value = os.readlink(base_path)
                clean_base_path = base_path[:-(len(args.film_model_weight_path) + 1)]
                args.film_model_weight_path = '%s/%s/%s' % (clean_base_path, symlink_value, suffix)

            save_json({'restored_film_weight_path': args.film_model_weight_path},
                      output_dated_folder, 'restored_from.json')

            checkpoint = torch.load(args.film_model_weight_path, map_location=device)

            if device != 'cpu':
                if 'torch' in checkpoint['rng_state']:
                    checkpoint['rng_state']['torch'] = checkpoint['rng_state']['torch'].cpu()

                if 'torch_cuda' in checkpoint['rng_state']:
                    checkpoint['rng_state']['torch_cuda'] = checkpoint['rng_state']['torch_cuda'].cpu()

            # We need non-strict because feature extractor weight are not included in the saved state dict
            film_model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        if device != 'cpu':
            if args.perf_over_determinist:
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            else:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True


        film_model.to(device)

        print("Model ready to run")

        if not args.no_model_summary:
            # Printing summary affects the random state (Raw Vs Pre-Extracted Features).
            # We restore it to ensure reproducibility between input type
            random_state = get_random_state()
            summary(film_model, [(22,), (1,), input_image_torch_shape], device=device)
            set_random_state(random_state)

        if use_tensorboard and args.tensorboard_save_graph:
            # FIXME : For now we are ignoring TracerWarnings. Not sure the saved graph is 100% accurate...
            import warnings
            warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)

            # FIXME : Test on GPU
            dummy_input = [torch.ones(2, 22, dtype=torch.long),
                           torch.ones(2, 1, dtype=torch.long),
                           torch.ones(2, *input_image_torch_shape, dtype=torch.float)]
            tensorboard_writers['train'].add_graph(film_model, dummy_input)

    if create_output_folder:
        create_symlink_to_latest_folder(output_experiment_folder, current_datetime_str)

    if task == "train_film":
        trainable_parameters = filter(lambda p: p.requires_grad, film_model.parameters())

        # FIXME : Not sure what is the current behavious when specifying weight decay to Adam Optimizer. INVESTIGATE THIS
        optimizer = torch.optim.Adam(trainable_parameters, lr=film_model_config['optimizer']['learning_rate'],
                                     weight_decay=film_model_config['optimizer']['weight_decay'])

        start_epoch = 0
        if args.continue_training:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            set_random_state(checkpoint['rng_state'])

        # scheduler = torch.optim.lr_scheduler   # FIXME : Using a scheduler give the ability to decay only each N epoch.

        train_model(device=device, model=film_model, dataloaders={'train': train_dataloader, 'val': val_dataloader},
                    output_folder=output_dated_folder, criterion=nn.CrossEntropyLoss(), optimizer=optimizer,
                    nb_epoch=args.nb_epoch, nb_epoch_to_keep=args.nb_epoch_stats_to_keep,
                    start_epoch=start_epoch, tensorboard_writers=tensorboard_writers)

    elif task == "inference":
        set_inference(device=device, model=film_model, dataloader=test_dataloader, output_folder=output_dated_folder)

    elif task == "create_dict":
        create_dict_from_questions(train_dataset, force_all_answers=args.force_dict_all_answer,
                                   output_folder_name=args.dict_folder)

    elif task == "feature_extract":
        extract_features(device=device, feature_extractor=film_model.feature_extractor,
                         dataloaders={'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader},
                         output_folder_name=args.preprocessed_folder_name)

    elif task == "visualize_gamma_beta":
        visualize_gamma_beta(args.gamma_beta_path,
                             datasets={'train': train_dataset, 'val': val_dataset, 'test': test_dataset},
                             output_folder=output_dated_folder)

    elif task == "visualize_grad_cam":
        grad_cam_visualization(device=device, model=film_model, dataloader=train_dataloader,
                               output_folder=output_dated_folder)

    if tensorboard_writers:
        for key, writer in tensorboard_writers.items():
            writer.close()

    time_elapsed = str(datetime.now() - current_datetime)

    print("Execution took %s" % time_elapsed)

    if create_output_folder:
        save_json({'time_elapsed': time_elapsed}, output_dated_folder, filename='timing.json')

        print("All Done")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

        # TODO : Extract Beta And Gamma Parameters + T-SNE
        # TODO : Feed RAW image directly to the FiLM network
        # TODO : Resize images ? Do we absolutely need 224x224 for the resnet preprocessing ?
        #        Since we extract a layer in resnet, we can feed any size, its fully convolutional up to that point
        #        When feeding directly to FiLM, we can use original size ?
        # TODO : What is the optimal size for our spectrograms ?
        # TODO : Train with different amount of residual blocks. Other modifications to the architecture ?

        #   Tasks :
        #       Train from extracted features (No resnet - preprocessed)
        #       Train from Raw Images (Resnet with fixed weights)
        #       Train from Raw Images (Resnet or similar retrained (Only firsts couple layers))
        #       Run from extracted features (No resnet - preprocessed)
        #       Run from Raw Images (Resnet With fixed weights)
        #       Run from Raw Images with VISUALIZATION



