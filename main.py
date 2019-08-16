import argparse
from _datetime import datetime
import subprocess
import shutil
import os

from tqdm import tqdm
import numpy as np

from utils import set_random_seed, create_folder_if_necessary, get_config, process_predictions, process_gamma_beta
from utils import create_symlink_to_latest_folder, save_training_stats, save_json, sort_stats, is_date_string
from utils import calc_mean_and_std, save_gamma_beta_h5

from visualization import visualize_gamma_beta, grad_cam_visualization
from preprocessing import create_dict_from_questions, extract_features

# NEW IMPORTS
from models.torch_film_model import CLEAR_FiLM_model
from data_interfaces.torch_dataset import CLEAR_dataset, CLEAR_collate_fct
from data_interfaces.transforms import ToTensor, ResizeImg, ImgBetweenZeroOne
from models.torchsummary import summary     # Custom version of torchsummary to fix bugs with input
import torch
import time
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms


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
parser.add_argument("--raw_img_resize", type=str, default='224,224', help="Specify the size to which the image will be"
                                                                          "resized (when working with RAW img)"
                                                                          "Format : width,height")
parser.add_argument("--keep_image_range", help="Will NOT scale the image between 0-1 (RAW img)", action='store_true')
parser.add_argument("--gamma_beta_path", type=str, default=None, help="Path where gamma_beta values are stored "
                                                                          "(when using --visualize_gamma_beta)")


# Output parameters
parser.add_argument("-output_root_path", type=str, default='output', help="Directory with image")

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


def set_inference(device, model, dataloader, output_folder):
    set_type = dataloader.dataset.set

    # TODO : load pretrained weights

    _, acc, predictions, gammas_betas = process_dataloader(False, device, model, dataloader)

    save_json(predictions, output_folder, filename='%s_predictions.json' % set_type)
    save_json(gammas_betas, output_folder, filename='%s_predictions.json' % set_type)

    print("%s Accuracy : %.5f" % (set_type, acc))


def train_model(device, model, dataloaders, output_folder, criterion=None, optimizer=None, scheduler=None,
                nb_epoch=25, nb_epoch_to_keep=None, start_epoch=0):

    stats_file_path = "%s/stats.json" % output_folder
    removed_epoch = []

    since = time.time()

    for epoch in range(start_epoch, start_epoch + nb_epoch):
        epoch_output_folder_path = "%s/Epoch_%.2d" % (output_folder, epoch)
        create_folder_if_necessary(epoch_output_folder_path)
        print('Epoch {}/{}'.format(epoch, start_epoch + nb_epoch - 1))
        print('-' * 10)

        time_before_train = datetime.now()
        train_loss, train_acc, train_predictions = process_dataloader(True, device, model,
                                                                      dataloaders['train'],
                                                                      criterion, optimizer,
                                                                      gamma_beta_path="%s/train_gamma_beta.h5" % epoch_output_folder_path)
        epoch_train_time = datetime.now() - time_before_train

        print('\n{} Loss: {:.4f} Acc: {:.4f}'.format('Train', train_loss, train_acc))

        val_loss, val_acc, val_predictions = process_dataloader(False, device, model,
                                                                dataloaders['val'], criterion,
                                                                gamma_beta_path="%s/val_gamma_beta.h5" % epoch_output_folder_path)
        print('\n{} Loss: {:.4f} Acc: {:.4f}'.format('Val', val_loss, val_acc))

        stats = save_training_stats(stats_file_path, epoch, train_acc, train_loss, val_acc, val_loss, epoch_train_time)

        save_json(train_predictions, epoch_output_folder_path, filename="train_predictions.json")
        save_json(val_predictions, epoch_output_folder_path, filename="val_predictions.json")

        print("Training took %s" % str(epoch_train_time))

        # Save training weights
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.get_cleaned_state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss
        }, '%s/model.pt.tar' % epoch_output_folder_path)

        sorted_stats = sort_stats(stats)

        if nb_epoch_to_keep is not None:
            # FIXME : Probably not the most efficient way to do this
            epoch_to_remove = sorted_stats[nb_epoch_to_keep:]

            for epoch_stat in epoch_to_remove:
                if epoch_stat['epoch'] not in removed_epoch:
                    removed_epoch.append(epoch_stat['epoch'])

                    shutil.rmtree("%s/%s" % (output_folder, epoch_stat['epoch']))

        # Create a symlink to best epoch output folder
        best_epoch = sorted_stats[0]
        print("Best Epoch is %s" % best_epoch['epoch'])
        best_epoch_symlink_path = '%s/best' % output_folder
        subprocess.run("ln -snf %s %s" % (best_epoch['epoch'], best_epoch_symlink_path), shell=True)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {}'.format(best_epoch['val_acc']))

    # TODO : load best model weights ?
    #model.load_state_dict(best_model_state)
    return model


def process_dataloader(is_training, device, model, dataloader, criterion=None, optimizer=None, gamma_beta_path=None,
                       write_to_file_every=500):
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

        if is_training:
            # zero the parameter gradients
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            outputs, outputs_softmax = model(questions, seq_lengths, images)
            _, preds = torch.max(outputs, 1)
            if criterion:
                loss = criterion(outputs, answers)

            if is_training:
                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

        batch_processed_predictions = process_predictions(dataloader.dataset, preds.tolist(), answers.tolist(),
                                                          questions_id.tolist(), scenes_id.tolist(),
                                                          outputs_softmax.tolist())

        processed_predictions += batch_processed_predictions

        gammas, betas = model.get_gammas_betas()
        processed_gammas_betas += process_gamma_beta(batch_processed_predictions, gammas, betas)

        if gamma_beta_path is not None and batch_idx % write_to_file_every == 0 and batch_idx != 0:
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

    return epoch_loss, epoch_acc, processed_predictions


def main(args):

    mutually_exclusive_params = [args.training, args.inference, args.feature_extract,
                                 args.create_dict, args.visualize_gamma_beta, args.visualize_grad_cam]

    assert sum(mutually_exclusive_params) == 1, \
        "[ERROR] Can only do one task at a time " \
        "(--training, --inference, --visualize_gamma_beta, --create_dict, --feature_extract)"

    if args.training:
        task = "train_film"
    elif args.inference:
        task = "inference"
    elif args.visualize_gamma_beta:
        task = "visualize_gamma_beta"
    elif args.visualize_grad_cam:
        task = "visualize_grad_cam"
    elif args.feature_extract:
        task = "feature_extract"
    elif args.create_dict:
        task = "create_dict"

    print("Task : %s\n" % task.replace('_', ' ').title())

    if args.random_seed is not None:
        set_random_seed(args.random_seed)

    # Paths
    data_path = "%s/%s" % (args.data_root_path, args.version_name)

    output_task_folder = "%s/%s" % (args.output_root_path, task)
    output_experiment_folder = "%s/%s" %(output_task_folder, args.version_name)
    current_datetime = datetime.now()
    current_datetime_str = current_datetime.strftime("%Y-%m-%d_%Hh%M")
    output_dated_folder = "%s/%s" % (output_experiment_folder, current_datetime_str)

    tensorboard_folder = "%s/tensorboard" % args.output_root_path

    train_writer_folder = '%s/train/%s' % (tensorboard_folder, current_datetime_str)
    val_writer_folder = '%s/val/%s' % (tensorboard_folder, current_datetime_str)
    test_writer_folder = '%s/test/%s' % (tensorboard_folder, current_datetime_str)
    beholder_folder = '%s/beholder' % tensorboard_folder

    # If path specified is a date, we construct the path to the best model weights for the specified run
    if args.film_model_weight_path is not None:

        base_path = "%s/train_film/%s/%s" % (args.output_root_path, args.version_name, args.film_model_weight_path)
        suffix = "best/model.pt.tar"        # FIXME : We might redo some epoch when continuing training because the 'best' epoch is not necessarely the last

        if is_date_string(args.film_model_weight_path):
            args.film_model_weight_path = "%s/%s" % (base_path, suffix)
        elif args.film_model_weight_path == 'latest':
            # The 'latest' symlink will be overriden by this run (If continuing training).
            # Use real path of latest experiment
            symlink_value = os.readlink(base_path)
            clean_base_path = base_path[:-(len(args.film_model_weight_path) + 1)]
            args.film_model_weight_path = '%s/%s/%s' % (clean_base_path, symlink_value, suffix)

    restore_model_weights = args.inference or (args.training and args.continue_training) or args.visualize_grad_cam

    if args.dict_file_path is None:
        args.dict_file_path = "%s/preprocessed/dict.json" % data_path

    film_model_config = get_config(args.config_path)

    create_output_folder = not args.create_dict and not args.feature_extract
    instantiate_model = not args.create_dict and 'gamma_beta' not in task

    if create_output_folder:
        # TODO : See if this is optimal file structure
        create_folder_if_necessary(args.output_root_path)
        create_folder_if_necessary(output_task_folder)
        create_folder_if_necessary(output_experiment_folder)
        create_folder_if_necessary(output_dated_folder)
        create_symlink_to_latest_folder(output_experiment_folder, current_datetime_str)

        # Save arguments & config to output folder
        save_json(args, output_dated_folder, filename="arguments.json")

        if instantiate_model:
            save_json(film_model_config, output_dated_folder, filename='config_%s.json' % film_model_config['input']['type'])

            # Copy dictionary file used
            shutil.copyfile(args.dict_file_path, "%s/dict.json" % output_dated_folder)

    if args.no_img_resize:
        args.raw_img_resize = None
    else:
        args.raw_img_resize = tuple([int(s) for s in args.raw_img_resize.split(',')])

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

        if args.raw_img_resize:
            transforms_list.append(ResizeImg(args.raw_img_resize))

        # TODO : Add data augmentation ?

        transforms_list += to_tensor_transform

        if args.normalize_with_imagenet_stats:
            imagenet_stats = film_model_config['feature_extractor']['imagenet_stats']
            transforms_list.append(transforms.Normalize(mean=imagenet_stats['mean'], std=imagenet_stats['std']))
        elif args.normalize_with_clear_stats:
            assert False, "Normalization with CLEAR stats not implemented"

    else:
        transforms_list += to_tensor_transform
        feature_extractor_config = None

    transforms_to_apply = transforms.Compose(transforms_list)

    print("Creating Datasets")
    dict_file_path = None if not args.create_dict else args.dict_file_path

    train_dataset = CLEAR_dataset(args.data_root_path, args.version_name, film_model_config['input'], 'train', dict_file_path=dict_file_path,
                                  transforms=transforms_to_apply, tokenize_text=not args.create_dict)

    val_dataset = CLEAR_dataset(args.data_root_path, args.version_name, film_model_config['input'], 'val', dict_file_path=dict_file_path,
                                transforms=transforms_to_apply, tokenize_text=not args.create_dict)

    test_dataset = CLEAR_dataset(args.data_root_path, args.version_name, film_model_config['input'], 'test', dict_file_path=dict_file_path,
                                 transforms=transforms_to_apply, tokenize_text=not args.create_dict)

    #trickytest_dataset = CLEAR_dataset(data_path, film_model_config['input'], 'trickytest',
    #                             dict_file_path=args.dict_file_path,
    #                             transforms=transforms.Compose(transforms_list + [ToTensor()]))

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
    # FIXME : THIS IS AFFECTING THE RESULTS --> Changing the seed. Could be a preprocessing step ?
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

            checkpoint = torch.load(args.film_model_weight_path, map_location=device)
            # We need non-strict because feature extractor weight are not included in the saved state dict
            film_model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        if device != 'cpu':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True

        film_model.to(device)

        print("Model ready to run")

        # FIXME : Printing summary affect the output of the model (RAW vs Conv)
        #         Doesn't seem to be a random state problem (At least not torch.randn())
        summary(film_model, [(22,), (1,), input_image_torch_shape], device=device)

    if task == "train_film":
        trainable_parameters = filter(lambda p: p.requires_grad, film_model.parameters())

        # FIXME : Not sure what is the current behavious when specifying weight decay to Adam Optimizer. INVESTIGATE THIS
        optimizer = torch.optim.Adam(trainable_parameters, lr=film_model_config['optimizer']['learning_rate'],
                                     weight_decay=film_model_config['optimizer']['weight_decay'])

        start_epoch = 0
        if args.continue_training:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1

        # scheduler = torch.optim.lr_scheduler   # FIXME : Using a scheduler give the ability to decay only each N epoch.

        train_model(device=device, model=film_model, dataloaders={'train': train_dataloader, 'val': val_dataloader},
                    output_folder=output_dated_folder, criterion=nn.CrossEntropyLoss(), optimizer=optimizer,
                    nb_epoch=args.nb_epoch, nb_epoch_to_keep=args.nb_epoch_stats_to_keep,
                    start_epoch=start_epoch)

    elif task == "inference":
        set_inference(device=device, model=film_model, dataloader=test_dataloader, output_folder=output_dated_folder)

    elif task == "create_dict":
        create_dict_from_questions(train_dataset, force_all_answers=args.force_dict_all_answer)

    elif task == "feature_extract":
        extract_features(device=device, feature_extractor=film_model.feature_extractor,
                         dataloaders={'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader})

    elif task == "visualize_gamma_beta":
        visualize_gamma_beta(args.gamma_beta_path,
                             datasets={'train': train_dataset, 'val': val_dataset, 'test': test_dataset},
                             output_folder=output_dated_folder)

    elif task == "visualize_grad_cam":
        grad_cam_visualization(device=device, model=film_model, dataloader=train_dataloader,
                               output_folder=output_dated_folder)

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



