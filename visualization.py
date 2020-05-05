import numpy as np

from collections import defaultdict

from utils.file import read_gamma_beta_h5

from utils.visualization import visualize_cam
from utils.processing import process_predictions

import plotly.graph_objects as go
from plotly.io import write_html

from sklearn.manifold import TSNE as sk_TSNE
from tsnecuda import TSNE as cuda_TSNE

from models.tools.gradcam import GradCAM

from torchvision.utils import make_grid

from data_interfaces.CLEAR_dataset import CLEAR_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from tqdm import tqdm

special_ending_nodes_correspondence = {
  'add': 'count',
  'relate_filter_count': 'count',
  'filter_count': 'count',
  'count_different_instrument': 'count',
  'or':  'exist',
  'relate_filter_exist': 'exist',
  'filter_exist': 'exist',
  'equal_integer': 'compare_integer',
  'greater_than': 'compare_integer',
  'less_than': 'compare_integer',
  'query_position': 'query_position_absolute',
  'query_human_note': 'query_musical_note'

}

special_intermediary_nodes_correspondence = {
  'duration': ['filter_longest_duration', 'filter_shortest_duration'],
  'relation': ['relate_filter', 'relate_filter_unique', 'relate_filter_not_unique', 'relate_filter_count', 'relate_filter_exist']
}


def get_question_type(question_nodes):
    last_node_type = question_nodes[-1]['type']

    if last_node_type in special_ending_nodes_correspondence:
        last_node_type = special_ending_nodes_correspondence[last_node_type]

    return last_node_type.title().replace('_', ' ')


def gamma_beta_2d_vis_per_feature_map(gamma_per_resblock, beta_per_resblock, resblock_keys, nb_dim_resblock, questions_type):
    question_type_color_map = {q_type: i for i, q_type in enumerate(set(questions_type))}

    question_type_colors = [question_type_color_map[q_type] for q_type in questions_type]

    for resblock_key in resblock_keys:
        gamma = gamma_per_resblock[resblock_key]
        beta = beta_per_resblock[resblock_key]
        for dim_idx in range(nb_dim_resblock):
            fig = go.Figure(data=go.Scattergl(x=gamma[:, dim_idx],
                                              y=beta[:, dim_idx],
                                              mode='markers',
                                              text=questions_type,
                                              marker_color=question_type_colors)
                            )
            fig.update_layout(title="Gamma Vs Beta -- %s dim %d" % (resblock_key, dim_idx))

            fig.show()


def do_tsne(values, with_cuda=False):
    if with_cuda:
        TSNE = cuda_TSNE
    else:
        TSNE = sk_TSNE

    model = TSNE(n_components=2, perplexity=30, early_exaggeration=12.0, learning_rate=200, n_iter=1000,
                 n_iter_without_progress=300, min_grad_norm=1e-7,
                 #metric, init, method,
                 )

    return model.fit_transform(values)


def stack_gamma_beta_resblocks(gammas_betas):
    gammas = []
    betas = []
    question_indexes = []
    for gamma_beta in gammas_betas:
        gamma_resblocks = []
        beta_resblocks = []

        sorted_iterator = sorted(gamma_beta.items(), key=lambda x: x[0])
        for key, val in sorted_iterator:
            if key == 'question_index':
                question_indexes.append(val)
            else:
                gamma_resblocks.append(val['gamma_vector'])
                beta_resblocks.append(val['beta_vector'])
        gammas.append(np.stack(gamma_resblocks, axis=0))
        betas.append(np.stack(beta_resblocks, axis=0))

    return np.stack(gammas, axis=0), np.stack(betas, axis=0), question_indexes


def visualize_gamma_beta(gamma_beta_path, dataloaders, output_folder):
    set_type, gammas_betas_dict = read_gamma_beta_h5(gamma_beta_path)
    gammas, betas, question_indexes = stack_gamma_beta_resblocks(gammas_betas_dict)
    gammas_betas = np.concatenate([gammas, betas], axis=2)
    dataset = dataloaders[set_type].dataset

    # This is redundant for continuous id datasets, only useful if non continuous
    idx_to_questions = {}
    for question in dataset.questions:
        idx_to_questions[question['question_index']] = question

    questions = [idx_to_questions[idx] for idx in question_indexes]
    # FIXME : Question type now included in dataset object
    question_types = [get_question_type(q['program']) for q in questions]

    question_type_to_color_map = {q_type: i for i, q_type in enumerate(set(question_types))}
    question_types = {
        'labels': question_types,
        'colors': [question_type_to_color_map[q_type] for q_type in question_types],
        'type_to_color': question_type_to_color_map
    }

    print("Starting T-SNE Computation")
    gamma_figs = plot_tsne_per_resblock(gammas, question_types, 'Gamma T-SNE')
    beta_figs = plot_tsne_per_resblock(betas, question_types, 'Beta T-SNE')
    gamma_beta_figs = plot_tsne_per_resblock(gammas_betas, question_types, 'Gamma and Beta T-SNE')

    all_figs = gamma_figs + beta_figs + gamma_beta_figs

    for fig in all_figs:
        filename = fig.layout.title.text.replace(' ', '_')
        write_html(fig, '%s/%s.html' % (output_folder, filename))

    print("Visualizations have been written to '%s'." % output_folder)


def plot_tsne_per_resblock(vals, question_types, title="T-SNE"):
    nb_vals, nb_resblock, nb_dim = vals.shape

    idx_grouped_by_type = defaultdict(lambda : [])

    for i, type in enumerate(question_types['labels']):
        idx_grouped_by_type[type].append(i)

    # Apply T-SNE
    embeddings = []
    for i in range(nb_resblock):
        embeddings.append(do_tsne(vals[:, i, :]))

    figs = []
    for i in range(nb_resblock):
        fig = go.Figure()
        fig.update_layout(title= "%s - Resblock %d" % (title, i))
        figs.append(fig)

    # TODO : better management of traces visility -- Add Drodown for different metrics
    traces = defaultdict(lambda : [])

    for question_type, indexes in idx_grouped_by_type.items():

        for i in range(nb_resblock):
            embedding = embeddings[i]

            fig = figs[i]

            traces[question_type].append(fig.add_trace(
                go.Scattergl(x=embedding[indexes, 0], y=embedding[indexes, 1], mode='markers', text=question_type,
                             marker_color=question_types['type_to_color'][question_type], name=question_type
                             )
                )
            )

    return figs

from utils.visualization import get_gradcam_heatmap, merge_gradcam_heatmap_with_image
def one_game_gradcam(device, model, game, collate_fn, class_idx=None, return_heatmap=True, apply_relu=False,
                     target_layers=None):
    one_game_batch = collate_fn([game])

    # Set up model in eval mode
    model.eval()

    # Copy data to GPU
    image = one_game_batch['image'].to(device)
    question = one_game_batch['question'].to(device)
    seq_length = one_game_batch['seq_length'].to(device)

    if target_layers is None:
        # FIXME : Should we target directly the conv layer instead of relu ?
        # FIXME : This is not scalable, if the network architecture change the target layers need to be changed
        target_layers = {
            #"stem_conv": model.stem_conv[0],
            'resblocks[0].conv1': model.resblocks[0].conv1[0],
            'resblocks[0].conv2': model.resblocks[0].conv2[0],
            'resblocks[0].film_layer': model.resblocks[0].film_layer[0],
            'resblocks[0].out': model.resblocks[0],
            'classif_conv': model.classifier.classif_conv[0]
        }

    cam_model = GradCAM(model, target_layers, apply_relu=apply_relu)

    if class_idx is None:
        class_idx = one_game_batch['answer']
    elif type(class_idx) == int:
        class_idx = torch.tensor(class_idx).unsqueeze(0)
    elif isinstance(class_idx, torch.Tensor):
        while len(class_idx.shape) < 1:
            class_idx = class_idx.unsqueeze(0)

    class_idx = class_idx.to(device)

    saliency_maps, confidence, logit = cam_model(question, seq_length, image, class_idx=class_idx)

    images_to_return = {}

    for layer_name, saliency_map in saliency_maps.items():
        if return_heatmap:
            heatmap = get_gradcam_heatmap(saliency_map)
            images_to_return[layer_name] = heatmap
        else:
            images_to_return[layer_name] = saliency_map.squeeze(0)

    # Since our batch contain only 1 game, we squeeze() the first dimension
    return images_to_return, confidence.squeeze(0)


def grad_cam_visualization(device, model, dataloader, output_folder, nb_game_per_img=10, limit_dataset=45):
    orig_dataloader = dataloader

    # FIXME : Should only be allowed for raw and raw_h5 ?
    #assert orig_dataloader.dataset.is_raw_img(), 'Only support raw img config for now'

    if limit_dataset is None:
        dataset = dataloader.dataset
    else:
        dataset = CLEAR_dataset.from_dataset_object(orig_dataloader.dataset,
                                                    orig_dataloader.dataset.games[:limit_dataset])

    dataloader = DataLoader(dataset, shuffle=False, num_workers=1, collate_fn=orig_dataloader.collate_fn,
                            batch_size=min(len(dataset), dataloader.batch_size))

    model.eval()

    cam = GradCAM(model, model.classif_conv)
    # TODO : GradCam PP
    # TODO : Guided Backprop ?

    # TODO : Annotate ground truth image - Rectangle over duration of queried sound
    #        What happen when we have multiple sounds used to answer ? (Ex : count) Put annotation on all sounds ?

    toPILimg_transform = transforms.ToPILImage()

    images = []
    nb_game_left = nb_game_per_img
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        input_images = batch['image'].to(device)
        questions = batch['question'].to(device)
        answers = batch['answer'].to(device)
        seq_lengths = batch['seq_length'].to(device)

        # Those are not processed by the network, only used to create statistics. Therefore, no need to copy to GPU
        questions_id = batch['id']
        scenes_id = batch['scene_id']
        images_padding = batch['image_padding']

        batch_size = input_images.size(0)

        # TODO : We might want to use something else than ground truths for class_idx (Ex : Highest prob prediction -> Pass class_idx = None)
        #        Might want to compare ground truth CAM with answer in the case in the case of wrong answers
        saliency_maps, confidences, model_outputs = cam(questions, seq_lengths, input_images, class_idx=answers)

        _, preds = torch.max(model_outputs, 1)

        processed_predictions = process_predictions(dataloader.dataset, preds.tolist(), answers.tolist(),
                                                    questions_id.tolist(), scenes_id.tolist(),
                                                    model_outputs.tolist(), images_padding.tolist())

        iterator = zip(range(batch_size), saliency_maps, input_images, confidences, processed_predictions)
        for i, saliency_map, input_image, confidence, processed_prediction in iterator:
            # TODO : Save visualization to disk, plotly ?
            heatmap, result = visualize_cam(saliency_map, input_image)
            print("Confidence %f -- Correct : %s -- Correct Family : %s" % (confidence,
                                                                            processed_prediction['correct'],
                                                                            processed_prediction['correct_answer_family']))
            # TODO : Add confidence to the graph

            images += [input_image.detach(), heatmap.detach(), result.detach()]

            nb_game_left -= 1
            if nb_game_left == 0:
                grid = toPILimg_transform(make_grid(images, nrow=3))
                current_idx = batch_idx * dataloader.batch_size + i
                filepath = '%s/gradcam_visualization_%05d-%05d.png' % (output_folder, current_idx - nb_game_per_img + 1, current_idx)
                grid.save(filepath)

                nb_game_left = nb_game_per_img
                images = []

    nb_game_to_write_left = len(images) / 3
    if nb_game_to_write_left > 0:
        grid = toPILimg_transform(make_grid(images, nrow=3))
        current_idx = len(dataloader.dataset) - 1
        filepath = '%s/gradcam_visualization_%05d-%05d.png' % (
        output_folder, current_idx - nb_game_to_write_left, current_idx)
        grid.save(filepath)

    print("GradCAM Visualization Done. See %s for results" % output_folder)

