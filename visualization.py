import cv2
import tensorflow as tf
import numpy as np

from collections import defaultdict

from utils import read_gamma_beta_h5

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.io import write_html

from sklearn.manifold import TSNE as sk_TSNE
from tsnecuda import TSNE as cuda_TSNE

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


def visualize_gamma_beta(gamma_beta_path, datasets, output_folder):
    set_type, gammas_betas_dict = read_gamma_beta_h5(gamma_beta_path)
    gammas, betas, question_indexes = stack_gamma_beta_resblocks(gammas_betas_dict)
    gammas_betas = np.concatenate([gammas, betas], axis=2)
    dataset = datasets[set_type]

    # This is redundant for continuous id datasets, only useful if non continuous
    idx_to_questions = {}
    for question in dataset.questions:
        idx_to_questions[question['question_index']] = question

    questions = [idx_to_questions[idx] for idx in question_indexes]
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


def grad_cam_visualization(sess, network_wrapper, film_ckpt_path, resnet_ckpt_path):
    activated_class_index = 0

    network = network_wrapper.get_network()
    dataset = network_wrapper.get_dataset()

    sess.run(tf.variables_initializer(network_wrapper.film_variables))

    if dataset.is_raw_img():
        sess.run(tf.variables_initializer(network_wrapper.feature_extractor_variables))
        network_wrapper.restore_feature_extractor_weights(sess, resnet_ckpt_path)

    network_wrapper.restore_film_network_weights(sess, film_ckpt_path)

    samples = [dataset.games['train'][0]]

    questions = [s.question for s in samples]
    answers = [s.answer for s in samples]
    images = [s.image.get_image() for s in samples]

    questions, seq_length = dataset.tokenizer.pad_tokens(questions)

    feed_dict = network_wrapper.get_feed_dict(False, questions, answers, images, seq_length)

    nb_class = dataset.tokenizer.no_answers

    loss = tf.multiply(network.softmax, tf.one_hot([activated_class_index], nb_class))
    reduced_loss = tf.reduce_sum(loss[0])
    layer_to_visualize = network.classif_conv

    grads = tf.gradients(reduced_loss, layer_to_visualize)[0]

    output, grads_val = sess.run([layer_to_visualize, grads], feed_dict=feed_dict)

    weights = np.mean(grads_val, axis=(1,2))    # FIXME : Shouldn't it be axis (0,1) ? Maybe not because axis 0 is batch index
    cams = np.sum(weights * output, axis=3)

    cam = cams[0]
    image = np.uint8(images[0][::,::-1] * 255.0)    # RGB -> BGR
    cam = cv2.resize(cam, (224,224))
    cam = np.maximum(cam, 0)

    heatmap = cam / np.max(cam)
    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    cam = np.uint8(cam)

    cv2.imshow('orig', image)
    cv2.imshow('cam', cam)
    cv2.imshow('heatmap', (heatmap * 255.0).astype(np.uint8))
    cv2.imshow('segmentation', (heatmap[:, :, None].astype(np.float) * image).astype(np.uint8))
    cv2.waitKey(0)