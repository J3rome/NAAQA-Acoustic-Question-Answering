import cv2
import tensorflow as tf
import numpy as np

from collections import defaultdict

from utils import read_gamma_beta_h5

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

import plotly.graph_objects as go

def visualize_gamma_beta(gamma_beta_path, datasets):
    set_type, gammas_betas = read_gamma_beta_h5(gamma_beta_path)

    dataset = datasets[set_type]

    # This is redundant for continuous id datasets, only useful if non continuous
    idx_to_questions = {}
    for question in dataset.questions:
        idx_to_questions[question['question_index']] = question

    resblock_keys = sorted(list(set(gammas_betas[0].keys()) - {'question_index'}))
    nb_dim_resblock = len(gammas_betas[0]['resblock_0']['gamma_vector'])

    gamma_per_resblock = defaultdict(lambda : [])
    beta_per_resblock = defaultdict(lambda: [])
    gamma_beta_per_resblock = defaultdict(lambda: [])



    questions = []
    questions_type = []
    for gamma_beta in gammas_betas:
        question = idx_to_questions[gamma_beta['question_index']]
        question['type'] = get_question_type(question['program'])
        questions_type.append(question['type'])
        questions.append(question)

        for resblock_key in resblock_keys:
            gamma_per_resblock[resblock_key].append(gamma_beta[resblock_key]['gamma_vector'])
            beta_per_resblock[resblock_key].append(gamma_beta[resblock_key]['beta_vector'])
            gamma_beta_per_resblock[resblock_key].append(np.concatenate([gamma_beta[resblock_key]['gamma_vector'],
                                                                         gamma_beta[resblock_key]['beta_vector']]))

    for resblock_key in resblock_keys:
        gamma_per_resblock[resblock_key] = np.stack(gamma_per_resblock[resblock_key], axis=0)
        beta_per_resblock[resblock_key] = np.stack(beta_per_resblock[resblock_key], axis=0)
        gamma_beta_per_resblock[resblock_key] = np.stack(gamma_beta_per_resblock[resblock_key], axis=0)



    # TODO : Beta vs Gamma in 2d plot (For each feature maps)
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



    # TODO : T-SNE : Gamma, Beta, Gamma Beta together





    print("Gamma beta")
