import cv2
import tensorflow as tf
import numpy as np

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


def visualize_gamma_beta(gamma_beta_path, datasets):
    set_type, gammas_betas = read_gamma_beta_h5(gamma_beta_path)

    dataset = datasets[set_type]

    print("Gamma beta")
