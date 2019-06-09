import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.resnet_v1 as resnet_v1
import tensorflow.contrib.slim.python.slim.nets.resnet_utils as slim_utils

import os

def create_resnet(image_input, is_training, scope="", chosen_layer="block4", resnet_version=101):
    """
    Create a resnet by overidding the classic batchnorm with conditional batchnorm
    :param image_input: placeholder with image
    :param is_training: are you using the resnet at training_time or test_time
    :param scope: tensorflow scope
    :param resnet_version: 50/101/152
    :param chosen_layer: name of the chosen output layer
    :return: the resnet output
    """

    # Pick the correct version of the resnet
    if resnet_version == 50:
        resnet_network = resnet_v1.resnet_v1_50
    elif resnet_version == 101:
        resnet_network = resnet_v1.resnet_v1_101
    elif resnet_version == 152:
        resnet_network = resnet_v1.resnet_v1_152
    else:
        raise ValueError("Unsupported resnet version")

    resnet_scope = os.path.join('resnet_v1_{}/'.format(resnet_version), chosen_layer)

    with slim.arg_scope(slim_utils.resnet_arg_scope()):
        net, end_points = resnet_network(image_input, 1000, is_training=is_training)  # 1000 is the number of softmax class

    # TODO : Verify, what is the use of this scope parameter ?
    if len(scope) > 0 and not scope.endswith("/"):
        scope += "/"

    # FIXME : The graph still have reference to layers after the chosen layer. Can we delete them ?
    # FIXME : Shouldn't have a significant performance impact since the floating nodes won't be executed
    chosen_resnet_layer = end_points[scope + resnet_scope]

    return chosen_resnet_layer
