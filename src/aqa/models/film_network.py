import tensorflow as tf
import tensorflow.contrib.layers as tfc_layers
import tensorflow.contrib.rnn as tfc_rnn
import tensorflow.contrib.slim as slim

from aqa.models.abstract_network import ResnetModel     # TODO : Verify, is this really usefull ?

import aqa.models.utils as model_utils

def film_layer(ft, context, reuse=False):
    """
    A very basic FiLM layer with a linear transformation from context to FiLM parameters
    :param ft: features map to modulate. Must be a 3-D input vector (+batch size)
    :param context: conditioned FiLM parameters. Must be a 1-D input vector (+batch size)
    :param reuse: reuse variable, e.g, multi-gpu
    :return: modulated features
    """

    height = int(ft.get_shape()[1])
    width = int(ft.get_shape()[2])
    feature_size = int(ft.get_shape()[3])

    film_params_vector = slim.fully_connected(context,
                                       num_outputs=2 * feature_size,
                                       activation_fn=None,
                                       reuse=reuse,
                                       scope="film_projection")

    film_params = tf.expand_dims(film_params_vector, axis=[1])
    film_params = tf.expand_dims(film_params, axis=[1])
    film_params = tf.tile(film_params, [1, height, width, 1])

    gammas = film_params[:, :, :, :feature_size]
    betas = film_params[:, :, :, feature_size:]

    output = (1 + gammas) * ft + betas

    return output, film_params_vector


class FiLMResblock(object):
    def __init__(self, features, context, is_training,
                 film_layer_fct=film_layer,
                 kernel1=list([1, 1]),
                 kernel2=list([3, 3]),
                 spatial_location=True, reuse=None):

        # Retrieve the size of the feature map
        feature_size = int(features.get_shape()[3])

        # Append a mask with spatial location to the feature map
        if spatial_location:
            features = model_utils.append_spatial_location(features)

        # First convolution
        self.conv1_out = slim.conv2d(features,
                                 num_outputs=feature_size,
                                 kernel_size=kernel1,
                                 activation_fn=tf.nn.relu,
                                 scope='conv1',
                                 reuse=reuse)

        # Second convolution
        self.conv2 = slim.conv2d(self.conv1_out,
                                 num_outputs=feature_size,
                                 kernel_size=kernel2,
                                 activation_fn=None,
                                 scope='conv2',
                                 reuse=reuse)

        # Center/reduce output (Batch Normalization with no training parameters)
        self.conv2_bn = slim.batch_norm(self.conv2,
                                        center=False,
                                        scale=False,
                                        scope='conv2_bn',
                                        decay=0.9,
                                        is_training=is_training,
                                        reuse=reuse)

        # Apply FILM layer Residual connection
        with tf.variable_scope("FiLM", reuse=reuse):
            self.conv2_film, self.gamma_beta = film_layer_fct(self.conv2_bn, context, reuse=reuse)

        # Apply ReLU
        self.conv2_out = tf.nn.relu(self.conv2_film)

        # Residual connection
        self.output = self.conv2_out + self.conv1_out

    def get(self):
        return self.output


class FiLM_Network(ResnetModel):

    def __init__(self, config, no_words, no_answers, input_image_tensor = None, reuse=False, device=''):
        ResnetModel.__init__(self, "clevr", device=device)      # TODO : Change scope to clear

        with tf.variable_scope(self.scope_name, reuse=reuse):

            self.batch_size = None
            self._is_training = tf.placeholder(tf.bool, name="is_training")

            dropout_keep_scalar = float(config.get("dropout_keep_prob", 1.0))
            dropout_keep = tf.cond(self._is_training,
                                   lambda: tf.constant(dropout_keep_scalar),
                                   lambda: tf.constant(1.0))

            #####################
            #   QUESTION
            #####################

            self._question = tf.placeholder(tf.int32, [self.batch_size, None], name='question')
            self._seq_length = tf.placeholder(tf.int32, [self.batch_size], name='seq_length')
            self._answer = tf.placeholder(tf.int64, [self.batch_size], name='answer')

            word_emb = tfc_layers.embed_sequence(
                ids=self._question,
                vocab_size=no_words,
                embed_dim=config["question"]["word_embedding_dim"],
                scope="word_embedding",
                reuse=reuse)

            # word_emb = tf.nn.dropout(word_emb, dropout_keep)
            rnn_cell = tfc_rnn.GRUCell(
                num_units=config["question"]["rnn_state_size"],
                activation=tf.nn.tanh,
                reuse=reuse)

            _, self.rnn_state = tf.nn.dynamic_rnn(
                cell=rnn_cell,
                inputs=word_emb,
                dtype=tf.float32,
                sequence_length=self._seq_length)

            #####################
            #   IMAGES
            #####################

            if input_image_tensor is None:
                self._image = tf.placeholder(tf.float32, [self.batch_size] + config['image']["dim"], name='image')
            else:
                self._image = input_image_tensor        # FIXME : Make sure we have the correct scope

            assert len(self._image.get_shape()) == 4, \
                "Incorrect image input and/or attention mechanism (should be none)"

            #####################
            #   STEM
            #####################

            with tf.variable_scope("stem", reuse=reuse):

                stem_features = self._image
                if config["stem"]["spatial_location"]:
                    stem_features = model_utils.append_spatial_location(stem_features)

                self.stem_conv = tfc_layers.conv2d(stem_features,
                                                   num_outputs=config["stem"]["conv_out"],
                                                   kernel_size=config["stem"]["conv_kernel"],
                                                   normalizer_fn=tf.layers.batch_normalization,
                                                   normalizer_params={"training": self._is_training, "reuse": reuse},
                                                   activation_fn=tf.nn.relu,
                                                   reuse=reuse,
                                                   scope="stem_conv")

            #####################
            #   FiLM Layers
            #####################

            with tf.variable_scope("resblocks", reuse=reuse):

                res_output = self.stem_conv
                self.resblocks = []

                for i in range(config["resblock"]["no_resblock"]):
                    with tf.variable_scope("ResBlock_{}".format(i), reuse=reuse):

                        #rnn_dropout = tf.nn.dropout(self.rnn_state , dropout_keep)

                        resblock = FiLMResblock(res_output, self.rnn_state,
                                                 kernel1=config["resblock"]["kernel1"],
                                                 kernel2=config["resblock"]["kernel2"],
                                                 spatial_location=config["resblock"]["spatial_location"],
                                                 is_training=self._is_training,
                                                 reuse=reuse)

                        self.resblocks.append(resblock)
                        res_output = resblock.get()

            #####################
            #   Classifier
            #####################

            with tf.variable_scope("classifier", reuse=reuse):

                classif_features = res_output
                if config["classifier"]["spatial_location"]:
                    classif_features = model_utils.append_spatial_location(classif_features)

                # 2D-Conv
                self.classif_conv = tfc_layers.conv2d(classif_features,
                                                      num_outputs=config["classifier"]["conv_out"],
                                                      kernel_size=config["classifier"]["conv_kernel"],
                                                      normalizer_fn=tf.layers.batch_normalization,
                                                      normalizer_params={"training": self._is_training, "reuse": reuse},
                                                      activation_fn=tf.nn.relu,
                                                      reuse=reuse,
                                                      scope="classifier_conv")

                self.max_pool = tf.reduce_max(self.classif_conv, axis=[1,2], keep_dims=False, name="global_max_pool")

                self.hidden_state = tfc_layers.fully_connected(self.max_pool,
                                                               num_outputs=config["classifier"]["no_mlp_units"],
                                                               normalizer_fn=tf.layers.batch_normalization,
                                                               normalizer_params= {"training": self._is_training, "reuse": reuse},
                                                               activation_fn=tf.nn.relu,
                                                               reuse=reuse,
                                                               scope="classifier_hidden_layer")

                self.out = tfc_layers.fully_connected(self.hidden_state,
                                                             num_outputs=no_answers,
                                                             activation_fn=None,
                                                             reuse=reuse,
                                                             scope="classifier_softmax_layer")

            #####################
            #   Loss
            #####################

            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.out, labels=self._answer, name='cross_entropy')
            self.loss = tf.reduce_mean(self.cross_entropy)

            self.softmax = tf.nn.softmax(self.out, name='answer_prob')
            self.prediction = tf.argmax(self.out, axis=1, name='predicted_answer')  # no need to compute the softmax

            with tf.variable_scope('accuracy'):
                self.accuracy = tf.equal(self.prediction, self._answer)
                self.accuracy = tf.reduce_mean(tf.cast(self.accuracy, tf.float32))

            tf.summary.scalar('accuracy', self.accuracy)

            print('FiLM Model... built!')

    def get_feed_dict(self, image_var,  is_training, question, answer, image, seq_length):
        return {
            self._is_training : is_training,
            self._question : question,
            self._answer : answer,
            #self._image : image,
            image_var: image,
            self._seq_length : seq_length
        }

    def get_loss(self):
        return self.loss

    def get_accuracy(self):
        return self.accuracy
