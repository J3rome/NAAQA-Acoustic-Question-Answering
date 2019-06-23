import tensorflow as tf
from aqa.models.resnet import create_resnet
from aqa.models.film_network import FiLM_Network

class FiLM_Network_Wrapper():
    def __init__(self, config, dataset):

        self.config = config
        self.dataset = dataset

        if self.config['input']['type'] == 'raw':
            self.input_image = tf.placeholder(tf.float32, [dataset.batch_size] + self.config['input']['dim'], name='clear/image')  # FIXME : would it be better to use a fixed batch_size instead of None ?

            if "resnet" in self.config['feature_extractor']['type'].lower():
                # Feature extractor (Resnet 101)
                self.feature_extractor, feature_extractor_variables = create_resnet(self.input_image, resnet_version=self.config['feature_extractor']['version'],
                                                               chosen_layer=self.config['feature_extractor']['output_layer'], is_training=False)

                self.feature_extractor_saver = tf.train.Saver(var_list=feature_extractor_variables)

                film_input_tensor = self.feature_extractor
            else:
                print("[ERROR] Only Resnet feature extractor is implemented.")
                exit(1)
        elif self.config['input']['type'] == 'conv':
            self.input_image = tf.placeholder(tf.float32, [dataset.batch_size] + dataset.input_shape, name='clear/image')  # FIXME : would it be better to use a fixed batch_size instead of None ?
            film_input_tensor = self.input_image
        else:
            print("[ERROR] input type '%s' not implemented." % self.config['input']['type'])
            exit(1)

        # Adding the FiLM network after the chosen resnet layer
        self.film_network = FiLM_Network(config, input_image_tensor=film_input_tensor,
                               no_words=dataset.tokenizer.no_words, no_answers=dataset.tokenizer.no_answers,
                               device=0)  # FIXME : Not sure that device 0 is appropriate for CPU

        #self.network.get_parameters()      # FIXME : network.get_parameters() doesn't return the moving_mean
        film_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="clear")
        self.film_network_saver = tf.train.Saver(max_to_keep=None, var_list=film_variables)

    def get_input_image(self):
        return self.input_image

    def get_feature_extractor(self):
        return self.feature_extractor

    def get_network(self):
        return self.film_network

    def get_network_prediction(self):
        return self.film_network.get_prediction()

    def get_network_accuracy(self):
        return self.film_network.get_accuracy()

    def restore_feature_extractor_weights(self, sess, ckpt_path):
        self.feature_extractor_saver.restore(sess, ckpt_path)

    def restore_film_network_weights(self, sess, ckpt_path):
        # FIXME : Restore only inference weights ? (Should also be able to restore training weights via parameter)
        self.film_network_saver.restore(sess, ckpt_path)

    def save_film_checkpoint(self, sess, checkpoint_path):
        # FIXME : Does splitting in folder make it so that we loose the ability to keep only the last X
        self.film_network_saver.save(sess, checkpoint_path)

    def get_feed_dict(self, is_training, question, answer, image, seq_length):
        return self.film_network.get_feed_dict(is_training, question, answer, image, seq_length, image_var=self.input_image)
