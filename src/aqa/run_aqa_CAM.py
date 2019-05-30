import argparse
import os
import tensorflow as tf
import logging
from generic.data_provider.image_loader import get_img_builder

import numpy as np

# Visualisation
from tf_cnnvis import *

# FIXME : Extract from generic code
from clevr.models.clevr_film_network import FiLM_CLEVRNetwork
from clevr.data_provider.clevr_tokenizer import CLEVRTokenizer
from clevr.data_provider.clevr_dataset import Image
from generic.data_provider.nlp_utils import padder

import hashlib
import json

##############################
#  Usage
##############################
# python src/clevr/run/run_clevr.py -config config/clevr/film.json -data_dir data/CLEVR_v1.0 -img_dir data/CLEVR_v1.0 -load_checkpoint out/clevr/run/params.ckpt

# python src/clevr/run/run_aqa_CAM.py -config config/clevr/film.json -data_dir data/{VERSION} -img_dir data/{VERSION}/preprocessed -load_checkpoint out/{VERSION}/{CONFIG_HASH}/film-checkpoint-best


###############################
#  LOAD CONFIG
#############################

parser = argparse.ArgumentParser('AQA network baseline!', fromfile_prefix_chars='@')

parser.add_argument("-data_dir", type=str, help="Directory with data")
parser.add_argument("-img_dir", type=str, help="Directory with image")
parser.add_argument("-exp_dir", type=str, help="Directory in which experiments are stored")
parser.add_argument("-set_type", type=str, help="Set type {train, val, test}")
parser.add_argument("-load_checkpoint", type=str, help="Load model parameters from specified checkpoint")
parser.add_argument("-config", type=str, help='Config file')
parser.add_argument("-dict_path", type=str, default=None, help="Path to dictionary file")


parser.add_argument("-gpu_ratio", type=float, default=0.95, help="How many GPU ram is required? (ratio)")
parser.add_argument("-no_gpu", type=int, default=1, help="How many GPU ?")
parser.add_argument("-no_thread", type=int, default=6, help="How many thread should be used ?")


def load_image(image_id, set_type, image_builder):
  if not type(image_id) == str:
    image_id = "%06d" % int(image_id)

  filename = "CLEAR_%s_%s.png" % (set_type, image_id)
  return Image(int(image_id), filename, image_builder, set_type).get_image()


def load_questions(question_path):
  with open(question_path, 'r') as f:
    questions = json.load(f)['questions']

  for question in questions:
    # Not needing the program
    del question['program']

  return questions


def prepare_input(network, image, question, answer, tokenizer, is_training=False):
  tokenized_question = tokenizer.encode_question(question)
  tokenized_question, seq_length = padder([tokenized_question], padding_symbol=tokenizer.padding_token)

  return {
    network._question: tokenized_question,
    network._image: np.expand_dims(image, 0),
    network._answer: np.expand_dims(tokenizer.encode_answer(answer), 0),
    network._seq_length : seq_length,
    network._is_training: is_training
  }

  # return {
  #   'clevr/question:0': tokenized_question,
  #   'clevr/image:0': [image],
  #   'clevr/answer:0' : [tokenizer.encode_answer(answer)],
  #   'clevr/seq_length:0': seq_length,
  #   'clevr/is_training:0': is_training
  # }

def main():
  args = parser.parse_args()
  logger = logging.getLogger()

  # Load config
  with open(args.config, 'rb') as f_config:
    config_str = f_config.read()
    exp_identifier = hashlib.md5(config_str).hexdigest()
    config = json.loads(config_str.decode('utf-8'))

  # Load dictionary
  if args.dict_path:
    dict_path = args.dict_path
  else:
    # FIXME : Do not hardcode folder name
    dict_path = os.path.join(args.data_dir, 'preprocessed', config["dico_name"])

  tokenizer = CLEVRTokenizer(dict_path)
  image_builder = get_img_builder(config['model']['image'], args.img_dir, bufferize=None)
  network = FiLM_CLEVRNetwork(config['model'], no_words=tokenizer.no_words, no_answers=tokenizer.no_answers, device=0)  # FIXME : Not sure that device 0 is appropriate for CPU

  questions = load_questions(os.path.join(args.data_dir, 'questions', 'CLEAR_%s_questions.json' % args.set_type))

  # create a saver to store/load checkpoint
  saver = tf.train.Saver()

  # FIXME : Gpu related
  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_ratio)
  gpu_options = tf.GPUOptions(allow_growth=True)
  session_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)



  with tf.Session(config=session_config) as sess:
    # Load checkpoints or pre-trained networks
    sess.run(tf.global_variables_initializer())

    # Set the weight of the network according to the provided params.ckpt
    saver.restore(sess, args.load_checkpoint)

    # retrieve gamma and beta
    gamma_beta = network.resblocks[0].gamma_beta
    writer = tf.summary.FileWriter('./tensorboard', sess.graph)


    # TODO : Group questions by type
    # TODO : Batchify the questions
    for question in questions:
      scene_spectrogram = load_image(question['scene_index'], args.set_type, image_builder)

      feed_dict = prepare_input(network, scene_spectrogram, question['question'], question['answer'], tokenizer)

      #prediction = sess.run([network.prediction, network._image, network.classif_conv, network.max_pool, tf.summary.merge_all()], feed_dict=feed_dict)
      prediction = sess.run([network.prediction, gamma_beta], feed_dict=feed_dict)

      print("Image Id : " + str(question["scene_index"]))
      print("Question asked : " + question['question'])
      print("Answer received : " + tokenizer.decode_answer(prediction[0][0]))
      print("Ground truth : " + str(question['answer']))

      layers = ["r", "p", "c"]

      is_success = activation_visualization(sess_graph_path = sess, value_feed_dict = feed_dict,
                                  input_tensor=network._image, layers=layers, 
                                  path_logdir=os.path.join("Vis_Log","FirstRun"), 
                                  path_outdir=os.path.join("Vis_Output","FirstRun"))
      print("Is Sucess : " + str(is_success))


      return

    # TODO : Add tensorboard


if __name__ == "__main__":
  main()

