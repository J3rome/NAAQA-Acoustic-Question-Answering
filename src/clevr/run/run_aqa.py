import argparse
import os
import tensorflow as tf
import logging
from generic.data_provider.image_loader import get_img_builder

from clevr.data_provider.clevr_tokenizer import CLEVRTokenizer
from generic.data_provider.iterator import Iterator
from generic.tf_utils.evaluator import MultiGPUEvaluator
from clevr.models.clever_network_factory import create_network

from multiprocessing.pool import ThreadPool
from multiprocessing import Pool

from clevr.data_provider.clevr_dataset import Image
from generic.data_provider.nlp_utils import padder
from clevr.data_provider.clevr_dataset import AQADataset
from clevr.data_provider.clevr_batchifier import CLEVRBatchifier

import hashlib
import json

##############################
#  Usage
##############################
# python src/clevr/run/run_clevr.py -config config/clevr/film.json -data_dir data/CLEVR_v1.0 -img_dir data/CLEVR_v1.0 -load_checkpoint out/clevr/run/params.ckpt


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
parser.add_argument("-gpu_ratio", type=float, default=0.95, help="How many GPU ram is required? (ratio)")
parser.add_argument("-no_gpu", type=int, default=1, help="How many GPU ?")
parser.add_argument("-no_thread", type=int, default=6, help="How many thread should be used ?")

args = parser.parse_args()
logger = logging.getLogger()

# Load config
with open(args.config, 'rb') as f_config:
  config_str = f_config.read()
  exp_identifier = hashlib.md5(config_str).hexdigest()
  config = json.loads(config_str.decode('utf-8'))

# Load images
image_builder = get_img_builder(config['model']['image'], args.img_dir, bufferize=None)

# Load dictionary
# FIXME : Do not hardcode folder name
tokenizer = CLEVRTokenizer(os.path.join(args.data_dir, 'preprocessed', config["dico_name"]))

# Load data
dataset = AQADataset(args.data_dir, which_set=args.set_type, image_builder=image_builder)

# Some parameters
batch_size = 64

# Build Network
logger.info('Building multi_gpu network..')
networks = []
tower_scope_names = []
for i in range(args.no_gpu):
  logging.info('Building network ({})'.format(i))

  with tf.device('gpu:{}'.format(i)):
    with tf.name_scope('tower_{}'.format(i)) as tower_scope:

      network = create_network(
        config=config["model"],
        no_words=tokenizer.no_words,
        no_answers=tokenizer.no_answers,
        reuse=(i > 0), device=i)

      networks.append(network)
      tower_scope_names.append(os.path.join(tower_scope, network.scope_name))


assert len(networks) > 0, "you need to set no_gpu > 0 even if you are using CPU"

# create a saver to store/load checkpoint
saver = tf.train.Saver()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_ratio)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
  # Load checkpoints or pre-trained networks
  sess.run(tf.global_variables_initializer())

  # Set the weight of the network according to the provided params.ckpt
  saver.restore(sess, args.load_checkpoint)

  # Batchifier
  sources = networks[0].get_sources(sess)
  evaluator = MultiGPUEvaluator(sources, tower_scope_names, networks=networks, tokenizer=tokenizer)
  batchifier = CLEVRBatchifier(tokenizer, sources)

  # CPU/GPU option
  # h5 requires a Tread pool while raw images requires to create new process
  if image_builder.is_raw_image():
    cpu_pool = Pool(args.no_thread, maxtasksperchild=1000)
  else:
    cpu_pool = ThreadPool(args.no_thread)
    cpu_pool._maxtasksperchild = 1000

  iterator = Iterator(dataset,
                      batch_size=batch_size,
                      batchifier=batchifier,
                      shuffle=False,
                      pool=cpu_pool)

  encoded_answers = evaluator.process(sess, iterator, outputs=[networks[0].prediction])[0]

  correct_answer_count = 0
  total_answer_count = len(dataset.games)
  results = []

  for i, game in enumerate(dataset.games):
    answer = tokenizer.decode_answer(encoded_answers[i])

    results.append({
      "question_id": game.id,
      "question": game.question,
      "generated_answer": answer,
      "ground_truth": game.answer,
      "scene_id": game.image.id
    })

    if answer == game.answer:
      correct_answer_count += 1

  print("Answered %d questions correctly on %s" % (correct_answer_count, total_answer_count))
  accuracy = correct_answer_count/total_answer_count
  print("Accuracy is : %f" % accuracy)

  if not os.path.isdir(args.exp_dir):
    os.mkdir(args.exp_dir)

  result_filepath = os.path.join(args.exp_dir, '%s_results.json' % args.set_type)
  with open(result_filepath, 'w') as f:
    json.dump(results, f, indent=2)

  accuracy_filepath = os.path.join(args.exp_dir, '%s_accuracy.json' % args.set_type)
  with open(accuracy_filepath, 'w') as f:
    json.dump({
      'set': args.set_type,
      'accuracy': accuracy
    }, f, indent=2)

  print("Results have been saved in '%s'" % result_filepath)

