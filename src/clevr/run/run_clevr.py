import argparse
import os
import tensorflow as tf
from generic.data_provider.image_loader import get_img_builder

from clevr.data_provider.clevr_tokenizer import CLEVRTokenizer
from clevr.models.clever_network_factory import create_network


from clevr.data_provider.clevr_dataset import Image
from generic.data_provider.nlp_utils import padder

import hashlib
import json

##############################
#  Usage
##############################
# python src/clevr/run/run_clevr.py -config config/clevr/film.json -data_dir data/CLEVR_v1.0 -img_dir data/CLEVR_v1.0 -load_checkpoint out/clevr/run/params.ckpt


###############################
#  LOAD CONFIG
#############################

parser = argparse.ArgumentParser('CLEVR network baseline!')

parser.add_argument("-data_dir", type=str, help="Directory with data")
parser.add_argument("-img_dir", type=str, help="Directory with image")
parser.add_argument("-load_checkpoint", type=str, help="Load model parameters from specified checkpoint")
parser.add_argument("-config", type=str, help='Config file')
parser.add_argument("-gpu_ratio", type=float, default=0.95, help="How many GPU ram is required? (ratio)")

args = parser.parse_args()

# Load config
with open(args.config, 'rb') as f_config:
    config_str = f_config.read()
    exp_identifier = hashlib.md5(config_str).hexdigest()
    config = json.loads(config_str.decode('utf-8'))

# Load images
image_builder = get_img_builder(config['model']['image'], args.img_dir, bufferize=None)

# Load dictionary
tokenizer = CLEVRTokenizer(os.path.join(args.data_dir, config["dico_name"]))

# Building the network
network = create_network(
    config=config["model"],
    no_words=tokenizer.no_words,
    no_answers=tokenizer.no_answers,
    reuse=False, device=0)

# create a saver to store/load checkpoint
saver = tf.train.Saver()

# TODO  : Investiate if using another option than the per_process help with gpu_raio problems
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_ratio)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:

    # Load checkpoints or pre-trained networks
    sess.run(tf.global_variables_initializer())

    # Set the weight of the network according to the provided params.ckpt
    saver.restore(sess, args.load_checkpoint)

    testImage = {
        'image_filename': 'CLEVR_val_002770.png',
        "image_id": 2770,
        "answer": "no",
        'set':'val',
        "question": "Are there any other things that have the same color as the block?",
        "question_index": -1
    }

    question = tokenizer.encode_question(testImage['question'])

    # Get answers
    answer = tokenizer.encode_answer(testImage['answer'])

    question, seq_length = padder([question], padding_symbol=tokenizer.padding_token)

    feed_dict = {
        'clevr/question:0':question,
        'clevr/image:0' : [Image(testImage['image_id'], testImage['image_filename'], image_builder, testImage['set']).get_image()],
        'clevr/is_training:0': False,
        'clevr/seq_length:0' : seq_length
    }

    # Run the session
    networkOutput = sess.run(network.prediction, feed_dict=feed_dict)

    print("Image Id : "+str(testImage["image_id"]))
    print("Question asked : "+testImage['question'])
    print("Answer received : "+tokenizer.decode_answer(networkOutput[0]))
    print("Ground truth : "+testImage['answer'])


