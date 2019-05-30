import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.resnet_v1 as resnet_v1
import tensorflow.contrib.slim.python.slim.nets.resnet_utils as slim_utils

from aqa.models.film_network import FiLM_Network
from aqa.data_interface.CLEAR_tokenizer import CLEARTokenizer


if __name__ == "__main__":

    # Paths
    experiment_name = "v1.0.0_10k_scenes_20_inst_per_scene"
    resnet_ckpt_path = "data/resnet/resnet_v1_101.ckpt"
    resnet_chosen_layer_scope = "resnet_v1_101/block3/unit_22/bottleneck_v1"
    dict_path = "data/%s/preprocessed/dict.json" % experiment_name

    film_model_config = {
        "image": {
            "dim": [14, 14, 1024],
        },
        "question": {
            "word_embedding_dim": 200,
            "rnn_state_size": 4096
        },
        "stem": {
            "spatial_location": True,
            "conv_out": 128,
            "conv_kernel": [3, 3]
        },
        "resblock": {
            "no_resblock": 4,
            "spatial_location": True,
            "kernel1": [1, 1],
            "kernel2": [3, 3]
        },
        "classifier": {
            "spatial_location": True,
            "conv_out": 512,
            "conv_kernel": [1, 1],
            "no_mlp_units": 1024
        }
    }

    # Input Image
    images = tf.placeholder(tf.float32, [None, 224, 224, 3], name='image')      # FIXME : would it be better to use a fixed batch_size instead of None ?

    # Import Resnet101
    with slim.arg_scope(slim_utils.resnet_arg_scope()):
        net, end_points = resnet_v1.resnet_v1_101(images, 1000, is_training=False)  # 1000 is the number of softmax class

    resnet_chosen_layer = end_points[resnet_chosen_layer_scope]

    # FIXME : The graph still have reference to layers after the chosen layer. Can we delete them ?
    # TODO : See in tensorboard what it look like

    # Adding the FiLM network after the chosen resnet layer
    tokenizer = CLEARTokenizer(dict_path)
    network = FiLM_Network(film_model_config, input_image_tensor=resnet_chosen_layer, no_words=tokenizer.no_words, no_answers=tokenizer.no_answers, device=0)  # FIXME : Not sure that device 0 is appropriate for CPU


    # GPU Options
    gpu_options = tf.GPUOptions(allow_growth=True)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())

        trainable_variables = tf.trainable_variables()

        # Restore the pretrained weight of resnet
        resnet_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="resnet_v1_101")
        resnet_saver = tf.train.Saver(var_list=resnet_variables)
        resnet_saver.restore(sess, resnet_ckpt_path)
        print("FatKid")
    #
    #     for one_set in set_type:
    #
    #         print("Load dataset -> set: {}".format(one_set))
    #         dataset_args["which_set"] = one_set
    #         dataset = dataset_cstor(**dataset_args)
    #
    #         # hack dataset to only keep one game by image
    #         image_id_set = {}
    #         games = []
    #         for game in dataset.games:
    #             if game.image.id not in image_id_set:
    #                 games.append(game)
    #                 image_id_set[game.image.id] = 1
    #
    #         dataset.games = games
    #         no_images = len(games)
    #
    #         source_name = os.path.basename(img_input.name[:-2])
    #         dummy_tokenizer = DummyTokenizer()
    #         batchifier = batchifier_cstor(tokenizer=dummy_tokenizer, sources=[source_name])
    #         iterator = Iterator(dataset,
    #                             batch_size=batch_size,
    #                             pool=cpu_pool,
    #                             batchifier=batchifier)
    #
    #         ############################
    #         #  CREATE FEATURES
    #         ############################
    #         print("Start computing image features...")
    #         filepath = os.path.join(out_dir, "{}_features.h5".format(one_set))
    #         with h5py.File(filepath, 'w') as f:
    #
    #             ft_shape = [int(dim) for dim in ft_output.get_shape()[1:]]
    #             ft_dataset = f.create_dataset('features', shape=[no_images] + ft_shape, dtype=np.float32)
    #             idx2img = f.create_dataset('idx2img', shape=[no_images], dtype=np.int32)
    #             pt_hd5 = 0
    #
    #             for batch in tqdm(iterator):
    #                 feat = sess.run(ft_output, feed_dict={img_input: numpy.array(batch[source_name])})
    #
    #                 # Store dataset
    #                 batch_size = len(batch["raw"])
    #                 ft_dataset[pt_hd5: pt_hd5 + batch_size] = feat
    #
    #                 # Store idx to image.id
    #                 for i, game in enumerate(batch["raw"]):
    #                     idx2img[pt_hd5 + i] = game.image.id
    #
    #                 # update hd5 pointer
    #                 pt_hd5 += batch_size
    #             print("Start dumping file: {}".format(filepath))
    #         print("Finished dumping file: {}".format(filepath))

