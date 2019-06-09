import tensorflow as tf
import tensorflow.contrib.layers as tfc_layers

def create_optimizer(network, config, finetune=list(),
                     optim_cst=tf.train.AdamOptimizer, var_list=None,
                     apply_update_ops=True, loss=None):

    # Retrieve conf
    lrt = config['learning_rate']
    clip_val = config.get('clip_val', 0)
    weight_decay = config.get('weight_decay', 0)

    # create optimizer
    optimizer = optim_cst(learning_rate=lrt)

    # Extract trainable variables if not provided
    if var_list is None:
        var_list = network.get_parameters(finetune=finetune)

    # Apply weight decay
    if loss is None:
        loss = network.get_loss()

    # Apply weight decay
    training_loss = loss
    if weight_decay > 0:
        training_loss = loss + l2_regularization(var_list, weight_decay=weight_decay)


    # compute gradient
    grad = optimizer.compute_gradients(training_loss, var_list=var_list)

    # apply gradient clipping
    if clip_val > 0:
        grad = clip_gradient(grad, clip_val=clip_val)

    # add update ops (such as batch norm) to the optimizer call
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimize = optimizer.apply_gradients(grad)

    accuracy = network.get_accuracy()

    return optimize, [loss, accuracy]


def clip_gradient(gvs, clip_val):
    clipped_gvs = [(tf.clip_by_norm(grad, clip_val), var) for grad, var in gvs]
    return clipped_gvs


def l2_regularization(params, weight_decay):
    with tf.variable_scope("l2_normalization"):

        # Old code (faster but handcrafted)
        # l2_reg = [tf.nn.l2_loss(v) for v in params]
        # l2_reg = weight_decay * tf.add_n(l2_reg)

        weights_list = [v for v in params if v.name.endswith("weights:0")]
        regularizer = tfc_layers.l2_regularizer(scale=weight_decay)

        return tfc_layers.apply_regularization(regularizer, weights_list=weights_list)
