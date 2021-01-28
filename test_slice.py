import numpy as np
import tensorflow as tf
from training_net import get_anchors, apply_box_deltas_graph
from config import Config


class Anchors(tf.keras.layers.Layer):
    def __init__(self, cfg):
        super(Anchors, self).__init__()
        self.cfg = cfg

    def call(self, inputs, **kwargs):
        anchor = get_anchors(self.cfg, self.cfg.IMAGE_SHAPE)
        anchor = np.broadcast_to(anchor, (self.cfg.BATCH_SIZE,) + anchor.shape)
        return tf.Variable(anchor)


def batch_slice(inputs, graph_fn, batch_size, names=None):
    """Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.

    inputs: list of tensors. All must have the same first dimension length
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        print('slice', inputs_slice)
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result


def balanced_sample(flags, proposal, gt_box):
    positive = tf.where(tf.greater_equal(flags, 0))
    negative = tf.where(tf.equal(flags, 0))
    print('tf.shape(positive)', tf.shape(positive))

    positive = tf.squeeze(positive, -1)
    negative = tf.squeeze(negative, -1)
    print('tf.shape(positive)', tf.shape(positive))

    num_pos = 256
    # protect against not enough positive examples
    num_pos = tf.minimum(tf.size(positive), num_pos)
    num_neg = 512 - num_pos
    # protect against not enough negative examples
    num_neg = tf.minimum(tf.size(negative), num_neg)

    # randomly select positive and negative examples
    # print(num_pos)
    pos_idx_per_image = tf.slice(tf.random.shuffle(positive), [0], [num_pos])
    neg_idx_per_image = tf.slice(tf.random.shuffle(negative), [0], [num_neg])

    idx_per_image = tf.concat([pos_idx_per_image, neg_idx_per_image], axis=-1)
    labels_sampled = tf.gather(flags, idx_per_image)

    proposal_sampled = tf.gather(proposal, idx_per_image)
    gt_box_sampled = tf.gather(gt_box, idx_per_image)
    # reg_target_sampled = cal_reg_target(gt_box_sampled, proposal_sampled)
    print(labels_sampled)
    return labels_sampled


class ProposalLayer(tf.keras.layers.Layer):

    def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold

    def call(self, inputs):
        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = inputs[0][:, :, 1]
        # Box deltas [batch, num_rois, 4]
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])
        # Anchors
        anchors = inputs[2]

        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True, name="top_anchors").indices
        print('dd', ix.shape, scores.shape)
        scores = batch_slice([scores, ix], lambda x, y: tf.gather(x, y), self.config.IMAGES_PER_GPU)
        deltas = batch_slice([deltas, ix], lambda x, y: tf.gather(x, y), self.config.IMAGES_PER_GPU)

        pre_nms_anchors = batch_slice([anchors, ix], lambda a, x: tf.gather(a, x),
                                      self.config.IMAGES_PER_GPU,
                                      names=["pre_nms_anchors"])
        print('scores.shape', scores.shape)
        print('deltas.shape', deltas.shape)
        print('pre_nms_anchors.shape', pre_nms_anchors.shape)
        # Apply deltas to anchors to get refined anchors.
        # [batch, N, (y1, x1, y2, x2)]
        boxes = batch_slice([scores, pre_nms_anchors, deltas],
                            lambda x, y, z: balanced_sample(x, y, z),
                            self.config.IMAGES_PER_GPU,
                            names=["refined_anchors"])

        return boxes


input_image = tf.keras.layers.Input(shape=[None, None, 3], name="input_image")
input_score = tf.keras.layers.Input(shape=[None, 2], name="input_image")
input_delta = tf.keras.layers.Input(shape=[None, 4], name="input_image")

config = Config()
config.display()
anchors = Anchors(config)(input_image)

p = ProposalLayer(1000, 0.5, config)
out = p([input_score, input_delta, anchors])
