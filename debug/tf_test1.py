import numpy as np
import tensorflow as tf


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(feature0, feature1, feature2, feature3):
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {
        'feature0': _int64_feature(feature0),
        'feature1': _int64_feature(feature1),
        'feature2': _bytes_feature(feature2),
        'feature3': _float_feature(feature3),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    # return  example_proto
    return example_proto.SerializeToString()


n_observations = int(1e4)
feature0 = np.random.choice([False, True], n_observations)
feature1 = np.random.randint(0, 5, n_observations)
strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
feature2 = strings[feature1]
feature3 = np.random.randn(n_observations)

# serialized_example = serialize_example(False, 4, b'goat', 0.9876)
# print(serialized_example)

filename = 'test1.tfrecord'
# with tf.io.TFRecordWriter(filename) as writer:
#     for i in range(n_observations):
#         example = serialize_example(feature0[i], feature1[i], feature2[i], feature3[i])
#         writer.write(example)

filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)
print(raw_dataset)