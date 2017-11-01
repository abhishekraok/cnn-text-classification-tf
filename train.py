"""
Originally from
https://github.com/dennybritz/cnn-text-classification-tf
Modified by Abhishek Rao
"""
import numpy as np
import tensorflow as tf
import yaml
import time
from tensorflow.contrib import learn

import data_helpers
from model_training import train_cnn

tf.flags.DEFINE_float("dev_sample_percentage", .1,
                      "Ratio of the training data to use for validation (Default: 10%=0.1)")
tf.flags.DEFINE_string("positive_data_file", "", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "", "Data source for the negative data.")
tf.flags.DEFINE_string("tsv_data_file", "",
                       "TSV data source where first column is data and second is label. (Default: '')")
tf.flags.DEFINE_string("output_dir", "output", "Location of output")
tf.flags.DEFINE_string("pretrained_embedding", "", "Location of pretrained embedding (space separated, glove format)")

# Model Hyperparameters
tf.flags.DEFINE_boolean("enable_word_embeddings", False,
                        "Enable/disable the pretrained word embedding (default: False)")
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("is_word2vec", 0, "Whether the pre trained word vectors are word2vec "
                                          "binary format. default 0 = False")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0, "L2 regularization lambda (default: 0)")
tf.flags.DEFINE_float("min_learning_rate", "0.0001", "Minimum learning rate (alpha) (default 0.0001). "
                                                     "Max is 50 times this.")
tf.flags.DEFINE_float("beta1", 0.9, "Adam param beta1, first momentum (default: 0.9)")
tf.flags.DEFINE_float("beta2", 0.999, "Adam param beta2, second momentum (default: 0.999)")
tf.flags.DEFINE_float("decay_coefficient", 2.5, "Decay coefficient (default: 2.5)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 2, "Number of training epochs (default: 2)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("fully_connected_units", 64,
                        "Number of units in the fully connected layer, after convolution filters (default: 64)")
tf.flags.DEFINE_integer("min_frequency", 5,
                        "Minimum number of times for a word to occur to be considered (default: 5)")  # Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Load data
print("Loading data...")
if FLAGS.tsv_data_file is not "":
    x_text, y = data_helpers.load_from_tsv(FLAGS.tsv_data_file)
else:
    x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, min_frequency=FLAGS.min_frequency)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
if dev_sample_index == 0:  # special case for reverse indexing
    dev_sample_index = len(y)
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

start_time = time.time()
train_cnn(FLAGS, x_train, y_train, vocab_processor, x_dev, y_dev)
end_time = time.time()
print('Total time for training {0} minutes'.format((end_time - start_time) / 60))
