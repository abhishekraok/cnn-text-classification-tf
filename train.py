"""
Originally from
https://github.com/dennybritz/cnn-text-classification-tf
Modified by Abhishek Rao
"""
import numpy as np
import tensorflow as tf
import yaml
from tensorflow.contrib import learn

import data_helpers
from trainer import train_cnn

tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos",
                       "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg",
                       "Data source for the negative data.")
tf.flags.DEFINE_string("output_dir", "output", "Location of output")
tf.flags.DEFINE_string("pretrained_embedding", "", "Location of pretrained embedding (space separated, glove format)")

# Model Hyperparameters
tf.flags.DEFINE_boolean("enable_word_embeddings", False,
                        "Enable/disable the pretrained word embedding (default: False)")
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("is_word2vec",0, "Whether the pre trained word vectors are word2vec "
                                         "binary format. default 0 = False")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0, "L2 regularization lambda (default: 0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 2, "Number of training epochs (default: 2)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("use_config", False, "Whether to read the config for settings")
tf.flags.DEFINE_float("decay_coefficient", 2.5, "Decay coefficient (default: 2.5)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

# Load data
print("Loading data...")
if FLAGS.use_config:
    dataset_name = cfg["datasets"]["default"]
    x_text, y = data_helpers.load_config_dataset(cfg=cfg, dataset_name=dataset_name)
else:
    x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
if dev_sample_index == 0:  # special case for reverse indexing
    dev_sample_index = len(y)
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

train_cnn(FLAGS, x_train, y_train, vocab_processor, x_dev, y_dev)
