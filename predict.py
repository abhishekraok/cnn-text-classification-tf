#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import data_helpers
from tensorflow.contrib import learn


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))


def predict_from_trained(x_id_list, prediction_flags):
    """
    Classifies sentences using trained model

    :param x_id_list: list of integer id's for each word
    """
    checkpoint_file = tf.train.latest_checkpoint(prediction_flags.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=prediction_flags.allow_soft_placement,
            log_device_placement=prediction_flags.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            scores = graph.get_operation_by_name("output/scores").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_id_list), prediction_flags.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []
            all_probabilities = None

            for x_id_list in batches:
                batch_predictions_scores = sess.run([predictions, scores],
                                                    {input_x: x_id_list, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions_scores[0]])
                probabilities = softmax(batch_predictions_scores[1])
                if all_probabilities is not None:
                    all_probabilities = np.concatenate([all_probabilities, probabilities])
                else:
                    all_probabilities = probabilities
    return all_predictions, all_probabilities


if __name__ == '__main__':
    # Parameters
    # ==================================================

    # Data Parameters
    tf.flags.DEFINE_string("positive_data_file", "", "Data source for the positive data.")
    tf.flags.DEFINE_string("negative_data_file", "", "Data source for the positive data.")
    tf.flags.DEFINE_string("tsv_data_file", "",
                           "TSV data source where first column is data and second is label. (Default: '')")

    # Eval Parameters
    tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")

    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    FLAGS = tf.flags.FLAGS
    tf.flags.DEFINE_string("prediction_file", os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv"),
                           "Predicted output")
    FLAGS._parse_flags()
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    if FLAGS.tsv_data_file is not "":
        x_raw = data_helpers.load_x_from_tsv(FLAGS.tsv_data_file)
    else:
        x_raw, _ = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

    # Map data into vocabulary
    vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_raw)))

    print("\nEvaluating...\n")

    predicted, probabilities = predict_from_trained(x_test, FLAGS)  # Save the evaluation to a csv
    predictions_human_readable = np.column_stack((predicted,
                                                  ["{}".format(probability[1]) for probability in probabilities],
                                                  np.array(x_raw)))
    output_filename = FLAGS.prediction_file
    print("Saving evaluation to {0}".format(output_filename))
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.writelines('{0}\t{1}\t{2}\n'.format(int(round(float(i[0]))), i[2], i[3]) for i in
                     predictions_human_readable)
