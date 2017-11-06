#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import data_helpers
from tensorflow.contrib import learn
from predict import predict_from_trained

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
    tf.flags.DEFINE_boolean("calculate_pr", False, "Calculate Precision recall, pr curve")

    FLAGS = tf.flags.FLAGS
    tf.flags.DEFINE_string("prediction_file", os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv"),
                           "Predicted output")
    FLAGS._parse_flags()
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    if FLAGS.tsv_data_file is not "":
        x_raw, y_2column = data_helpers.load_from_tsv(FLAGS.tsv_data_file)
    else:
        x_raw, y_2column = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    y_test = np.argmax(y_2column, axis=1)

    # Map data into vocabulary
    vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_raw)))

    print("\nEvaluating...\n")

    all_predictions, all_probabilities = predict_from_trained(x_test, FLAGS)
    # Print accuracy if y_test is defined
    if y_test is not None:
        correct_predictions = float(sum(all_predictions == y_test))
        print("Total number of test examples: {}".format(len(y_test)))
        print("Accuracy: {:g}".format(correct_predictions / float(len(y_test))))

    # Save the evaluation to a csv
    predictions_human_readable = np.column_stack((all_predictions,
                                                  y_test,
                                                  ["{}".format(probability[1]) for probability in all_probabilities],
                                                  np.array(x_raw)))
    output_filename = FLAGS.prediction_file
    print("Saving evaluation to {0}".format(output_filename))
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.writelines('{0}\t{1}\t{2}\t{3}\n'.format(int(round(float(i[0]))), int(float(i[1])), i[2], i[3]) for i in
                     predictions_human_readable)

    if FLAGS.calculate_pr:
        from sklearn.metrics import classification_report, precision_recall_curve

        pr_report = classification_report(y_true=y_test, y_pred=all_predictions)
        with open(output_filename + '.pr.txt', 'w') as f:
            f.write(pr_report)
        print(pr_report)

        import matplotlib.pyplot as plt

        precision, recall, _ = precision_recall_curve(y_test, all_probabilities[:, 1])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve')
        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2,
                         color='b')
        plt.savefig(output_filename + 'prcurve.png')
