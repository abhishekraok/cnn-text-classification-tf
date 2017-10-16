from unittest import TestCase
from trainer import train_cnn, TrainingFlags
from tensorflow.contrib import learn
import numpy as np
import os
import shutil
from data_helpers import load_from_tsv, load_data_and_labels
from sklearn.model_selection import train_test_split


class TestCNN(TestCase):
    def test_cnn_initialized(self):
        options = TrainingFlags()
        x_text = np.array(['haha', 'hoho how are you'])
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length=100)
        x_train = x_dev = np.array(list(vocab_processor.fit_transform(x_text)))
        y_train = y_dev = np.array([[0, 1], [1, 0]])
        train_cnn(flags=options, x_train=x_train, y_train=y_train, x_dev=x_dev, y_dev=y_dev,
                  vocab_processor=vocab_processor)
        cleanup()

    # Long test
    def test_cnn_polarity_good_metrics(self):
        options = TrainingFlags()
        polarity_positive_filename = 'data/rt-polaritydata/rt-polarity.pos'
        polartity_negative_filename = 'data/rt-polaritydata/rt-polarity.pos'
        x_text, y = load_data_and_labels(polarity_positive_filename, polartity_negative_filename)
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length=100)
        x = np.array(list(vocab_processor.fit_transform(x_text)))
        x_train, x_dev, y_train, y_dev = train_test_split(x, y)
        train_cnn(flags=options, x_train=x_train, y_train=y_train, x_dev=x_dev, y_dev=y_dev,
                  vocab_processor=vocab_processor)
        cleanup()

    def test_cnn_word_vector(self):
        options = TrainingFlags(enable_word_embeddings=True, pretrained_embedding='small_100.vec', embedding_dim=100)
        x_text = np.array(['haha', 'hoho how are you'])
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length=100)
        x_train = x_dev = np.array(list(vocab_processor.fit_transform(x_text)))
        y_train = y_dev = np.array([[0, 1], [1, 0]])
        train_cnn(flags=options, x_train=x_train, y_train=y_train, x_dev=x_dev, y_dev=y_dev,
                  vocab_processor=vocab_processor)
        cleanup()

    def test_load_tsv(self):
        tsv_file = 'data/test_data.tsv'
        x, y = load_from_tsv(tsv_file)
        self.assertEqual(len(x), 2)
        self.assertEqual(len(y), 2)
        self.assertListEqual(list(y[0]), [1, 0])
        self.assertListEqual(list(y[1]), [0, 1])


def cleanup():
    default_vocab_filename = 'vocab'
    if os.path.isfile(default_vocab_filename):
        os.remove(default_vocab_filename)
    default_summaries_foldername = '/summaries/*'
    if os.path.isdir(default_summaries_foldername):
        shutil.rmtree(default_summaries_foldername, ignore_errors=True)
    if os.path.isdir('checkpoints'):
        shutil.rmtree('checkpoints', ignore_errors=True)
