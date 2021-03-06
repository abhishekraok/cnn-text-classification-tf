from unittest import TestCase
from model_training import train_cnn
from training_flags import TrainingFlags
from tensorflow.contrib import learn
import numpy as np
import os
import shutil
from data_helpers import load_from_tsv, load_data_and_labels
from sklearn.model_selection import train_test_split
from random_parameter_generator import generate_parameter
from sentence_data import SentenceData


class TestCNN(TestCase):
    def test_cnn_initialized(self):
        initial_folder = 'test_out_initial'
        options = TrainingFlags(summaries_folder=initial_folder)
        x_text = np.array(['haha', 'hoho how are you'])
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length=100)
        x_train = x_dev = np.array(list(vocab_processor.fit_transform(x_text)))
        y_train = y_dev = np.array([[0, 1], [1, 0]])
        train_cnn(flags=options, x_train=x_train, y_train=y_train, x_dev=x_dev, y_dev=y_dev,
                  vocab_processor=vocab_processor)
        cleanup()
        shutil.rmtree(initial_folder, ignore_errors=True)

    def test_cnn_word_vector(self):
        wv_summ = 'test_out_wv'
        options = TrainingFlags(enable_word_embeddings=True, pretrained_embedding='small_100.vec', embedding_dim=100,
                                summaries_folder=wv_summ)
        x_text = np.array(['haha', 'hoho how are you'])
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length=100)
        x_train = x_dev = np.array(list(vocab_processor.fit_transform(x_text)))
        y_train = y_dev = np.array([[0, 1], [1, 0]])
        train_cnn(flags=options, x_train=x_train, y_train=y_train, x_dev=x_dev, y_dev=y_dev,
                  vocab_processor=vocab_processor)
        cleanup()
        shutil.rmtree(wv_summ, ignore_errors=True)

    def test_load_tsv(self):
        tsv_file = 'data/test_data.tsv'
        x, y = load_from_tsv(tsv_file)
        self.assertEqual(len(x), 3)
        self.assertEqual(len(y), 3)
        self.assertListEqual(list(y[0]), [1, 0])
        self.assertListEqual(list(y[1]), [0, 1])

    def test_random_parameter(self):
        temp_filename = 'gen.txt'
        generate_parameter(temp_filename)
        if os.path.isfile(temp_filename):
            os.remove(temp_filename)

    # Long test
    def test_cnn_polarity(self):
        polarity_summaries = 'test_out_polarity'
        options = TrainingFlags(summaries_folder=polarity_summaries)
        polarity_positive_filename = 'data/rt-polaritydata/rt-polarity.pos'
        polartity_negative_filename = 'data/rt-polaritydata/rt-polarity.pos'
        x_text, y = load_data_and_labels(polarity_positive_filename, polartity_negative_filename)
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length=100)
        x = np.array(list(vocab_processor.fit_transform(x_text)))
        x_train, x_dev, y_train, y_dev = train_test_split(x, y)
        train_cnn(flags=options, x_train=x_train, y_train=y_train, x_dev=x_dev, y_dev=y_dev,
                  vocab_processor=vocab_processor)
        cleanup()
        shutil.rmtree(polarity_summaries, ignore_errors=True)

    def test_summary(self):
        sentence_data = SentenceData.load_from_tsv('data/test_data.tsv')
        sample_size, positive_count, positive_percentage = sentence_data.summary()
        self.assertEqual(sample_size, 3)
        self.assertEqual(positive_count, 2)
        self.assertEqual(round(positive_percentage), 67)
        cleanup()


def cleanup():
    default_vocab_filename = 'vocab'
    if os.path.isfile(default_vocab_filename):
        os.remove(default_vocab_filename)
    default_summaries_foldername = 'summaries'
    if os.path.isdir(default_summaries_foldername):
        shutil.rmtree(default_summaries_foldername, ignore_errors=True)
    if os.path.isdir('checkpoints'):
        shutil.rmtree('checkpoints', ignore_errors=True)
