from unittest import TestCase
from trainer import train_cnn, TrainingFlags
from tensorflow.contrib import learn
import numpy as np
import os

class TestCNN(TestCase):
    def test_cnn_initialized(self):
        options = TrainingFlags()
        x_text = x_dev = np.array(['haha', 'hoho how are you'])
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length=100)
        x_train = x_dev = np.array(list(vocab_processor.fit_transform(x_text)))
        y_train = y_dev = np.array([[0, 1], [1, 0]])
        train_cnn(flags=options, x_train=x_train, y_train=y_train, x_dev=x_dev, y_dev=y_dev,
                  vocab_processor=vocab_processor)
        if os.path.isfile('vocab'):
            os.remove('vocab')

