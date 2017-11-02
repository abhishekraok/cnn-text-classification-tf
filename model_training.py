import datetime
import math
import os
import data_helpers
from cnn_3_layer import CNN3Layer

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


def train_cnn(flags, x_train, y_train, vocab_processor, x_dev, y_dev):
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=flags.allow_soft_placement,
            log_device_placement=flags.log_device_placement)
        sess = tf.Session(config=session_conf)
        embedding_dimension = flags.embedding_dim
        with sess.as_default():
            cnn = CNN3Layer(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=embedding_dimension,
                filter_sizes=list(map(int, flags.filter_sizes.split(","))),
                num_filters=flags.num_filters,
                l2_reg_lambda=flags.l2_reg_lambda,
                device='/gpu:0',
                pre_trained_embeddings=flags.enable_word_embeddings,
                num_units_in_fcl=flags.fully_connected_units)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(cnn.learning_rate, beta1=flags.beta1, beta2=flags.beta2)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            out_dir = flags.output_dir
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, flags.summaries_folder, "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, flags.summaries_folder, "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=flags.num_checkpoints, save_relative_paths=True)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            if flags.enable_word_embeddings:
                vocabulary = vocab_processor.vocabulary_
                # load embedding vectors from the glove
                pre_trained_embedding_path = flags.pretrained_embedding
                print("Loading pre trained embedding from {}".format(pre_trained_embedding_path))
                if flags.is_word2vec == 0:
                    initW = data_helpers.load_embedding_vectors_glove(vocabulary,
                                                                      pre_trained_embedding_path,
                                                                      embedding_dimension)
                else:
                    initW = data_helpers.load_embedding_vectors_word2vec(vocabulary,
                                                                         pre_trained_embedding_path,
                                                                         binary=True)
                print("pre trained embedding file has been loaded\n")
                sess.run(cnn.W.assign(initW))

            def train_step(x_batch, y_batch, learning_rate):
                """
                A single training step
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: flags.dropout_keep_prob,
                    cnn.learning_rate: learning_rate
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                if step % 200 == 0:
                    print("{}: step {}, loss {:g}, acc {:g}, learning_rate {:g}"
                          .format(time_str, step, loss, accuracy, learning_rate))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), flags.batch_size, flags.num_epochs)
            # It uses dynamic learning rate with a high value at the beginning to speed up the training
            min_learning_rate = flags.min_learning_rate
            max_learning_rate = 50 * min_learning_rate
            decay_speed = flags.decay_coefficient * len(y_train) / flags.batch_size
            # Training loop. For each batch...
            for counter, batch in enumerate(batches):
                learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(
                    -counter / decay_speed)
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch, learning_rate)
                current_step = tf.train.global_step(sess, global_step)
                if (current_step % flags.evaluate_every == 0) and (len(x_dev) > 0) and (len(y_dev) > 0):
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % flags.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
