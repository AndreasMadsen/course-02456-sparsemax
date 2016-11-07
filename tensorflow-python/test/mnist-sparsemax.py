import _test  # In order to import modules from other directories
import argparse

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

import sparsemax_tf_ops


FLAGS = None


def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float64, [None, 784], name="input.x")
    W = tf.cast(tf.Variable(tf.zeros([784, 10])), tf.float64, name="weights")
    b = tf.cast(tf.Variable(tf.zeros([10])), tf.float64, name="biases")
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float64, [None, 10])

    spm = sparsemax_tf_ops.sparsemax_op(y)
    sparsemax_loss = tf.reduce_mean(
        sparsemax_tf_ops.sparsemax_loss_op(
            y, spm, y_))
    train_step = tf.train.GradientDescentOptimizer(
        0.5).minimize(sparsemax_loss)

    with tf.Session() as sess:
        # Train
        tf.initialize_all_variables().run()
        for _ in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        # Test trained model
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print("Accuracy: ")
        print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                            y_: mnist.test.labels}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/data',
                        help='Directory for storing data')
    FLAGS = parser.parse_args()
    tf.app.run()
