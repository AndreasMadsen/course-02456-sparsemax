import tensorflow as tf
import numpy as np

import kernel


def sparsemax(z):
    logits = tf.placeholder(tf.float64, name='z')
    sparsemax = kernel.sparsemax(logits)

    with tf.Session() as sess:
        return sparsemax.eval({logits: z})


def test_sparsemax():
    """check sparsemax proposition 1, part 1"""
    z = np.array([[4, 3, 2, 5, 123, 100],
                  [7, 6, 1, 8, 90, 78],
                  [0, 3, 2, 4, 87, 12]],dtype=np.float64)
    x = sparsemax(z)
    print(x)

test_sparsemax()

