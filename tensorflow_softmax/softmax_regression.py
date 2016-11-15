from tensorflow_python.sparsemax_tf_ops import sparsemax_op, sparsemax_loss_op
import tensorflow as tf
import scipy
import math

def initializeW(input_size, output_size, random_state):
    W = scipy.stats.truncnorm.rvs(
            -2, 2, size=(input_size, output_size),
            random_state=random_state
        )
    W *= math.sqrt(2 / (input_size + output_size))
    return W


class SoftmaxRegression:
    def __init__(self, input_size, output_size,
                 regualizer=1, learning_rate=1e-2,
                 random_state=None, dtype=tf.float64):

        self.graph = tf.Graph()

        with self.graph.as_default():
            # setup inputs
            self.x = tf.placeholder(dtype, [None, input_size], name='x')
            self.t = tf.placeholder(dtype, [None, output_size], name='t')

            # setup variables
            W = tf.Variable(
                initializeW(input_size, output_size, random_state),
                name='W', dtype=dtype
            )
            b = tf.Variable(
                tf.zeros([output_size], dtype=dtype),
                name='b', dtype=dtype
            )

            # setup init op
            self._reset = tf.initialize_all_variables()

            # setup model
            logits = tf.matmul(self.x, W) + b
            self._prediction = tf.nn.softmax(logits, name=None)

            # setup loss
            self._loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits, self.t)
            ) + regualizer * (tf.nn.l2_loss(W) + tf.nn.l2_loss(b))

            # setup train function
            self._train = tf.train.GradientDescentOptimizer(
                learning_rate
            ).minimize(self._loss)

            # setup error function
            self._error = tf.reduce_mean(
                tf.cast(tf.not_equal(
                    tf.argmax(self._prediction,1),
                    tf.argmax(self.t, 1)
                ), dtype)
            )

    def __enter__(self):
        # create session and reset variables
        self._sess = tf.Session(graph=self.graph)
        self._sess.run(self._reset)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # close session
        self._sess.close()

    def reset(self):
        self._sess.run(self._reset)

    def update(self, inputs, targets):
        self._sess.run(self._train, {
            self.x: inputs,
            self.t: targets
        })

    def loss(self, inputs, targets):
        return self._sess.run(self._loss, {
            self.x: inputs,
            self.t: targets
        })

    def predict(self, inputs):
        return self._sess.run(self._prediction, {
            self.x: inputs
        })

    def error(self, inputs, targets):
        return self._sess.run(self._error, {
            self.x: inputs,
            self.t: targets
        })
