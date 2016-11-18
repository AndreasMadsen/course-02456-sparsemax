import tensorflow as tf
import scipy
import math

from tensorflow_sparsemax.kernel import sparsemax, sparsemax_loss


def initializeW(input_size, output_size, random_state):
    W = scipy.stats.truncnorm.rvs(
            -2, 2, size=(input_size, output_size),
            random_state=random_state
        )
    W *= math.sqrt(2 / (input_size + output_size))
    return W


class SparsemaxRegression:
    def __init__(self, input_size, output_size, observations=None,
                 regualizer=1e-1, learning_rate=None, cpu_only=False,
                 random_state=None, dtype=tf.float64):
        self.name = 'TF Native'
        self.fast = observations is not None
        self.cpu_only = cpu_only

        self.graph = tf.Graph()

        with self.graph.as_default():
            # setup inputs
            x = self.x_init = self.x = tf.placeholder(
                dtype, [observations, input_size], name='x'
            )
            t = self.t_init = self.t = tf.placeholder(
                dtype, [observations, output_size], name='t'
            )

            if observations is not None:
                x = self.x_init = self.x_var = tf.Variable(
                    self.x, trainable=False, collections=[]
                )
                t = self.t_init = self.t_var = tf.Variable(
                    self.t, trainable=False, collections=[]
                )

            # setup variables
            W = tf.Variable(
                initializeW(input_size, output_size, random_state),
                name='W', dtype=dtype
            )

            b = tf.Variable(
                tf.zeros([output_size], dtype=dtype),
                name='b', dtype=dtype
            )

            # setup model
            logits = tf.matmul(x, W) + b
            self._prediction = sparsemax(logits)

            # setup loss
            self._loss = tf.reduce_mean(
                sparsemax_loss(logits, self._prediction, t)
            ) + regualizer * (tf.nn.l2_loss(W) + tf.nn.l2_loss(b))

            # setup train function
            self._train = tf.train.AdamOptimizer().minimize(self._loss)

            # setup error function
            self._error = tf.reduce_mean(
                tf.cast(tf.not_equal(
                    tf.argmax(self._prediction, 1),
                    tf.argmax(t, 1)
                ), dtype)
            )

            # setup init op
            self._reset = tf.initialize_all_variables()

    def __enter__(self):
        # create session and reset variables
        config = None
        if self.cpu_only:
            config = tf.ConfigProto(device_count={'GPU': 0})
        self._sess = tf.Session(graph=self.graph, config=config)
        self._sess.run(self._reset)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # close session
        self._sess.close()

    def reset(self):
        self._sess.run(self._reset)

    def update(self, inputs, targets, epochs=1):
        if self.fast and epochs > 1:
            self._sess.run([
                self.x_var.initializer,
                self.t_var.initializer
            ], feed_dict={
                self.x: inputs,
                self.t: targets
            })

            for _ in range(epochs):
                self._sess.run(self._train)
        else:
            for _ in range(epochs):
                self._sess.run(self._train, {
                    self.x_init: inputs,
                    self.t_init: targets
                })

    def loss(self, inputs, targets):
        return self._sess.run(self._loss, {
            self.x_init: inputs,
            self.t_init: targets
        })

    def predict(self, inputs):
        return self._sess.run(self._prediction, {
            self.x_init: inputs
        })

    def error(self, inputs, targets):
        return self._sess.run(self._error, {
            self.x_init: inputs,
            self.t_init: targets
        })
