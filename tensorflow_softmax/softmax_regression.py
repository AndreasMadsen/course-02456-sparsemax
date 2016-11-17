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
    def __init__(self, input_size, output_size, observations=None,
                 regualizer=1, learning_rate=1e-2,
                 random_state=None, dtype=tf.float64):
        self.name = 'softmax - tensorflow'
        self.fast = observations is not None

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

            # setup init op
            self._reset = tf.initialize_all_variables()

            # setup model
            logits = tf.matmul(x, W) + b
            self._prediction = tf.nn.softmax(logits, name=None)

            # setup loss
            self._loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits, t)
            ) + regualizer * (tf.nn.l2_loss(W) + tf.nn.l2_loss(b))

            # setup train function
            self._train = tf.train.GradientDescentOptimizer(
                learning_rate
            ).minimize(self._loss)

            # setup error function
            self._error = tf.reduce_mean(
                tf.cast(tf.not_equal(
                    tf.argmax(self._prediction, 1),
                    tf.argmax(t, 1)
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
