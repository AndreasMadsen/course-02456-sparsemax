from tensorflow_python.sparsemax_tf_ops import sparsemax_op, sparsemax_loss_op
import tensorflow as tf


class SparsemaxRegression:
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
                tf.truncated_normal(
                    [input_size, output_size],
                    stddev=0.1, seed=random_state, dtype=dtype
                ),
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
            self._prediction = sparsemax_op(logits)

            # setup loss
            self._loss = tf.reduce_mean(
                sparsemax_loss_op(logits, self._prediction, self.t)
            ) + regualizer * (tf.nn.l2_loss(W) + tf.nn.l2_loss(b))

            # setup train function
            self._train = tf.train.GradientDescentOptimizer(
                learning_rate
            ).minimize(self._loss)

            # setup error function
            self._error = tf.reduce_mean(
                tf.cast(tf.not_equal(
                    tf.argmax(self._prediction, 1),
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
