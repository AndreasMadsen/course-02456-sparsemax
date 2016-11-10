# sparsemax tensorflow implementation

This is an implementation of the sparsemax transformation presented in
https://arxiv.org/abs/1602.02068.

## implementations

The repository contains:

* a numpy-python implementation which works as an implementation reference.
* a tensorflow implementation that uses numpy for custom ops and tensorflow
graphs for the gradients.
* a tensorflow implementation that uses C++ for the custom ops. This where
most of our focus is going.

## API

The tensorflow sparsemax customs ops can be used as:

```python
from tensorflow_sparsemax import sparsemax_loss, sparsemax
import tensorflow as tf

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
logits = tf.matmul(x, W) + b
pred = sparsemax(logits)

# Cost
cost = tf.reduce_mean(sparsemax_loss(logits, pred, y))
```
