
import tensorflow as tf
import kernel

with tf.Session():
    var = tf.placeholder(tf.int32, name='x')
    data = [[1, 2], [3, 4]]

    print('forward: ')
    square = kernel.square(var)
    print(square.eval({var: data}))
    print()

    print('shape: ')
    print(kernel.square(data).get_shape())
    print(tf.shape(square).eval({var: data}))  # this evaluates via c++
    print()

    print('gradient: ')
    var_grad = tf.gradients(square, [var])[0]
    print(var_grad.eval({var: data}))
    print()
