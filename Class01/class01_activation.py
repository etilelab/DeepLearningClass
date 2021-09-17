import tensorflow as tf
#from tensorflow.math import exp, maimum
from tensorflow._api.v2.math import exp, maximum
from tensorflow.keras.layers import Activation


x = tf.random.normal(shape=(1,5))

sigmoid = Activation('sigmoid')
tanh = Activation('tanh')
relu = Activation('relu')

y_sigmoid_tf = sigmoid(x)
y_tanh_tf = tanh(x)
y_relu_tf = relu(x)

print('Sigmoid : {}\n{}'.format(y_sigmoid_tf, y_sigmoid_tf.numpy()))
print('tanh : {}\n{}'.format(y_tanh_tf, y_tanh_tf.numpy()))
print('relu : {}\n{}'.format(y_relu_tf, y_relu_tf.numpy()))

y_sigmoid_man = 1 / (1+exp(-x)) # sigmoid
y_tanh_man = (exp(x)-exp(x))/(exp(x) + exp(-x))  # tanh
y_relu_man = maximum(x,0)
