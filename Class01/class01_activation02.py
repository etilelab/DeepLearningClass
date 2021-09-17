import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow._api.v2.math import exp, maximum

x = tf.random.normal(shape=(1,5))

dense_sigmoid = Dense(units=1, activation='sigmoid')  # affine func -> activation func 통과까지 해줌
dense_tanh = Dense(units=1, activation='tanh')  # affine func -> activation func 통과까지 해줌
dense_relu = Dense(units=1, activation='relu')  # affine func -> activation func 통과까지 해줌

# forward propagation
y_sigmoid = dense_sigmoid(x)
y_tanh = dense_tanh(x)
y_relu = dense_relu(x)


W, b = dense_sigmoid.get_weights()
z = tf.linalg.matmul(x, W) + b
a = 1/ (1+exp(-z))

print("Activation value Tensorflow {}".format(y_sigmoid))
print("Activation value Manual {}".format(a))
